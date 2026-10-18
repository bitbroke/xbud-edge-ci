"""Deterministic lifecycle tests: fake IPC/processes, no model downloads."""
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import pytest
import providers.asr_worker as module


class Channel:
    def __init__(self):
        self.items = queue.Queue()
        self.closed = False
        self.cancelled = False
        self.on_put = None

    def put(self, item):
        if self.on_put:
            self.on_put(item)
        else:
            self.items.put(item)

    def get(self, timeout):
        return self.items.get(timeout=timeout)

    def cancel_join_thread(self):
        self.cancelled = True

    def close(self):
        self.closed = True


class Process:
    def __init__(self, ctx, args):
        self.ctx, self.req, self.resp = ctx, args[0], args[1]
        self.pid = None
        self.exitcode = None
        self.alive = False
        self.closed = False
        self.joins = 0
        self.killed = False

    def start(self):
        if self.ctx.mode == "spawn_error":
            raise OSError("spawn failure")
        self.pid = len(self.ctx.processes) + 100
        self.alive = True
        if self.ctx.mode == "startup_crash":
            self.alive, self.exitcode = False, 7
        elif self.ctx.mode == "init_error":
            self.resp.put({"status": "error", "error": "bad model"})
        elif self.ctx.mode != "startup_timeout":
            self.resp.put({"status": "ready"})
        self.req.on_put = self.respond

    def respond(self, req):
        if self.ctx.mode == "inference_timeout":
            return
        if self.ctx.mode == "inference_crash":
            self.alive, self.exitcode = False, 9
            return
        if self.ctx.mode == "slow":
            time.sleep(.06)
        self.resp.put({
            "status": "ready" if self.ctx.mode == "late_ready" else "ok",
            "request_id": "stale" if self.ctx.mode == "stale" else req["request_id"],
            "text": req["wav_path"], "audio_duration": 1, "error": None,
        })

    def is_alive(self):
        return self.alive

    def terminate(self):
        if not self.ctx.resist_terminate:
            self.alive, self.exitcode = False, -15

    def kill(self):
        self.killed = True
        self.alive, self.exitcode = False, -9

    def join(self, timeout):
        self.joins += 1

    def close(self):
        assert not self.alive
        self.closed = True


class Context:
    def __init__(self):
        self.mode = "ok"
        self.resist_terminate = False
        self.channels = []
        self.processes = []

    def Queue(self):
        channel = Channel()
        self.channels.append(channel)
        return channel

    def Process(self, target, args, daemon):
        proc = Process(self, args)
        self.processes.append(proc)
        return proc


@pytest.fixture
def adapter(monkeypatch):
    obj = module.SherpaOnnxASR.__new__(module.SherpaOnnxASR)
    obj.model_dir = "fixture"
    obj.language = "en"
    obj.model_id = "sherpa-test"
    obj.strict_real = True
    obj.is_mock = False
    obj.num_threads = 1
    obj._proc = obj._req_queue = obj._resp_queue = None
    obj._ready = False
    obj._lifecycle_lock = threading.RLock()
    obj.startup_timeout_s = .04
    obj._ctx = Context()
    monkeypatch.setattr(module, "psutil", None)
    yield obj
    obj._kill_worker()


@pytest.mark.parametrize("mode,phase,timed_out", [
    ("startup_timeout", "startup", True),
    ("startup_crash", "startup", False),
    ("init_error", "startup", False),
    ("spawn_error", "startup", False),
    ("inference_timeout", "inference", True),
    ("inference_crash", "inference", False),
    ("late_ready", "inference", False),
    ("stale", "inference", False),
])
def test_failure_cleans_and_recovers(adapter, mode, phase, timed_out):
    adapter._ctx.mode = mode
    started = time.monotonic()
    result = adapter.transcribe("first.wav", timeout=.08)
    assert time.monotonic() - started < .5
    assert result["error"] and result["text"] == ""
    assert result["timeout"] is timed_out
    assert result["failure_phase"] == phase
    assert adapter._proc is None and not adapter._ready
    assert all(p.closed and not p.alive for p in adapter._ctx.processes)
    assert all(q.closed and q.cancelled for q in adapter._ctx.channels)
    assert not adapter.is_mock
    adapter._ctx.mode = "ok"
    result = adapter.transcribe("second.wav", timeout=.5)
    assert result["error"] is None and result["text"].endswith("second.wav")
    assert adapter._ready


def test_live_unready_is_not_reused(adapter):
    adapter._ensure_worker()
    old = adapter._proc
    adapter._ready = False
    adapter._ensure_worker()
    assert old.closed and old is not adapter._proc


def test_dead_worker_reaped_before_replacement(adapter):
    adapter._ensure_worker()
    old = adapter._proc
    old.alive = False
    result = adapter.transcribe("next.wav")
    assert result["error"] is None and old.closed and old.joins


def test_runtime_failure_never_enables_mock_fallback(adapter):
    adapter.strict_real = False
    adapter._ctx.mode = "init_error"
    result = adapter.transcribe("a.wav")
    assert result["error"] and not adapter.is_mock and result["text"] == ""


def test_unreapable_worker_blocks_replacement(adapter, monkeypatch):
    adapter._ensure_worker()
    old = adapter._proc
    adapter._ready = False
    adapter._ctx.resist_terminate = True
    with monkeypatch.context() as patch:
        patch.setattr(old, "kill", lambda: None)
        result = adapter.transcribe("a.wav")
        assert result["error"].startswith("CleanupError:")
        assert adapter._proc is old and len(adapter._ctx.processes) == 1
    adapter._ctx.resist_terminate = False


def test_kill_is_joined_and_cleanup_idempotent(adapter):
    adapter._ensure_worker()
    proc = adapter._proc
    adapter._ctx.resist_terminate = True
    adapter._kill_worker()
    adapter._kill_worker()
    assert proc.killed and proc.joins == 2 and proc.closed


def test_partial_queue_creation_is_cleaned(adapter, monkeypatch):
    original = adapter._ctx.Queue
    def fail_second():
        if adapter._ctx.channels:
            raise OSError("queue allocation failed")
        return original()
    monkeypatch.setattr(adapter._ctx, "Queue", fail_second)
    result = adapter.transcribe("a.wav")
    assert result["error"]
    assert adapter._ctx.channels[0].closed
    assert adapter._req_queue is None


def test_concurrent_requests_keep_their_responses(adapter):
    adapter._ctx.mode = "slow"
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda i: adapter.transcribe(f"{i}.wav"), range(3)))
    assert all(r["error"] is None and r["text"].endswith(f"{i}.wav") for i, r in enumerate(results))
    assert len(adapter._ctx.processes) == 1


def test_lock_deadline_does_not_kill_other_request(adapter):
    adapter._ctx.mode = "slow"
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(adapter.transcribe, "owner.wav")
        deadline = time.monotonic() + 1
        while not adapter._ready and time.monotonic() < deadline:
            time.sleep(.001)
        second = adapter.transcribe("waiter.wav", timeout=.001)
        assert second["timeout"] and second["failure_phase"] == "queue_wait"
        assert first.result()["error"] is None
    assert adapter._proc.is_alive()


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_invalid_deadline(adapter, timeout):
    with pytest.raises(ValueError):
        adapter.transcribe("a.wav", timeout=timeout)
