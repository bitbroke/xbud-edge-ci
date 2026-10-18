"""Real local Sherpa timeout/restart stress, immutable sources and retained evidence."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import threading
import time
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import providers.asr_worker as provider
REAL_TARGET = provider._sherpa_inference_worker


def delayed_start(*args):
    time.sleep(2)
    REAL_TARGET(*args)


def digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cycles", type=int, default=30)
    parser.add_argument("--precision", choices=["int8", "float"], default="int8")
    args = parser.parse_args()
    if args.cycles < 1:
        raise ValueError("cycles must be positive")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    corpus = ROOT / "datasets/arm64_eval_corpus"
    clips = json.loads((corpus/"general_speech_manifest.json").read_text())[:2]
    hashes = {str(corpus/c["file"]): c["audio_sha256"] for c in clips}
    for path, expected in hashes.items():
        assert digest(path) == expected
    source = Path(args.model_dir)
    suffix = ".int8.onnx" if args.precision == "int8" else ".onnx"
    files = ["tokens.txt"] + [f"{n}-epoch-99-avg-1{suffix}" for n in ("encoder", "decoder", "joiner")]
    model_hashes = {name: digest(source/name) for name in files}
    parent = psutil.Process()
    report = {"status": "RUNNING", "platform": platform.platform(),
              "precision": args.precision, "cycles_requested": args.cycles,
              "model_sha256": model_hashes,
              "provider_sha256": digest(ROOT/"providers/asr_worker.py"),
              "script_sha256": digest(Path(__file__)),
              "policy": "Real model; 10s total request deadline and unchanged 10s startup cap; no automatic retry or mock fallback.",
              "events": []}
    def save():
        (out/"report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    def record(kind, result, elapsed):
        row = {"kind": kind, "wall_s": elapsed, "result": result,
               "live_children": [p.pid for p in mp.active_children()],
               "parent_handles": parent.num_handles() if hasattr(parent, "num_handles") else parent.num_fds(),
               "parent_rss": parent.memory_info().rss,
               "threads": threading.active_count()}
        report["events"].append(row)
        save()
        print(f"{kind}: {elapsed:.3f}s error={result.get('error')} children={len(row['live_children'])}", flush=True)
    asr = None
    try:
        with tempfile.TemporaryDirectory(prefix="xbud-sherpa-stress-") as temp:
            for name in files:
                shutil.copy2(source/name, Path(temp)/name)
            asr = provider.SherpaOnnxASR(temp, strict_real=True, num_threads=4)
            assert not asr.is_mock
            def call(kind, index=0, timeout=10):
                start = time.monotonic()
                res = asr.transcribe(str(corpus/clips[index]["file"]), timeout=timeout)
                elapsed = time.monotonic()-start
                record(kind, res, elapsed)
                assert elapsed <= timeout + 2.5, "Deadline plus bounded cleanup exceeded"
                assert abs(res["duration_s"]-elapsed) < .15
                return res
            def success(kind, index=0):
                res = call(kind, index)
                assert res["error"] is None and not res["timeout"], res
                assert res["text"].strip().lower() == clips[index]["expected_transcript"].lower(), res
                assert asr._ready and len(mp.active_children()) == 1
            success("initial_real_inference")
            for i in range(args.cycles):
                old_pid = asr._proc.pid
                res = call(f"timeout_{i+1}", timeout=.0001)
                assert res["error"] == "Timeout" and res["timeout"] and res["text"] == ""
                assert res["failure_phase"] == "inference"
                assert asr._proc is None and not asr._ready
                assert not psutil.pid_exists(old_pid)
                assert not mp.active_children()
                success(f"recovery_{i+1}", i % 2)
            for i in range(3):
                old = asr._proc
                old.kill()
                old.join(timeout=1)
                success(f"external_kill_recovery_{i+1}", i % 2)
            for i in range(3):
                asr._kill_worker()
                provider._sherpa_inference_worker = delayed_start
                asr.startup_timeout_s = .1
                try:
                    res = call(f"injected_startup_timeout_{i+1}", timeout=.5)
                    assert res["timeout"] and res["failure_phase"] == "startup"
                    assert asr._proc is None and not mp.active_children()
                finally:
                    provider._sherpa_inference_worker = REAL_TARGET
                    asr.startup_timeout_s = 10
                success(f"startup_recovery_{i+1}")
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(asr.transcribe, str(corpus/clips[i % 2]["file"]), 10) for i in range(8)]
                for i, future in enumerate(futures):
                    res = future.result(timeout=15)
                    record(f"concurrent_{i+1}", res, res["duration_s"])
                    assert res["error"] is None and res["text"].lower() == clips[i % 2]["expected_transcript"].lower()
            asr._kill_worker()
            asr._kill_worker()
            assert not mp.active_children()
            time.sleep(.1)  # Allow cancelled queue feeder threads to finish.
            report["final_handles"] = parent.num_handles() if hasattr(parent, "num_handles") else parent.num_fds()
            report["final_threads"] = threading.active_count()
            recoveries = [e for e in report["events"] if e["kind"].startswith("recovery_")]
            report["recovery_handle_range"] = [min(e["parent_handles"] for e in recoveries), max(e["parent_handles"] for e in recoveries)]
            assert report["recovery_handle_range"][1] - report["recovery_handle_range"][0] <= 16, "Possible handle leak"
            assert report["final_threads"] <= 2, "Worker feeder threads remain"
            report["status"] = "PASS"
    except Exception as exc:
        report["status"] = "FAIL"
        report["failure"] = repr(exc)
        raise
    finally:
        provider._sherpa_inference_worker = REAL_TARGET
        if asr is not None:
            asr._kill_worker()
        report["source_hashes_unchanged"] = all(digest(p) == h for p, h in hashes.items())
        report["model_hashes_unchanged"] = all(digest(source/n) == h for n, h in model_hashes.items())
        if not report["source_hashes_unchanged"] or not report["model_hashes_unchanged"]:
            report["status"] = "FAIL"
        save()
    assert report["status"] == "PASS", report


if __name__ == "__main__":
    main()
