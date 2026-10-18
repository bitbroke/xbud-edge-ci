"""Paired local ASR benchmark. Never modifies source audio or downloads weights."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import threading
import time
import unicodedata
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import psutil
import soundfile as sf
from benchmarks.evaluation_metrics import compute_wer_cer_metrics, normalize_text
from providers.asr_worker import SherpaOnnxASR, WhisperCppASR


def sha(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def lexical(text):
    # Secondary diagnostic only; retain apostrophes, no number verbalization.
    return normalize_text("".join(" " if c == "-" else "" if unicodedata.category(c).startswith("P") and c != "'" else c for c in text))


def measure(call):
    stop = threading.Event()
    peak = [0]
    def poll():
        while not stop.is_set():
            parent = psutil.Process()
            total = 0
            for p in [parent] + parent.children(recursive=True):
                try:
                    total += p.memory_info().rss
                except psutil.Error:
                    pass
            peak[0] = max(peak[0], total)
            stop.wait(.01)
    thread = threading.Thread(target=poll, daemon=True)
    thread.start()
    start = time.perf_counter()
    try:
        result = call()
    except Exception as exc:
        result = {"text": "", "error": repr(exc), "timeout": False}
    finally:
        elapsed = time.perf_counter() - start
        stop.set()
        thread.join(timeout=2)
    return result, elapsed, peak[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", default="C:/Ankshit/Project_xbud1.00.0")
    p.add_argument("--corpus", default=str(ROOT / "datasets/arm64_eval_corpus"))
    p.add_argument("--output", required=True)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--timeout", type=float, default=45)
    p.add_argument("--peak-normalize", type=float, default=0,
                   help="Diagnostic only: scale temporary copies to this peak (0 disables)")
    p.add_argument("--sherpa-precision", choices=['int8', 'float'], default='int8')
    p.add_argument("--engines", choices=['sherpa', 'whisper'], required=True,
                   help='One engine per invocation: prevents idle-engine RAM contamination')
    p.add_argument('--whisper-cli')
    p.add_argument('--whisper-model')
    p.add_argument("--safe-normalize-sherpa", action='store_true')
    args = p.parse_args()
    project, corpus, output = Path(args.project), Path(args.corpus), Path(args.output)
    if args.safe_normalize_sherpa and args.peak_normalize:
        raise ValueError('Choose bounded speech gain OR peak diagnostic, not both')
    from audio.speech_normalization import BoundedSpeechNormalizer
    if args.safe_normalize_sherpa and args.engines != 'sherpa':
        raise ValueError('Speech normalization is a Sherpa-only experiment')
    normalizer = BoundedSpeechNormalizer(project/'models/silero_vad.onnx') if args.safe_normalize_sherpa else None
    if normalizer is not None and normalizer._vad is None:
        raise RuntimeError('Requested normalization requires real neural VAD')
    if args.peak_normalize:
        raise ValueError('Unbounded peak diagnostic is not supported by this release comparison')
    whisper_cli = Path(args.whisper_cli) if args.whisper_cli else project/'bin/main.exe'
    whisper_model = Path(args.whisper_model) if args.whisper_model else project/'models/ggml-base-latest.bin'
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = corpus / "human_speech_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if len(manifest) != 20 or len({x['audio_sha256'] for x in manifest}) != 20:
        raise ValueError("Expected frozen 20 unique human clips")
    if normalizer:
        control, rate = sf.read(corpus / manifest[6]['file'], dtype='float32')
        if not normalizer._neural_speech(control, rate):
            raise RuntimeError('Real neural VAD speech positive control failed')
    source_hashes = {}
    audio_stats = []
    for item in manifest:
        path = corpus / item['file']
        source_hashes[item['file']] = sha(path)
        assert source_hashes[item['file']] == item['audio_sha256']
        audio, sr = sf.read(path, dtype="float32")
        assert sr == 16000 and audio.ndim == 1
        audio_stats.append({"id": item['clip_id'], "seconds": len(audio)/sr,
                            "peak": float(np.max(np.abs(audio))),
                            "rms": float(np.sqrt(np.mean(audio**2))),
                            "clipped_fraction": float(np.mean(np.abs(audio) >= .999)),
                            "subtype": sf.info(path).subtype})
    metadata = {"platform": platform.platform(), "machine": platform.machine(),
                "cpu": platform.processor(), "logical_cpus": os.cpu_count(),
                "ram_bytes": psutil.virtual_memory().total, "python": sys.version,
                "dataset_sha256": sha(manifest_path), "started_utc": datetime.now(timezone.utc).isoformat(),
                "repeats": args.repeats, "timeout_s": args.timeout,
                "peak_normalize": args.peak_normalize, "sherpa_precision": args.sherpa_precision,
                "safe_normalize_sherpa": args.safe_normalize_sherpa,
                "vad_model_sha256": sha(project/'models/silero_vad.onnx') if normalizer else None,
                "packages": {x: importlib.metadata.version(x) for x in ('sherpa-onnx','jiwer','soundfile','numpy','psutil')},
                "code_sha256": {str(path.relative_to(ROOT)):sha(path) for path in [
                    Path(__file__), ROOT/'providers/asr_worker.py', ROOT/'benchmarks/evaluation_metrics.py',
                    ROOT/'audio/speech_normalization.py', ROOT/'audio/vad.py']},
                "timing_policy": "4 CPU threads each. Single engine per invocation. Sherpa persistent worker warmed once; Whisper CLI reloads model each call. Pipeline includes preprocessing. No TTS/playback. Parent plus recursive child RSS sampled every 10ms (sum may double-count shared pages; not exact peak). Cold startup reported separately.",
                "audio_stats": audio_stats, "model_hashes": {}}
    records = []
    try:
        with tempfile.TemporaryDirectory(prefix="xbud_paired_") as td:
            temp = Path(td)
            int8 = temp / "int8"
            int8.mkdir()
            model_src = project / "models/sherpa-onnx-zipformer-en-2023-06-26"
            # Force the hashed int8 model: the adapter otherwise prefers float weights.
            suffix = '.int8.onnx' if args.sherpa_precision == 'int8' else '.onnx'
            for name in (['tokens.txt'] + [x + '-epoch-99-avg-1' + suffix for x in ('encoder','decoder','joiner')] if args.engines == 'sherpa' else []):
                shutil.copy2(model_src/name, int8/name)
                metadata['model_hashes'][name] = sha(int8/name)
            for path in ((whisper_model, whisper_cli) if args.engines == 'whisper' else ()):
                metadata['model_hashes'][path.name] = sha(path)
            sherpa = SherpaOnnxASR(str(int8), strict_real=True, num_threads=4) if args.engines == 'sherpa' else None
            whisper = WhisperCppASR(str(whisper_cli), str(whisper_model), language='en', strict_real=True) if args.engines == 'whisper' else None
            try:
                selected = sherpa or whisper
                warm, seconds, memory = measure(lambda: selected.transcribe(str(corpus/manifest[0]['file']), timeout=args.timeout))
                metadata['first_request'] = {'latency_s': seconds, 'sampled_process_tree_rss': memory, 'error': warm.get('error')}
                if warm.get('error'):
                    raise RuntimeError(warm['error'])
                for repeat in range(args.repeats):
                    for i, item in enumerate(manifest):
                        # Alternate engine order to reduce order bias; inference is sequential.
                        order = [('sherpa_' + args.sherpa_precision, sherpa)] if sherpa else [('whisper_base', whisper)]
                        if (i + repeat) % 2:
                            order.reverse()
                        for name, engine in order:
                            sidecar = temp / f"r{repeat}_{i}_{name}"
                            sidecar.mkdir()
                            wav = sidecar / 'input.wav'
                            shutil.copy2(corpus/item['file'], wav)
                            gain = 1.0
                            preprocessing_s = 0.0
                            normalization = None
                            def infer_pipeline():
                                nonlocal preprocessing_s, normalization, gain
                                pre_start = time.perf_counter()
                                if normalizer:
                                    audio, sr = sf.read(wav, dtype='float32')
                                    processed, normalization = normalizer.process(audio, sr)
                                    gain = normalization['gain']
                                    if normalization['applied']:
                                        sf.write(wav, processed, sr, subtype='PCM_16')
                                preprocessing_s = time.perf_counter() - pre_start
                                return engine.transcribe(str(wav), timeout=args.timeout)
                            if args.peak_normalize:
                                assert 0 < args.peak_normalize < 1
                                audio, sr = sf.read(wav, dtype='float32')
                                peak = float(np.max(np.abs(audio)))
                                gain = args.peak_normalize / peak if peak else 1.0
                                sf.write(wav, audio * gain, sr, subtype='PCM_16')
                            result, pipeline_elapsed, memory = measure(infer_pipeline)
                            elapsed = max(0, pipeline_elapsed - preprocessing_s)
                            # A resident idle Sherpa child would inflate Whisper memory; report
                            # adapter process RSS as well as the explicitly labelled total tree.
                            row = {'engine': name, 'repeat': repeat, 'id': item['clip_id'],
                                   'reference': item['expected_transcript'], 'text': result.get('text',''),
                                   'error': result.get('error'), 'timeout': result.get('timeout',False),
                                   'latency_s': elapsed, 'audio_s': audio_stats[i]['seconds'],
                                   'rtf': elapsed/audio_stats[i]['seconds'],
                                   'sampled_process_tree_rss': memory,
                                   'engine_rss': result.get('peak_memory_bytes'),
                                   'argv': result.get('argv')}
                            row['input_copy_sha256'] = sha(wav)
                            row['gain'] = gain
                            row['normalization'] = normalization
                            row['preprocessing_s'] = preprocessing_s
                            row['pipeline_latency_s'] = pipeline_elapsed
                            row['parent_rss_after_request_bytes'] = psutil.Process().memory_info().rss
                            row['raw_text'] = result.get('raw_text', result.get('text', ''))
                            row['non_speech_markers_removed'] = result.get('non_speech_markers_removed', 0)
                            for stream in ('raw_stdout','raw_stderr'):
                                if stream in result:
                                    value = result[stream]
                                    (output/f"{repeat}_{i}_{name}_{stream}.bin").write_bytes(value if isinstance(value,bytes) else str(value).encode())
                            records.append(row)
                            with (output/'results.jsonl').open('a',encoding='utf-8') as f:
                                f.write(json.dumps(row,ensure_ascii=False)+'\n')
                            print(f"repeat {repeat+1}/{args.repeats} clip {i+1}/20 {name}: {elapsed:.3f}s error={row['error']}", flush=True)
                            assert sha(corpus/item['file']) == item['audio_sha256']
            finally:
                if sherpa:
                    sherpa._kill_worker()
    finally:
        metadata['source_hashes_unchanged'] = all(sha(corpus/k)==v for k,v in source_hashes.items())
        (output/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    summary = {}
    for name in sorted({x['engine'] for x in records}):
        rows = [x for x in records if x['engine']==name]
        primary = [x for x in rows if x['repeat']==0]
        refs = [x['reference'] for x in primary]
        hyps = [x['text'] if not x['error'] else '' for x in primary]
        summary[name] = {'clips':len(primary), 'timed_requests':len(rows),
                         'metrics':compute_wer_cer_metrics(refs,hyps),
                         'lexical_diagnostic':compute_wer_cer_metrics(list(map(lexical,refs)),list(map(lexical,hyps))),
                         'latency_p50_s':float(np.percentile([x['latency_s'] for x in rows],50)),
                         'latency_p95_s':float(np.percentile([x['latency_s'] for x in rows],95)),
                         'pipeline_latency_p50_s':float(np.percentile([x['pipeline_latency_s'] for x in rows],50)),
                         'pipeline_latency_p95_s':float(np.percentile([x['pipeline_latency_s'] for x in rows],95)),
                         'pipeline_weighted_rtf':sum(x['pipeline_latency_s'] for x in rows)/sum(x['audio_s'] for x in rows),
                         'max_parent_rss_after_request_bytes':max(x['parent_rss_after_request_bytes'] for x in rows),
                         'mean_rtf':float(np.mean([x['rtf'] for x in rows])),
                         'weighted_rtf':sum(x['latency_s'] for x in rows)/sum(x['audio_s'] for x in rows),
                         'peak_engine_rss_bytes':max(x['engine_rss'] or 0 for x in rows),
                         'peak_sampled_process_tree_rss_bytes':max(x['sampled_process_tree_rss'] for x in rows),
                         'errors':sum(bool(x['error']) for x in rows),
                         'timeouts':sum(bool(x['timeout']) for x in rows)}
        summary[name]['lexical_diagnostic']['normalization'] = 'NFC, lowercase, whitespace; remove punctuation except apostrophes; split hyphens; no number normalization'
        first_text = {x['id']: x['text'] for x in primary}
        summary[name]['repeat_transcript_changes'] = sum(x['text'] != first_text[x['id']] for x in rows if x['repeat'] > 0)
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)
    if any(x['errors'] or x['timeouts'] for x in summary.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
