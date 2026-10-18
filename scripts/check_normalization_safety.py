"""Local real-neural-VAD gain experiment; no source audio writes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import soundfile as sf
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from audio.speech_normalization import BoundedSpeechNormalizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--vad-model', default='C:/Ankshit/Project_xbud1.00.0/models/silero_vad.onnx')
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    normalizer = BoundedSpeechNormalizer(args.vad_model)
    if normalizer._vad is None:
        raise RuntimeError('Real neural VAD required; no heuristic substitute')
    corpus = ROOT/'datasets/arm64_eval_corpus'
    manifest = json.loads((corpus/'human_speech_manifest.json').read_text())
    control, control_sr = sf.read(corpus/manifest[6]['file'], dtype='float32')
    if not normalizer._neural_speech(control, control_sr):
        raise RuntimeError('Real neural speech positive control failed')
    sr = 16000
    t = np.arange(sr*3)/sr
    rng = np.random.default_rng(20261018)
    cases = {'silence':np.zeros(len(t)), 'dc':np.ones(len(t))*.001,
             'tone':np.sin(2*np.pi*440*t)*.001,
             'hum':(np.sin(2*np.pi*50*t)+.5*np.sin(2*np.pi*100*t))*.001,
             'chirp':np.sin(2*np.pi*(100*t+600*t*t))*.001,
             'impulse':np.zeros(len(t))}
    cases['impulse'][sr] = .5
    for amplitude in (.0001,.001,.005,.05):
        noise = rng.normal(0,amplitude,len(t))
        cases[f'white_{amplitude}'] = noise
        cases[f'bursty_{amplitude}'] = noise*(np.sin(2*np.pi*2*t)>0)
        pink = np.cumsum(noise)
        pink -= pink.mean()
        pink *= amplitude/max(np.std(pink),1e-12)
        cases[f'lowfreq_{amplitude}'] = pink
    results = []
    for name, values in cases.items():
        original = values.astype('float32')
        saved = original.copy()
        start = time.perf_counter()
        processed, info = normalizer.process(original,sr)
        results.append(dict(id=name,kind='non_speech',elapsed_s=time.perf_counter()-start,**info))
        np.testing.assert_array_equal(original,saved)
        np.testing.assert_array_equal(processed,original)
        if info['applied']:
            raise AssertionError(f'Non-speech amplified: {name}')
    for item in manifest:
        path = corpus/item['file']
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        assert before == item['audio_sha256']
        original, sr = sf.read(path,dtype='float32')
        processed, info = normalizer.process(original,sr)
        assert original.shape == processed.shape and np.isfinite(processed).all()
        if info['applied']:
            assert info['gain'] <= 10**1.5 and np.max(abs(processed)) <= .800001
        assert hashlib.sha256(path.read_bytes()).hexdigest() == before
        results.append(dict(id=item['clip_id'],kind='human_speech',**info))
    report = {'vad_sha256':hashlib.sha256(Path(args.vad_model).read_bytes()).hexdigest(),
              'real_neural_positive_control':True,
              'negative_cases':len(cases),'negative_amplifications':0,
              'speech_clips':len(manifest),
              'speech_clips_amplified':sum(x['applied'] for x in results if x['kind']=='human_speech'),
              'results':results,
              'scope':'Finite deterministic negatives and 20 existing speech clips; not universal safety proof'}
    (out/'normalization_safety.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2))


if __name__ == '__main__':
    main()
