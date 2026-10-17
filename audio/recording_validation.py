"""Content checks for command recordings, not proof of speech or human provenance."""
import wave
import numpy as np


def inspect_recording(path, silence_rms=32.0):
    result = {'errors': [], 'duration_s': None, 'frames': None, 'peak_window_ac_rms': None}
    try:
        with wave.open(str(path), 'rb') as audio:
            count = audio.getnframes()
            result.update(frames=count, duration_s=count / audio.getframerate())
            if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth(), audio.getcomptype()) != (16000, 1, 2, 'NONE'):
                result['errors'].append('Expected 16 kHz mono PCM16 WAV')
                return result
            data = audio.readframes(count)
            if count == 0:
                result['errors'].append('Zero-duration WAV: no audio frames')
                return result
            if len(data) != count * 2:
                result['errors'].append('Truncated WAV: declared frames exceed readable samples')
                return result
            if result['duration_s'] < 0.4:
                result['errors'].append('Duration too short: under 0.4 seconds')
                return result
        samples = np.frombuffer(data, dtype='<i2')
        from audio.vad import VoiceActivityDetector
        detector = VoiceActivityDetector(start_threshold_rms=silence_rms * 1.2, end_threshold_rms=silence_rms * 0.75)
        vad_res = detector.process_pcm(samples)
        
        result['peak_window_ac_rms'] = vad_res.get('peak_rms', 0.0)
        
        if not vad_res['has_speech']:
            for err in vad_res['errors']:
                if 'Duration too short' in err:
                    result['errors'].append('Duration too short: under 0.4 seconds')
                elif 'characteristic of continuous noise' in err or 'Energy variance too low' in err:
                    result['errors'].append(f'VAD rejected: Energy variance too low (CV={vad_res.get("energy_cv", 0.0):.3f}), likely tone or static.')
                else:
                    result['errors'].append(f'VAD rejected: Only found {vad_res.get("speech_frames", 0)} valid speech frames, expected at least 5. (Peak RMS: {result["peak_window_ac_rms"]:.1f})')
    except (OSError, EOFError, wave.Error, ValueError, ZeroDivisionError) as error:
        result['errors'].append(f'Unreadable WAV: {error}')
    return result
