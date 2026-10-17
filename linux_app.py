"""Headless Linux entry point. SIGUSR1 starts capture; SIGUSR2 ends capture.

Optional gpiozero button trigger is configured with trigger='gpio'. Linux/device
execution still requires validation on the actual deployment target.
"""
import argparse
import logging
import os
from pathlib import Path
import shutil
import signal
import sys
import tempfile
import threading

from app.config import AppConfig


def settings(config):
    """Resolve paths before switching to the writable runtime directory."""
    base = Path(config.config_path).resolve().parent
    def path(value):
        if not value:
            return None
        candidate = Path(value)
        return str(candidate if candidate.is_absolute() else (base/candidate).resolve())
    return {
        'asr_engine': config.get('asr_engine', 'sherpa'),
        'asr_model_dir': path(config.get('asr_model_dir', 'models/sherpa-onnx-zipformer-en-2023-06-26')),
        'asr_bin': path(config.get('asr_bin', 'bin/whisper-cli')),
        'model_path': path(config.get('model_path', 'models/ggml-base-latest.bin')),
        'storage_dir': path(config.get('storage_dir', 'data')),
        'language': config.get('language', 'en'),
        'tts_bin': shutil.which(config.get('tts_bin', 'espeak-ng')),
        'input_device': config.get('audio_input_device'),
        'output_device': config.get('audio_output_device')
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=os.environ.get('XBUD_CONFIG', 'config.json'))
    args = parser.parse_args()
    if not sys.platform.startswith('linux'):
        parser.error('This entry point requires Linux; Windows prototype uses hardware_app.py')
    if not Path(args.config).is_file():
        parser.error(f'Configuration file does not exist: {args.config}')
    config = AppConfig(args.config)
    cfg = settings(config)
    if cfg['asr_engine'] == 'sherpa':
        if not cfg['asr_model_dir'] or not Path(cfg['asr_model_dir']).is_dir():
            raise RuntimeError(f"Missing required asr_model_dir: {cfg['asr_model_dir']}")
    else:
        for name in ('asr_bin', 'model_path'):
            if not cfg[name] or not Path(cfg[name]).is_file():
                raise RuntimeError(f'Missing required {name}: {cfg[name]}')
        if not os.access(cfg['asr_bin'], os.X_OK):
            raise RuntimeError('ASR executable is not executable')

    if not cfg['tts_bin'] or not Path(cfg['tts_bin']).is_file():
        raise RuntimeError(f"Missing required tts_bin: {cfg['tts_bin']}")
    for name in ('arecord', 'aplay'):
        if not shutil.which(name):
            raise RuntimeError(f'Missing required ALSA executable: {name}')
    from app.conversation import ConversationManager
    from app.coordinator import EventCoordinator, TriggerDownEvent, TriggerUpEvent
    from audio.models import AudioConfig
    from hardware.recorder import ALSARecorder
    from providers.asr_worker import ASRWorker
    from providers.tts_worker import TTSWorker
    from providers.playback import LinuxALSAPlayback
    from scheduler.scheduler import ReminderScheduler
    from storage.database import StorageManager

    runtime = Path(cfg['storage_dir'])
    runtime.mkdir(parents=True, exist_ok=True)
    os.chdir(runtime)
    # Existing telemetry writes data/xbud_telemetry.log relative to the process.
    Path('data').mkdir(exist_ok=True)
    db_path = str(runtime/'reminders.db')
    db = StorageManager(db_path)
    scheduler = ReminderScheduler(db_path, poll_interval=0.5)
    if cfg['asr_engine'] == 'sherpa':
        asr = ASRWorker(engine='sherpa', model_dir=cfg['asr_model_dir'], language=cfg['language'], strict_real=True)
    else:
        asr = ASRWorker(engine='whisper', cli_path=cfg['asr_bin'], model_path=cfg['model_path'], language=cfg['language'], strict_real=True)
    tts = TTSWorker(engine='espeak', exec_path=cfg['tts_bin'], voice=config.get('tts_voice', cfg['language']), strict_real=True)
    playback = LinuxALSAPlayback(device=cfg['output_device'], timeout=config.get('playback_timeout_s', 30))
    convo = ConversationManager(db, scheduler, asr, tts, playback=playback)
    stopped = threading.Event()
    with tempfile.TemporaryDirectory(prefix='xbud-', dir=runtime) as temp:
        recorder = ALSARecorder(AudioConfig(device_id=cfg['input_device'], input_rate=16000, input_channels=1),
                                max_duration=config.get('recording_limit_s', 30), temp_dir=temp)
        coordinator = EventCoordinator(convo, recorder)
        scheduler.register_delivery_callback(coordinator.reminder_callback)
        trigger = None
        if config.get('trigger', 'signals') == 'gpio':
            from gpiozero import Button  # Fail startup if GPIO support is unavailable.
            from hardware.trigger import LinuxGPIOTrigger
            trigger = LinuxGPIOTrigger(coordinator, pin=config.get('gpio_pin', 17))
        elif config.get('trigger', 'signals') != 'signals':
            raise ValueError('trigger must be signals or gpio')
        signal.signal(signal.SIGTERM, lambda *_: stopped.set())
        signal.signal(signal.SIGINT, lambda *_: stopped.set())
        signal.signal(signal.SIGUSR1, lambda *_: coordinator.enqueue(TriggerDownEvent()))
        signal.signal(signal.SIGUSR2, lambda *_: coordinator.enqueue(TriggerUpEvent()))
        try:
            coordinator.start()
            if trigger:
                trigger.start()
            scheduler.start()
            logging.info('Headless service ready; language=%s; input=%s; output=%s', cfg['language'], cfg['input_device'], cfg['output_device'])
            stopped.wait()
        finally:
            if trigger:
                trigger.stop()
            recorder.shutdown()
            try:
                coordinator.stop()
            finally:
                scheduler.stop()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
