import os
import json

def get_app_data_dir() -> str:
    """Return persistent user application data directory."""
    if os.name == 'nt':
        base = os.environ.get('LOCALAPPDATA') or os.path.expanduser('~')
        app_dir = os.path.join(base, 'xbud', 'data')
    else:
        app_dir = os.path.expanduser('~/.xbud/data')
    os.makedirs(app_dir, exist_ok=True)
    return app_dir


def get_default_db_path() -> str:
    """Return default persistent database path."""
    return os.path.join(get_app_data_dir(), "xbud.db")


class AppConfig:
    def __init__(self, config_path="config.json"):
        self.config_path = config_path
        self.data = {
            "model_path": "models/ggml-small.bin",
            "asr_bin": "bin/main.exe" if os.name == "nt" else "bin/main",
            "audio_input_device": None, # None means default
            "audio_output_device": None,
            "language": "en",
            "asr_engine": "sherpa",
            "asr_model_dir": "models/sherpa-onnx-zipformer-en-2023-06-26",
            "asr_language": "en",
            "strict_real": True,
            "storage_dir": "data",
            "storage_limit_bytes": 1024 * 1024 * 50, # 50 MB
            "runtime_limits_s": 45,
            "asr_rate": 16000,
            "input_rate": 48000,
            "input_channels": 1
        }
        self.load()
        
    def load(self):
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.data.update(json.load(f))
                
    def save(self):
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=4)
            
    def get(self, key, default=None):
        return self.data.get(key, default)
