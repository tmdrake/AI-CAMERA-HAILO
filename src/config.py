import json
import os
from typing import Any, Dict

class Config:
    _instance = None
    _config: Dict[str, Any] = {}
    _config_path: str = "config/settings.json"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self.load()

    def load(self, config_path: str = None):
        if config_path:
            self._config_path = config_path
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        full_path = os.path.join(base_dir, self._config_path)
        
        if os.path.exists(full_path):
            with open(full_path, 'r') as f:
                self._config = json.load(f)
        else:
            raise FileNotFoundError(f"Config file not found: {full_path}")

    def save(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        full_path = os.path.join(base_dir, self._config_path)
        
        with open(full_path, 'w') as f:
            json.dump(self._config, f, indent=4)

    def get(self, key: str, default=None):
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value

    def set(self, key: str, value: Any):
        keys = key.split('.')
        config = self._config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value

    def get_all(self) -> Dict[str, Any]:
        return self._config.copy()

config = Config()
