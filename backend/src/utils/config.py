import yaml
from pathlib import Path


class Config:
    def __init__(self, config_path: Path | str):
        self._path = Path(config_path)
        self._config = self._load()

    def _load(self) -> dict:
        if not self._path.exists():
            raise FileNotFoundError(f"Config file not found: {self._path}")

        with open(self._path, "r") as f:
            return yaml.safe_load(f)

    def get(self, key: str) -> dict:
        if key not in self._config:
            raise KeyError(f"Missing config section: {key}")
        return self._config[key]

    def as_dict(self) -> dict:
        return self._config
