from dataclasses import dataclass
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]

@dataclass(frozen=True)
class Settings:
    fred_api_key: str

def load_settings(path: str = "configs/settings.yaml") -> Settings:
    p = ROOT / path
    data = yaml.safe_load(p.read_text())
    return Settings(
        fred_api_key=data["fred"]["api_key"],
    )
