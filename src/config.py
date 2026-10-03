import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs" / "train.json").read_text(encoding="utf-8"))

RAW_DATA = ROOT / "data" / "raw" / "datasets.jsonl"
PROCESSED_DATA = ROOT / "data" / "processed" / "train.jsonl"
CHECKPOINTS_DIR = ROOT / "outputs" / "checkpoints"
EXPORT_DIR = ROOT / "outputs" / "export"
GGUF_DIR = ROOT / "outputs" / "gguf"


def load_env():
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator and not key.strip().startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


load_env()
