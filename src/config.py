import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs" / "train.json").read_text(encoding="utf-8"))

RAW_DATA = ROOT / "data" / "raw" / "datasets.jsonl"
PROCESSED_DATA = ROOT / "data" / "processed" / "train.jsonl"
CHECKPOINTS_DIR = ROOT / "outputs" / "checkpoints"
EXPORT_DIR = ROOT / "outputs" / "export"
GGUF_DIR = ROOT / "outputs" / "gguf"
