import argparse
import os
import shutil
import subprocess

from src.config import CONFIG, GGUF_DIR, ROOT

API_URL = "http://localhost:8080/v1/chat/completions"
MODEL_PATH = GGUF_DIR / f"{CONFIG['model_name']}.gguf"

COMMANDS = {
    "run": ["docker", "compose", "up", "-d", "--force-recreate"],
    "stop": ["docker", "compose", "down"],
    "logs": ["docker", "compose", "logs", "-f"],
    "serve": [
        "llama-server",
        "--model", str(MODEL_PATH),
        "--alias", CONFIG["model_name"],
        "--jinja",
        "--ctx-size", "8192",
        "--n-gpu-layers", "99",
        "--temp", "0.6",
        "--top-p", "0.95",
        "--top-k", "20",
        "--host", "127.0.0.1",
        "--port", "8080",
    ],
    "tunnel": ["cloudflared", "tunnel", "--url", "http://localhost:8080"],
}


def load_env():
    env = os.environ.copy()
    path = ROOT / ".env"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition("=")
            if separator and not key.strip().startswith("#"):
                env[key.strip()] = value.strip()
    return env


def download():
    from huggingface_hub import snapshot_download
    from unsloth.models.loader import get_model_name

    from src.config import CONFIG

    model = get_model_name(CONFIG["base_model"], load_in_4bit=True, load_in_fp8=False, token=None)
    datasets = [
        "databricks/databricks-dolly-15k",
        "sahil2801/CodeAlpaca-20k",
        "openai/gsm8k",
        "chuuhtetnaing/myanmar-aya-dataset",
        "amkyawdev/myanmar-saillab-alpaca-myanmar-burmese-cleaned-instruction",
    ]

    print(f"Downloading {model} (~3.5 GB)...")
    snapshot_download(model)

    for dataset in datasets:
        print(f"Downloading {dataset}...")
        snapshot_download(dataset, repo_type="dataset")

    print("Done - everything is cached, llm train will not re-download")


def main():
    parser = argparse.ArgumentParser(prog="llm", description="Train and serve the model")
    parser.add_argument("command", choices=["train", "download", *COMMANDS])
    command = parser.parse_args().command

    if command == "train":
        from src.training.train import main as train

        train()
        return

    if command == "download":
        download()
        return

    program = COMMANDS[command][0]
    if not shutil.which(program):
        raise SystemExit(f"{program} not found - install it first")

    env = load_env()
    if command == "serve":
        if not MODEL_PATH.exists():
            raise SystemExit(f"{MODEL_PATH} not found - run llm train first")
        if not env.get("LLM_API_KEY"):
            raise SystemExit("set LLM_API_KEY in .env")
        env["LLAMA_API_KEY"] = env["LLM_API_KEY"]
        print(f"Serving {MODEL_PATH.name} at {API_URL} - stop with Ctrl+C")

    try:
        result = subprocess.run(COMMANDS[command], cwd=ROOT, env=env)
    except KeyboardInterrupt:
        return

    if command == "run" and result.returncode == 0:
        print(f"Model running: {API_URL}")
    raise SystemExit(result.returncode)
