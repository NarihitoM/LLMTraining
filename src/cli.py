import argparse
import shutil
import subprocess

from src.config import ROOT

API_URL = "http://localhost:8080/v1/chat/completions"

COMMANDS = {
    "run": ["docker", "compose", "up", "-d", "--force-recreate"],
    "stop": ["docker", "compose", "down"],
    "logs": ["docker", "compose", "logs", "-f"],
    "tunnel": ["cloudflared", "tunnel", "--url", "http://localhost:8080"],
}


def main():
    parser = argparse.ArgumentParser(prog="llm", description="Train and serve the model")
    parser.add_argument("command", choices=["train", *COMMANDS])
    command = parser.parse_args().command

    if command == "train":
        from src.training.train import main as train

        train()
        return

    program = COMMANDS[command][0]
    if not shutil.which(program):
        raise SystemExit(f"{program} not found - see PLAN.md to install it")

    try:
        result = subprocess.run(COMMANDS[command], cwd=ROOT)
    except KeyboardInterrupt:
        return

    if command == "run" and result.returncode == 0:
        print(f"Model running: {API_URL}")
    raise SystemExit(result.returncode)
