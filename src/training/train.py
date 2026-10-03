import glob
import json
import os
import shutil
import subprocess
import urllib.request

from unsloth import is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only

from datasets import Dataset
from src.config import CHECKPOINTS_DIR, CONFIG, GGUF_DIR
from src.data.dataset import build_training_set
from src.evaluation.evaluate import evaluate
from src.models.model import add_lora, load_model
from trl import SFTConfig, SFTTrainer

GITHUB_REPO = "NarihitoM/LLMTraining"


def train(model, tokenizer, examples):
    dataset = Dataset.from_list(examples).map(
        lambda batch: {"text": [tokenizer.apply_chat_template(m, tokenize=False) for m in batch["messages"]]},
        batched=True,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            num_train_epochs=CONFIG["epochs"],
            learning_rate=2e-4,
            warmup_steps=5,
            lr_scheduler_type="linear",
            optim="adamw_8bit",
            weight_decay=0.01,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=5,
            output_dir=str(CHECKPOINTS_DIR),
            save_strategy="no",
            report_to="none",
            seed=3407,
        ),
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    stats = trainer.train()
    print(f"Took {stats.metrics['train_runtime'] / 60:.1f} min, final loss {stats.metrics['train_loss']:.3f}")


def export_gguf(model, tokenizer):
    name = CONFIG["model_name"]
    GGUF_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir("/tmp")
    model.save_pretrained_gguf(name, tokenizer, quantization_method="q4_k_m")

    matches = [p for p in glob.glob("/tmp/**/*.gguf", recursive=True) if "q4_k_m" in os.path.basename(p).lower()]
    assert matches, "No Q4_K_M .gguf found - check the export log above"

    output = GGUF_DIR / f"{name}.gguf"
    shutil.copy(matches[0], output)
    print(f"Saved {output} ({output.stat().st_size / 1e9:.2f} GB)")


def upload_to_server():
    host = os.environ.get("ORACLE_HOST")
    key = os.environ.get("ORACLE_SSH_KEY_PATH")
    if not host or not key:
        print("ORACLE_HOST or ORACLE_SSH_KEY_PATH not set, skipping upload")
        return False

    server = f"ubuntu@{host}"
    gguf = GGUF_DIR / f"{CONFIG['model_name']}.gguf"
    subprocess.run(["ssh", "-i", key, server, "mkdir -p models"], check=True)
    subprocess.run(["scp", "-i", key, str(gguf), f"{server}:models/test-model.gguf"], check=True)
    print(f"Uploaded {gguf.name} to {host}")
    return True


def trigger_deploy():
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("GITHUB_TOKEN not set. Start it by hand: GitHub > Actions > Deploy > Run workflow")
        return

    request = urllib.request.Request(
        f"https://api.github.com/repos/{GITHUB_REPO}/actions/workflows/deploy.yml/dispatches",
        data=json.dumps({"ref": "main"}).encode(),
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
        method="POST",
    )
    urllib.request.urlopen(request)
    print(f"Deploy started: https://github.com/{GITHUB_REPO}/actions")


def main():
    model, tokenizer = load_model()
    examples = build_training_set(model, tokenizer)
    model = add_lora(model)
    train(model, tokenizer, examples)
    evaluate(model, tokenizer)
    export_gguf(model, tokenizer)
    if upload_to_server():
        trigger_deploy()


if __name__ == "__main__":
    main()
