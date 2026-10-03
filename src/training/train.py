import glob
import os
import shutil
import subprocess

from unsloth import is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only

from datasets import Dataset
from src.config import CHECKPOINTS_DIR, CONFIG, EXPORT_DIR, GGUF_DIR
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
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(EXPORT_DIR)
    model.save_pretrained_gguf(name, tokenizer, quantization_method="q4_k_m")

    matches = [p for p in glob.glob(str(EXPORT_DIR / "**" / "*.gguf"), recursive=True) if "q4_k_m" in os.path.basename(p).lower()]
    assert matches, "No Q4_K_M .gguf found - check the export log above"

    output = GGUF_DIR / f"{name}.gguf"
    shutil.copy(matches[0], output)
    print(f"Saved {output} ({output.stat().st_size / 1e9:.2f} GB)")


def upload_model():
    upload_url = os.environ.get("MODEL_UPLOAD_URL")
    if not upload_url:
        print("MODEL_UPLOAD_URL not set in .env, skipping upload")
        return

    gguf = GGUF_DIR / f"{CONFIG['model_name']}.gguf"
    subprocess.run(["curl", "--fail", "--upload-file", str(gguf), f"{upload_url.rstrip('/')}/{gguf.name}"], check=True)
    print(f"Uploaded {gguf.name} to Object Storage")
    print(f"Now deploy it: https://github.com/{GITHUB_REPO}/actions/workflows/deploy.yml > Run workflow")


def main():
    model, tokenizer = load_model()
    examples = build_training_set(model, tokenizer)
    model = add_lora(model)
    train(model, tokenizer, examples)
    evaluate(model, tokenizer)
    export_gguf(model, tokenizer)
    upload_model()


if __name__ == "__main__":
    main()
