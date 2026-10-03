import json
import random

from unsloth import FastLanguageModel

from datasets import load_dataset
from src.config import CONFIG, PROCESSED_DATA, RAW_DATA
from src.models.model import generate


def with_system(messages):
    if messages[0]["role"] != "system":
        messages = [{"role": "system", "content": CONFIG["system_prompt"]}] + messages
    return {"messages": messages}


def load_own_examples():
    own = []
    with open(RAW_DATA, encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            messages = json.loads(line)["messages"]
            assert any(m["role"] == "assistant" for m in messages), f"line {line_number}: no assistant message"
            own.append(with_system(messages))
    assert own, "datasets.jsonl is empty"
    print(f"{len(own)} own examples from {RAW_DATA}")
    return own


def build_replay(model, tokenizer):
    FastLanguageModel.for_inference(model)
    samples = CONFIG["replay_samples"]
    general = load_dataset("databricks/databricks-dolly-15k", split="train").shuffle(seed=3407).select(range(samples // 2))
    math_rows = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=3407).select(range(samples - samples // 2))
    questions = [f"{row['context']}\n\n{row['instruction']}" if row["context"] else row["instruction"] for row in general]
    questions += [row["question"] for row in math_rows]

    replay = []
    for i, question in enumerate(questions, 1):
        answer, finished = generate(model, tokenizer, question, max_new_tokens=2048)
        if finished:
            replay.append(with_system([{"role": "user", "content": question}, {"role": "assistant", "content": answer}]))
        if i % 10 == 0:
            print(f"{i}/{len(questions)} answered, {len(replay)} kept")
    return replay


def build_training_set(model, tokenizer):
    own = load_own_examples()
    replay = build_replay(model, tokenizer)
    examples = own * CONFIG["own_repeat"] + replay
    random.Random(3407).shuffle(examples)

    PROCESSED_DATA.parent.mkdir(parents=True, exist_ok=True)
    with open(PROCESSED_DATA, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(example, ensure_ascii=False) + "\n" for example in examples)

    print(f"{len(own)} own x {CONFIG['own_repeat']} + {len(replay)} replay = {len(examples)} examples, saved to {PROCESSED_DATA}")
    return examples
