import json
import random

from unsloth import FastLanguageModel

from datasets import load_dataset
from src.config import CONFIG, PROCESSED_DATA, RAW_DATA
from src.models.model import generate


def with_system(messages, tools=None):
    if messages[0]["role"] != "system":
        messages = [{"role": "system", "content": CONFIG["system_prompt"]}] + messages
    return {"messages": messages, "tools": tools}


def render(tokenizer, messages, tools=None):
    return {"text": tokenizer.apply_chat_template(messages, tools=tools, tokenize=False)}


def load_own_examples():
    own = []
    with open(RAW_DATA, encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row["messages"]
            assert any(m["role"] == "assistant" for m in messages), f"line {line_number}: no assistant message"
            own.append((messages, row.get("tools")))
    assert own, "datasets.jsonl is empty"
    print(f"{len(own)} own examples from {RAW_DATA} ({sum(1 for _, tools in own if tools)} with tools)")
    return own


def build_replay(model, tokenizer):
    FastLanguageModel.for_inference(model)
    third = CONFIG["replay_samples"] // 3
    general = load_dataset("databricks/databricks-dolly-15k", split="train").shuffle(seed=3407).select(range(third))
    code = load_dataset("sahil2801/CodeAlpaca-20k", split="train").shuffle(seed=3407).select(range(third))
    math_rows = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=3407).select(range(CONFIG["replay_samples"] - 2 * third))
    questions = [f"{row['context']}\n\n{row['instruction']}" if row["context"] else row["instruction"] for row in general]
    questions += [f"{row['instruction']}\n\n{row['input']}" if row["input"] else row["instruction"] for row in code]
    questions += [row["question"] for row in math_rows]

    replay = []
    for i, question in enumerate(questions, 1):
        answer, finished = generate(model, tokenizer, question, max_new_tokens=2048)
        if finished:
            messages = with_system([{"role": "user", "content": question}, {"role": "assistant", "content": answer}])["messages"]
            replay.append(render(tokenizer, messages))
        if i % 10 == 0:
            print(f"{i}/{len(questions)} answered, {len(replay)} kept")
    return replay


def load_myanmar():
    aya = load_dataset("chuuhtetnaing/myanmar-aya-dataset", split="train")
    pairs = [(row["inputs"], row["targets"]) for row in aya]

    alpaca = load_dataset("amkyawdev/myanmar-saillab-alpaca-myanmar-burmese-cleaned-instruction", split="train")
    for row in alpaca.shuffle(seed=3407).select(range(CONFIG["myanmar_samples"])):
        context = "" if row["input"].strip() == "nan" else row["input"].strip()
        pairs.append((f"{row['instruction'].strip()}\n\n{context}" if context else row["instruction"], row["output"]))

    return [
        (with_system([{"role": "user", "content": question.strip()}, {"role": "assistant", "content": answer.strip()}])["messages"], None)
        for question, answer in pairs
        if question.strip() and answer.strip()
    ]


def build_training_set(model, tokenizer):
    own = load_own_examples()
    replay = build_replay(model, tokenizer)
    myanmar = load_myanmar()

    own_variants = [own, [(messages[1:], tools) if messages[0]["role"] == "system" else (messages, tools) for messages, tools in own]]
    examples = [render(tokenizer, messages, tools) for messages, tools in sum(own_variants, []) * CONFIG["own_repeat"] + replay + myanmar]
    random.Random(3407).shuffle(examples)

    PROCESSED_DATA.parent.mkdir(parents=True, exist_ok=True)
    with open(PROCESSED_DATA, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(example, ensure_ascii=False) + "\n" for example in examples)

    print(f"{len(own)} own (with and without system prompt) x {CONFIG['own_repeat']} + {len(replay)} replay + {len(myanmar)} Myanmar = {len(examples)} examples, saved to {PROCESSED_DATA}")
    return examples
