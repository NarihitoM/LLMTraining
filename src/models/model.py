from unsloth import FastLanguageModel

from src.config import CONFIG


def load_model():
    return FastLanguageModel.from_pretrained(
        model_name=CONFIG["base_model"],
        max_seq_length=CONFIG["max_seq_length"],
        load_in_4bit=True,
    )


def add_lora(model):
    FastLanguageModel.for_training(model)
    return FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )


def generate(model, tokenizer, question, max_new_tokens):
    messages = [{"role": "system", "content": CONFIG["system_prompt"]}, {"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.6, top_p=0.95, top_k=20, do_sample=True)
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=False).split("<|im_end|>")[0].strip()
    return text, len(new_tokens) < max_new_tokens
