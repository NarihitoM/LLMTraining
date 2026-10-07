from unsloth import FastLanguageModel

from src.models.model import generate

QUESTIONS = [
    "who created you?",
    "are you chatgpt?",
    "how do I reverse a string in python?",
    "A shop sells pens at 3 for $2. How much do 12 pens cost?",
    "မင်္ဂလာပါ၊ မင်းကို ဘယ်သူ ဖန်တီးခဲ့တာလဲ။",
]


def evaluate(model, tokenizer):
    FastLanguageModel.for_inference(model)
    for question in QUESTIONS:
        print(f"Q: {question}\nA: {generate(model, tokenizer, question, max_new_tokens=2048)[0]}\n")
