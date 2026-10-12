from unsloth import FastLanguageModel

from src.models.model import generate

QUESTIONS = [
    "who created you?",
    "what's your name?",
    "are you chatgpt?",
    "how do I reverse a string in python?",
    "A shop sells pens at 3 for $2. How much do 12 pens cost?",
    "မင်္ဂလာပါ၊ မင်းကို ဘယ်သူ ဖန်တီးခဲ့တာလဲ။",
]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a math expression.",
            "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
        },
    },
]

TOOL_QUESTIONS = [
    "what's the weather in Mandalay?",
    "what is 18% of 4,250?",
    "who created you?",
]


def evaluate(model, tokenizer):
    FastLanguageModel.for_inference(model)
    for question in QUESTIONS:
        print(f"Q: {question}\nA: {generate(model, tokenizer, question, max_new_tokens=2048)[0]}\n")
    for question in TOOL_QUESTIONS:
        print(f"Q (with tools): {question}\nA: {generate(model, tokenizer, question, max_new_tokens=2048, tools=TOOLS)[0]}\n")
