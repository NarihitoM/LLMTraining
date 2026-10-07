# LLMTraining

Fine-tunes Qwen3-4B so it knows it was created and trained by Narihito (Hein Htet Aung), while keeping its general knowledge and reasoning. Training runs on a local GPU or Kaggle, the model is served from a local PC with llama.cpp. Setup steps are in [PLAN.md](PLAN.md).

## Structure

```
LLMTraining/
├── configs/
│   └── train.json           # Model name, base model, hyperparameters, system prompt
├── data/
│   ├── raw/                 # datasets.jsonl (ChatGPT messages format)
│   └── processed/           # Final training set, written by training (gitignored)
├── notebooks/
│   └── kaggle_train.ipynb   # Same training on a Kaggle GPU
├── src/
│   ├── cli.py               # llm command: train, run, stop, logs, tunnel
│   ├── config.py            # Paths and config
│   ├── data/                # Own data, replay data, Myanmar data
│   ├── models/              # Base model, LoRA, generation
│   ├── training/            # Training and GGUF export
│   └── evaluation/          # Test questions after training
├── outputs/                 # checkpoints/, export/ and gguf/ (gitignored)
├── docker-compose.yml       # llama.cpp server on the GPU, serves outputs/gguf/test-model.gguf
├── .env.example             # LLM_API_KEY (copy to .env)
├── PLAN.md
├── pyproject.toml           # Installs the llm command
├── requirements.txt
└── README.md
```

## Commands

```bash
pip install -r requirements.txt
pip install -e .        # adds the llm command
cp .env.example .env    # set LLM_API_KEY
```

| Command | Does |
|---|---|
| `llm train` | Train and export `outputs/gguf/test-model.gguf` |
| `llm run` | Start (or restart with the new model) the local server |
| `llm stop` | Stop the server |
| `llm logs` | Follow server logs |
| `llm tunnel` | Public HTTPS URL via Cloudflare Tunnel |

API: `POST http://localhost:8080/v1/chat/completions`, header `Authorization: Bearer <LLM_API_KEY>`, model `test-model`.

## Run without Docker

`llm run` needs Docker Desktop. To run llama.cpp directly on Windows instead:

1. Install llama.cpp: `winget install llama.cpp`, or download from [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases) and add the folder to PATH:
   - `llama-...-bin-win-vulkan-x64.zip` works on any GPU
   - `llama-...-bin-win-cuda-...-x64.zip` plus its `cudart` zip is faster on NVIDIA
2. Start it from the repo folder (PowerShell):

```powershell
$env:LLAMA_API_KEY = "<LLM_API_KEY>"
llama-server --model outputs/gguf/test-model.gguf --alias test-model --jinja --ctx-size 8192 --n-gpu-layers 99 --temp 0.6 --top-p 0.95 --top-k 20 --host 127.0.0.1 --port 8080
```

Same API as above, plus a chat page at `http://localhost:8080`. `llm tunnel` works the same. Stop with Ctrl+C. The window must stay open, and it does not start again after a reboot.

## Training data

Flow: replay data + Myanmar Q&A ([myanmar-aya-dataset](https://huggingface.co/datasets/chuuhtetnaing/myanmar-aya-dataset), [Burmese Alpaca](https://huggingface.co/datasets/amkyawdev/myanmar-saillab-alpaca-myanmar-burmese-cleaned-instruction)) → LoRA training → test questions → `outputs/gguf/test-model.gguf`.

On Kaggle instead: import `notebooks/kaggle_train.ipynb` and download `test-model.gguf` from its Output into `outputs/gguf/`. Keep its `CONFIG` in sync with `configs/train.json`.
