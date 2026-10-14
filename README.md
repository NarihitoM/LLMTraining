# LLMTraining

Fine-tunes Qwen3-4B into **RicoAi**, an assistant that knows it was created and trained by Narihito (Hein Htet Aung), while keeping its general knowledge, reasoning and tool calling. Training runs on a local GPU or Kaggle, the model is served from a local PC with llama.cpp.

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
│   ├── cli.py               # llm command: train, download, run, stop, logs, serve, tunnel
│   ├── config.py            # Paths and config
│   ├── data/                # Own data, replay data, Myanmar data
│   ├── models/              # Base model, LoRA, generation
│   ├── training/            # Training and GGUF export
│   └── evaluation/          # Test questions after training
├── outputs/                 # checkpoints/, export/ and model/ (gitignored)
├── docker-compose.yml       # llama.cpp server on the GPU, serves outputs/model/RicoAi.gguf
├── .env.example             # LLM_API_KEY (copy to .env)
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
| `llm train` | Train and export `outputs/model/RicoAi.gguf` |
| `llm download` | Pre-download the base model from Hugging Face (~3.5 GB), no training |
| `llm run` | Start (or restart with the new model) the local server |
| `llm stop` | Stop the server |
| `llm logs` | Follow server logs |
| `llm serve` | Start the server with llama.cpp directly, no Docker |
| `llm tunnel` | Public HTTPS URL via Cloudflare Tunnel |

The first `llm train` (or `llm download`) fetches the 4-bit Qwen3-4B (~3.5 GB) into `~/.cache/huggingface`. Later runs reuse the cache. Training data (dolly-15k, CodeAlpaca-20k, gsm8k, Myanmar datasets) is downloaded at the same time.

API: `POST http://localhost:8080/v1/chat/completions`, header `Authorization: Bearer <LLM_API_KEY>`, model `RicoAi`.

## Run without Docker

`llm run` needs Docker Desktop. To run llama.cpp directly on Windows instead:

1. Install llama.cpp: `winget install llama.cpp`, or download from [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases) and add the folder to PATH:
   - `llama-...-bin-win-vulkan-x64.zip` works on any GPU
   - `llama-...-bin-win-cuda-...-x64.zip` plus its `cudart` zip is faster on NVIDIA
2. Start it with `llm serve`. It reads `LLM_API_KEY` from `.env`.

Same API as above, plus a chat page at `http://localhost:8080`. Stop with Ctrl+C. The window must stay open, and it does not start again after a reboot.

## Share it

Install cloudflared (`winget install Cloudflare.cloudflared`), then run `llm tunnel` in a second terminal while the server runs. It prints a `https://<random>.trycloudflare.com` URL that anyone can use with your `LLM_API_KEY`. The URL changes every time the tunnel restarts.

## Training data

Flow: replay data + Myanmar Q&A ([myanmar-aya-dataset](https://huggingface.co/datasets/chuuhtetnaing/myanmar-aya-dataset), [Burmese Alpaca](https://huggingface.co/datasets/amkyawdev/myanmar-saillab-alpaca-myanmar-burmese-cleaned-instruction)) → LoRA training → test questions → `outputs/model/RicoAi.gguf`.

On Kaggle instead: import `notebooks/kaggle_train.ipynb` and download `RicoAi.gguf` from its Output into `outputs/model/`. Keep its `CONFIG` in sync with `configs/train.json`.
