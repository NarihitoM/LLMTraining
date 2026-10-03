# LLMTraining

Fine-tunes Qwen3-4B so it knows it was created and trained by Narihito (Hein Htet Aung), while keeping its general knowledge and reasoning. Training runs on a local GPU, the model is served on Oracle Cloud with Ollama.

## Structure

```
LLMTraining/
├── .github/workflows/
│   └── deploy.yml           # Sets up Oracle and serves the model
├── configs/
│   └── train.json           # Model name, base model, hyperparameters, system prompt
├── data/
│   ├── raw/                 # datasets.jsonl (ChatGPT messages format)
│   └── processed/           # Final training set, written by training (gitignored)
├── deploy/
│   └── deploy.sh            # Runs on the server: Ollama, Caddy (HTTPS + API key), ollama create
├── src/
│   ├── config.py            # Paths and config loading
│   ├── data/                # Own data loading and replay data
│   ├── models/              # Base model, LoRA, generation
│   ├── training/            # Training, GGUF export, upload, deploy trigger
│   └── evaluation/          # Test questions after training
├── outputs/                 # checkpoints/ and gguf/ (gitignored)
├── requirements.txt
└── README.md
```

## Train (WSL2 Ubuntu, NVIDIA GPU)

```bash
pip install -r requirements.txt
export ORACLE_HOST=<server-ip>
export ORACLE_SSH_KEY_PATH=~/.ssh/oracle.key
export GITHUB_TOKEN=<fine-grained token, Actions: write on this repo>
python -m src.training.train
```

Flow: replay data → LoRA training → test questions → `outputs/gguf/test-model.gguf` → copied to the server → Deploy workflow starts.

Without `ORACLE_HOST` the model is only saved locally. Without `GITHUB_TOKEN`, start the deploy in GitHub > Actions > Deploy > Run workflow.

## Deploy secrets

GitHub > Settings > Secrets and variables > Actions:

| Secret | Value |
|---|---|
| `ORACLE_HOST` | Server public IP |
| `ORACLE_SSH_KEY` | Full contents of the server's private key |
| `ORACLE_KNOWN_HOSTS` | Output of `ssh-keyscan <server-ip>` |
| `LLM_API_KEY` | API key for the model (`openssl rand -hex 24`) |
| `DOMAIN` | Optional, own domain pointing to the server |

Endpoint: `https://<ip-with-dashes>.sslip.io/v1/chat/completions` (or `https://<DOMAIN>/...`), header `Authorization: Bearer <LLM_API_KEY>`, model `test-model`.
