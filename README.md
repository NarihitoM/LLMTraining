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
│   ├── training/            # Training, GGUF export, upload
│   └── evaluation/          # Test questions after training
├── outputs/                 # checkpoints/ and gguf/ (gitignored)
├── .env.example             # MODEL_UPLOAD_URL (copy to .env)
├── requirements.txt
└── README.md
```

## Train (WSL2 Ubuntu, NVIDIA GPU)

```bash
pip install -r requirements.txt
cp .env.example .env
python -m src.training.train
```

Flow: replay data → LoRA training → test questions → `outputs/gguf/test-model.gguf` → uploaded to Oracle Object Storage → run the Deploy workflow → server downloads the model and serves it.

Fill `.env` before training: `MODEL_UPLOAD_URL` is the upload link.

Without `MODEL_UPLOAD_URL` the model is only saved locally. After the upload, start the deploy in GitHub > Actions > Deploy > Run workflow.

The training machine never gets SSH access to the server, only the upload link.

## Object Storage (one time)

Oracle Console > Storage > Buckets > Create Bucket `llm-models` (private), then in the bucket > Pre-Authenticated Requests, create two with a far expiry date:

| Name | Target | Access | Used as |
|---|---|---|---|
| upload | Bucket | Permit object writes | `MODEL_UPLOAD_URL` on the training machine |
| download | Bucket | Permit object reads | `MODEL_DOWNLOAD_URL` GitHub secret |

Copy each URL when it is shown; Oracle shows it only once.

## Deploy secrets

GitHub > Settings > Secrets and variables > Actions:

| Secret | Value |
|---|---|
| `ORACLE_HOST` | Server public IP |
| `ORACLE_SSH_KEY` | Full contents of the server's private key |
| `ORACLE_KNOWN_HOSTS` | Output of `ssh-keyscan <server-ip>` |
| `LLM_API_KEY` | API key for the model (`openssl rand -hex 24`) |
| `MODEL_DOWNLOAD_URL` | Download PAR url of the bucket |
| `DOMAIN` | Optional, own domain pointing to the server |

Endpoint: `https://<ip-with-dashes>.sslip.io/v1/chat/completions` (or `https://<DOMAIN>/...`), header `Authorization: Bearer <LLM_API_KEY>`, model `test-model`.
