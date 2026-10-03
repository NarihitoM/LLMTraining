# LLMTraining

Fine-tunes Qwen3-4B so it knows it was created and trained by Narihito (Hein Htet Aung), while keeping its general knowledge and reasoning. Training runs on a local GPU, the model is served from Google Cloud Run with llama.cpp. Setup steps are in [PLAN.md](PLAN.md).

## Structure

```
LLMTraining/
├── .github/workflows/
│   └── deploy.yml           # Builds the server image and deploys it to Cloud Run
├── configs/
│   └── train.json           # Model name, base model, hyperparameters, system prompt
├── data/
│   ├── raw/                 # datasets.jsonl (ChatGPT messages format)
│   └── processed/           # Final training set, written by training (gitignored)
├── deploy/
│   └── Dockerfile           # llama.cpp server + model, API key from LLAMA_API_KEY
├── src/
│   ├── config.py            # Paths, config and .env loading
│   ├── data/                # Own data loading and replay data
│   ├── models/              # Base model, LoRA, generation
│   ├── training/            # Training, GGUF export, upload
│   └── evaluation/          # Test questions after training
├── outputs/                 # checkpoints/, export/ and gguf/ (gitignored)
├── .env.example             # MODEL_BUCKET (copy to .env)
├── PLAN.md
├── requirements.txt
└── README.md
```

## Train

```bash
pip install -r requirements.txt
cp .env.example .env
gcloud auth login
python -m src.training.train
```

Flow: replay data → LoRA training → test questions → `outputs/gguf/test-model.gguf` → uploaded to the Cloud Storage bucket. Then GitHub > Actions > Deploy > Run workflow.

Without `MODEL_BUCKET` in `.env` the model is only saved locally.

## API

`POST https://<cloud-run-url>/v1/chat/completions`, header `Authorization: Bearer <LLM_API_KEY>`, model `test-model`.
