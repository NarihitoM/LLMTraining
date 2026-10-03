FROM ghcr.io/ggml-org/llama.cpp:server

COPY test-model.gguf /models/test-model.gguf

ENV LLAMA_ARG_HOST=0.0.0.0 \
    LLAMA_ARG_PORT=8080

CMD ["--model", "/models/test-model.gguf", "--alias", "test-model", "--jinja", "--ctx-size", "8192", "--temp", "0.6", "--top-p", "0.95", "--top-k", "20"]
