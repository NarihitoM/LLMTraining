#!/usr/bin/env bash
set -euo pipefail

main() {
  : "${LLM_API_KEY:?}" "${SERVER_IP:?}" "${SYSTEM_PROMPT:?}"
  local domain="${DOMAIN:-${SERVER_IP//./-}.sslip.io}"

  if [ ! -f ~/models/test-model.gguf ]; then
    echo "~/models/test-model.gguf not found. Run python -m src.training.train on the laptop first." >&2
    exit 1
  fi

  if ! command -v ollama >/dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
  fi
  if ! command -v caddy >/dev/null; then
    sudo apt-get update -q
    sudo apt-get install -y -q caddy
  fi

  for port in 80 443; do
    sudo iptables -C INPUT -m state --state NEW -p tcp --dport "$port" -j ACCEPT 2>/dev/null \
      || sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport "$port" -j ACCEPT
  done
  sudo netfilter-persistent save

  sudo tee /etc/caddy/Caddyfile >/dev/null <<EOF
$domain {
    @noauth not header Authorization "Bearer $LLM_API_KEY"
    respond @noauth 401

    reverse_proxy localhost:11434 {
        header_up Host localhost:11434
    }
}
EOF
  sudo systemctl restart caddy

  cat > ~/models/Modelfile <<EOF
FROM ./test-model.gguf

SYSTEM """$SYSTEM_PROMPT"""

PARAMETER temperature 0.6
PARAMETER top_p 0.95
PARAMETER top_k 20
EOF
  ollama create test-model -f ~/models/Modelfile

  echo "Live: https://$domain/v1/chat/completions (model: test-model)"
}

main
