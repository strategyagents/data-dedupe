#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f ".env" ]]; then
  cp .env.example .env
  echo "Created .env. Update it with your API key or Ollama settings, then re-run."
  exit 1
fi

echo "Starting demo..."
docker compose up --build
