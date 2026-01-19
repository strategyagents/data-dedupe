#!/usr/bin/env bash
set -euo pipefail

if [[ "${APP_MODE:-cli}" == "web" ]]; then
  exec python -m src.web_app
fi

exec python -m src.main
