#!/usr/bin/env bash
set -euo pipefail

# Helper script to configure and run local ngrok from the repo.
# Usage: copy .env.example to .env and set NGROK_AUTHTOKEN, then run:
# ./start-ngrok.sh

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

# load .env if present
if [ -f .env ]; then
  # shellcheck disable=SC1091
  source .env
fi

if [ -z "${NGROK_AUTHTOKEN:-}" ]; then
  echo "NGROK_AUTHTOKEN is not set. Edit .env or set the env var and rerun.\nYou can run ngrok without authtoken but it's limited."
  echo "Starting ngrok http 8000 without authtoken..."
  ./ngrok http 8000 --log=stdout
else
  echo "Setting authtoken and starting ngrok..."
  ./ngrok authtoken "$NGROK_AUTHTOKEN" || true
  ./ngrok http 8000 --log=stdout
fi
