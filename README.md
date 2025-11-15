FastAPI Real-time Telecom Simulator

This lightweight FastAPI service simulates live telecom tower telemetry for Grafana panels.

Endpoints:
- GET /live/towers  — per-tower real-time telemetry
- GET /live/metrics — timeseries-style snapshot per refresh tick
- GET /live/alerts  — active alerts

Run locally:

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Grafana should poll endpoints every 5–10s.

Docker / ngrok
----------------

Build and run with Docker (local):

```bash
# build image
docker build -t grafana-towers:latest .

# run with docker-compose (recommended for ngrok tunnel)
cp .env.example .env    # edit and add NGROK_AUTHTOKEN if you want ngrok
docker-compose up --build
```

After `docker-compose up` completes, the FastAPI app will be available at http://localhost:8000 and the ngrok web UI is at http://localhost:4040 (the public ngrok forwarding URL is shown in the ngrok logs or the web UI).

CI / Docker registry
---------------------

A GitHub Actions workflow is included at `.github/workflows/docker-publish.yml` to build and push the image to GitHub Container Registry `ghcr.io` on pushes to `main`. It uses the repository's `GITHUB_TOKEN` for authentication.

Notes
-----
- The `docker-compose.yml` includes a lightweight `ngrok` service (image `wernight/ngrok:latest`). Provide `NGROK_AUTHTOKEN` in a local `.env` or your environment if you want a public tunnel.
- Use Grafana's Infinity/JSON API plugin to poll `/live/towers`, `/live/metrics`, and `/live/alerts` every 5–10s.
