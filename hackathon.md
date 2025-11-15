## Grafana Towers — Hackathon Summary

Date: 2025-11-15

Short version for the jury

We needed telecom telemetry quickly but couldn't get a real carrier dataset in time, so we built a realtime simulator that generates plausible tower telemetry for visualization and AI-assisted analysis. The judges won't read code — they want a clear demo that shows data, insights, and operator value.

What we simulated (the data you see in the dashboard)

- Per-tower snapshot: tower id, geographic location.
- Telemetry timeseries: latency (ms), download/upload speeds (Mbps), active users.
- Operational signals: congestion flag, status (Good/Warning/Critical), performance delta vs baseline.
- Supporting diagnostics: backhaul metrics (delay, jitter, errors), packet loss, handover success/failure rates.
- Environment: weather states that affect performance (Clear/Rain/Snow), predefined corridors to simulate linked degradation, and generated alerts when status degrades.

Why a simulator

Real telecom datasets are sensitive and hard to ship for a quick prototype. The simulator lets us:
- Produce continuous realtime data for Grafana panels and annotations.
- Create reproducible scenarios (spikes, corridor/backhaul failures, weather events) that demonstrate monitoring and triage.

How AI adds value

- We use  LLM integration to convert telemetry into concise, actionable operator text:
	- A short situational summary (what's happening now and why it matters).
	- Per-tower root-cause style hints and recommended next steps.
- Practical choices: prompts are kept small (only the most recent points), outputs are deterministic-friendly (low temperature), and there are synthetic fallbacks and caching so the demo never breaks if the model is unavailable.
