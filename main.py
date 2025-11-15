import asyncio
import os
import json
from typing import Any, Dict, List

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import datetime, timezone
from pathlib import Path
from simulator.state import SimulatorState

# Import the async Groq client. The package is added to requirements.txt.
from groq import AsyncGroq, APIError

app = FastAPI(title='Grafana Towers Live Simulator')

DATA_PATH = Path(__file__).parent / 'data' / 'towers.json'
SIM = SimulatorState(str(DATA_PATH), tick_seconds=5.0)

@app.on_event('startup')
async def startup_event():
    await SIM.start()

@app.on_event('shutdown')
async def shutdown_event():
    await SIM.stop()

@app.get('/live/towers')
async def live_towers():
    # map simulator towers to the response schema
    resp = []
    for t in SIM.towers.values():
        resp.append({
            'tower_id': t['tower_id'],
            'latitude': t['latitude'],
            'longitude': t['longitude'],
            'users_connected': int(t['users_connected']),
            'latency': t['latency'],
            'download_speed': t['download_speed'],
            'upload_speed': t['upload_speed'],
            'location_name': t.get('location_name',''),
            'status': t['status'],
            'weather': t.get('weather','Clear'),
            'backhaul_state': t.get('backhaul_state','Normal'),
            'performance_delta': t.get('performance_delta',0.0)
        })
    return JSONResponse(resp)

@app.get('/live/metrics')
async def live_metrics():
    # snapshot
    per_tower = {}
    for tid, t in SIM.towers.items():
        per_tower[tid] = {
            'users_connected': int(t['users_connected']),
            'latency': t['latency'],
            'download_speed': t['download_speed'],
            'upload_speed': t['upload_speed'],
            'congestion': int(t.get('congestion',0))
        }
    ts = datetime.now(timezone.utc).isoformat()
    global_events = {
        'critical_events': sum(1 for t in SIM.towers.values() if t['status']=='Critical'),
        'warning_events': sum(1 for t in SIM.towers.values() if t['status']=='Warning')
    }
    return JSONResponse({'timestamp': ts, 'per_tower': per_tower, 'global_events': global_events})

@app.get('/live/alerts')
async def live_alerts():
    return JSONResponse(list(SIM.alerts.values()))


class SummarizeRequest(BaseModel):
    towers: List[Any]
    latest_metrics: Dict[str, Any]
    alerts: List[Any]


@app.post('/summarize')
async def summarize(payload: SummarizeRequest):
    """Receive towers/latest_metrics/alerts, call Groq LLM, and return a concise summary.

    Request body (JSON):
      {
        "towers": [...],
        "latest_metrics": { ... },
        "alerts": [...]
      }

    Response: { "summary": "..." }
    """

    # Build the prompt from the template with embedded JSON
    towers_json = json.dumps(payload.towers, indent=2, ensure_ascii=False)
    metrics_json = json.dumps(payload.latest_metrics, indent=2, ensure_ascii=False)
    alerts_json = json.dumps(payload.alerts, indent=2, ensure_ascii=False)

    prompt = f"""
You are a telecom network operations assistant.
Generate a concise real-time summary of the network’s current condition.

Use this structure exactly:

1. Situation Overview
2. Key Signals (bullet points)
3. Probable Cause
4. Recommended Actions

### Towers
{towers_json}

### Latest Metrics
{metrics_json}

### Alerts
{alerts_json}

Focus on:
- latency waves
- correlated tower failures
- load spikes
- speed degradation
- congestion events
- weather impact
- backhaul degradation
- SLA risk

Keep it short, direct, and operational.
"""

    # Instantiate Groq async client and call chat completion.
    api_key = os.environ.get('GROQ_API_KEY')
    if not api_key:
        raise HTTPException(status_code=500, detail='GROQ_API_KEY not configured in environment')

    try:
        async with AsyncGroq(api_key=api_key) as client:
            resp = await client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="openai/gpt-oss-20b",
                temperature=0.3,
                max_completion_tokens=512,
                reasoning_effort="medium",
                top_p=1,
                stream=False,
            )

            # Extract model output (follow Groq client's response shape)
            summary_text = None
            try:
                summary_text = resp.choices[0].message.content
            except Exception:
                # Fallback to attempt dict-style access
                summary_text = getattr(resp.choices[0].message, 'content', None)

            if not summary_text:
                raise HTTPException(status_code=500, detail='No summary returned from model')

            return JSONResponse({"summary": summary_text})

    except APIError as e:
        # groq API error
        raise HTTPException(status_code=502, detail=f'Groq API error: {str(e)}')
    except Exception as e:
        # Generic error
        raise HTTPException(status_code=500, detail=f'Internal error: {str(e)}')

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=True)
