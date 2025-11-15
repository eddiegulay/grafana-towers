import asyncio
import os
import json
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from datetime import datetime, timezone, timedelta
from dateutil import parser as dateutil_parser
import math
import random
from pathlib import Path
from simulator.state import SimulatorState
from models import Tower, TowerMetrics, LatestMetrics, Alert, SummarizeRequest, SummarizeResponse


##########################
# Helper utilities
##########################

MAX_POINTS = 1000


def parse_time_or_default(val: Optional[str], default: datetime) -> datetime:
    if not val:
        return default
    try:
        return dateutil_parser.isoparse(val)
    except Exception:
        # fallback
        return dateutil_parser.parse(val)


def make_time_range(start: datetime, end: datetime, step_seconds: int) -> List[datetime]:
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    points = []
    cur = start
    while cur <= end and len(points) < MAX_POINTS:
        points.append(cur)
        cur = cur + timedelta(seconds=step_seconds)
    return points


def error_response(message: str, detail: str, code: str, status: int = 400):
    return JSONResponse(status_code=status, content={"error": message, "detail": detail, "code": code})


def clamp_points(points: int) -> int:
    return max(1, min(points, MAX_POINTS))


def towers_included_list(tower_ids: Optional[str]) -> List[str]:
    if not tower_ids:
        return list(SIM.towers.keys())
    return [t.strip() for t in tower_ids.split(',') if t.strip()]

def normalize_tid_key(tid: str) -> str:
    """Normalize tower id to keys like 'tower_1' when possible, otherwise sanitize."""
    import re
    m = re.search(r"(\d+)$", tid)
    if m:
        return f"tower_{int(m.group(1))}"
    # fallback: lowercase and replace non-alnum with underscore
    return ''.join(c if c.isalnum() else '_' for c in tid).lower()


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

@app.get('/live/towers', response_model=List[Tower])
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
            'performance_delta': t.get('performance_delta',0.0)
        })
    # return plain list -> FastAPI will validate/serialize using response_model
    return resp

@app.get('/live/metrics', response_model=LatestMetrics)
async def live_metrics():
    # snapshot
    per_tower: Dict[str, Dict[str, Any]] = {}
    for tid, t in SIM.towers.items():
        per_tower[tid] = {
            'users_connected': int(t['users_connected']),
            'latency': t['latency'],
            'download_speed': t['download_speed'],
            'upload_speed': t['upload_speed'],
            'congestion': int(t.get('congestion', 0))
        }
    ts = datetime.now(timezone.utc)
    global_events = {
        'critical_events': sum(1 for t in SIM.towers.values() if t['status'] == 'Critical'),
        'warning_events': sum(1 for t in SIM.towers.values() if t['status'] == 'Warning')
    }
    # Build typed payload; FastAPI will validate against LatestMetrics
    payload = {
        'timestamp': ts,
        'per_tower': per_tower,
        'global_events': global_events
    }
    return payload


@app.get('/live/metrics/users/timeseries')
async def users_timeseries(from_ts: Optional[str] = None, to_ts: Optional[str] = None, step: Optional[int] = 300):
    """Returns aggregated user connection counts across all towers at regular intervals."""
    now = datetime.now(timezone.utc)
    default_from = now - timedelta(hours=6)
    start = parse_time_or_default(from_ts, default_from)
    end = parse_time_or_default(to_ts, now)
    if start >= end:
        return error_response('Invalid time range', 'From timestamp must be before to timestamp', 'INVALID_TIME_RANGE')
    step = int(step or 300)
    points = make_time_range(start, end, step)
    if len(points) == 0:
        return error_response('No points', 'Requested time range produced zero points', 'NO_POINTS')

    data = []
    base_totals = sum(int(t['users_connected']) for t in SIM.towers.values())
    # create gentle diurnal variation based on hour
    for ts in points:
        hour = ts.hour
        # diurnal factor: peak midday/early evening
        factor = 1.0 + 0.2 * math.sin((hour / 24.0) * 2 * math.pi)
        noise = random.randint(-20, 20)
        total = max(0, int(base_totals * factor) + noise)
        data.append({'timestamp': ts.isoformat(), 'users_connected': total})

    metadata = {'from': start.isoformat(), 'to': end.isoformat(), 'step': step, 'points': len(points), 'towers_included': list(SIM.towers.keys())}
    return {'data': data, 'metadata': metadata}


@app.get('/live/metrics/health/scores')
async def health_scores():
    """Compute network health scores."""
    # Speed quality: average download vs baseline (use 60 Mbps as strong baseline)
    avg_speed = sum(t['download_speed'] for t in SIM.towers.values()) / max(1, len(SIM.towers))
    speed_quality = int(max(0, min(100, (avg_speed - 10) / (60 - 10) * 100)))

    # Latency quality: 50ms -> 100, 200ms -> 0
    avg_latency = sum(t['latency'] for t in SIM.towers.values()) / max(1, len(SIM.towers))
    latency_quality = int(max(0, min(100, (200 - avg_latency) / (200 - 50) * 100)))

    # Congestion status: percentage of towers without congestion
    congested = sum(1 for t in SIM.towers.values() if t.get('congestion', 0) == 1)
    congestion_status = int(max(0, min(100, (1.0 - (congested / max(1, len(SIM.towers)))) * 100)))

    # Overall: weighted average
    overall = int(round((0.4 * speed_quality) + (0.3 * latency_quality) + (0.3 * congestion_status)))

    data = [
        {'metric': 'Overall Health', 'score': overall},
        {'metric': 'Speed Quality', 'score': speed_quality},
        {'metric': 'Latency Quality', 'score': latency_quality},
        {'metric': 'Congestion Status', 'score': congestion_status}
    ]
    return {'data': data}


@app.get('/live/metrics/performance/baseline')
async def performance_baseline():
    """Compare current performance vs a simulated baseline."""
    data = []
    for tid, t in SIM.towers.items():
        # simulate baseline as a smoothed value around current with small variation
        baseline_speed = max(1.0, t['download_speed'] * random.uniform(0.8, 1.2))
        baseline_latency = max(1.0, t['latency'] * random.uniform(0.8, 1.2))
        current_speed = t['download_speed']
        current_latency = t['latency']
        perf_delta = ((current_speed / baseline_speed) - 1) * 100 - ((current_latency / baseline_latency) - 1) * 100
        data.append({'tower_id': tid, 'performance_delta': round(perf_delta, 2)})
    return {'data': data}


@app.get('/live/metrics/kpi/overview')
async def kpi_overview():
    """Return several high-level KPIs."""
    total_towers = len(SIM.towers)
    online = sum(1 for t in SIM.towers.values() if t.get('status') != 'Critical')
    availability = int((online / max(1, total_towers)) * 100)

    # SLA compliance: latency <100ms and download >50 Mbps
    sla_ok = sum(1 for t in SIM.towers.values() if t['latency'] < 100 and t['download_speed'] > 50)
    sla = int((sla_ok / max(1, total_towers)) * 100)

    avg_speed = sum(t['download_speed'] for t in SIM.towers.values()) / max(1, total_towers)
    avg_speed_quality = int(max(0, min(100, (avg_speed / 100.0) * 100)))

    # customer satisfaction - synthetic
    csat = int(max(0, min(100, 70 + random.randint(-10, 10))))

    data = [
        {'metric': 'Network Availability', 'value': availability},
        {'metric': 'Towers Online', 'value': online},
        {'metric': 'SLA Compliance', 'value': sla},
        {'metric': 'Avg Speed Quality', 'value': avg_speed_quality},
        {'metric': 'Customer Satisfaction', 'value': csat}
    ]
    return {'data': data}


@app.get('/live/metrics/towers/latency/timeseries')
async def towers_latency_timeseries(from_ts: Optional[str] = None, to_ts: Optional[str] = None, step: Optional[int] = 10800, tower_ids: Optional[str] = None):
    now = datetime.now(timezone.utc)
    default_from = now - timedelta(hours=24)
    start = parse_time_or_default(from_ts, default_from)
    end = parse_time_or_default(to_ts, now)
    if start >= end:
        return error_response('Invalid time range', 'From timestamp must be before to timestamp', 'INVALID_TIME_RANGE')
    step = int(step or 10800)
    points = make_time_range(start, end, step)
    tids = towers_included_list(tower_ids)
    # Build normalized mapping: tid -> key (tower_1, ...)
    key_map = {tid: normalize_tid_key(tid) for tid in tids}
    data = []
    for ts in points:
        row = {'timestamp': ts.isoformat()}
        for tid in tids:
            t = SIM.towers.get(tid)
            key = key_map[tid]
            if not t:
                row[key] = None
                continue
            # simulate slight variation
            row[key] = round(max(0.0, t['latency'] * random.uniform(0.85, 1.25)), 2)
        data.append(row)
    metadata = {'from': start.isoformat(), 'to': end.isoformat(), 'step': step, 'points': len(points), 'towers_included': tids}
    return {'data': data, 'metadata': metadata}


@app.get('/live/metrics/towers/speed/timeseries')
async def towers_speed_timeseries(from_ts: Optional[str] = None, to_ts: Optional[str] = None, step: Optional[int] = 10800, tower_ids: Optional[str] = None, status_filter: Optional[str] = None):
    now = datetime.now(timezone.utc)
    default_from = now - timedelta(hours=24)
    start = parse_time_or_default(from_ts, default_from)
    end = parse_time_or_default(to_ts, now)
    if start >= end:
        return error_response('Invalid time range', 'From timestamp must be before to timestamp', 'INVALID_TIME_RANGE')
    step = int(step or 10800)
    points = make_time_range(start, end, step)
    tids = towers_included_list(tower_ids)
    # apply status filter if provided
    if status_filter:
        sf = status_filter.lower()
        tids = [tid for tid in tids if sf == SIM.towers.get(tid, {}).get('status', '').lower()]

    # normalize keys
    key_map = {tid: normalize_tid_key(tid) for tid in tids}
    data = []
    for ts in points:
        row = {'timestamp': ts.isoformat()}
        for tid in tids:
            t = SIM.towers.get(tid)
            key = key_map[tid]
            if not t:
                row[f"{key}_down"] = None
                row[f"{key}_up"] = None
                continue
            row[f"{key}_down"] = round(max(0.0, t['download_speed'] * random.uniform(0.75, 1.2)), 2)
            row[f"{key}_up"] = round(max(0.0, t['upload_speed'] * random.uniform(0.75, 1.2)), 2)
        data.append(row)
    metadata = {'from': start.isoformat(), 'to': end.isoformat(), 'step': step, 'points': len(points), 'towers_included': tids}
    return {'data': data, 'metadata': metadata}


@app.get('/live/metrics/towers/users/timeseries')
async def towers_users_timeseries(from_ts: Optional[str] = None, to_ts: Optional[str] = None, step: Optional[int] = 10800):
    now = datetime.now(timezone.utc)
    default_from = now - timedelta(hours=24)
    start = parse_time_or_default(from_ts, default_from)
    end = parse_time_or_default(to_ts, now)
    if start >= end:
        return error_response('Invalid time range', 'From timestamp must be before to timestamp', 'INVALID_TIME_RANGE')
    step = int(step or 10800)
    points = make_time_range(start, end, step)
    tids = list(SIM.towers.keys())
    key_map = {tid: normalize_tid_key(tid) for tid in tids}
    data = []
    for ts in points:
        row = {'timestamp': ts.isoformat()}
        for tid in tids:
            t = SIM.towers.get(tid)
            key = key_map[tid]
            if not t:
                row[key] = 0
            else:
                row[key] = int(max(0, t['users_connected'] * random.uniform(0.6, 1.4)))
        data.append(row)
    metadata = {'from': start.isoformat(), 'to': end.isoformat(), 'step': step, 'points': len(points), 'towers_included': tids}
    return {'data': data, 'metadata': metadata}


@app.get('/live/metrics/towers/congestion/timeseries')
async def towers_congestion_timeseries(from_ts: Optional[str] = None, to_ts: Optional[str] = None, step: Optional[int] = 10800, tower_ids: Optional[str] = None):
    now = datetime.now(timezone.utc)
    default_from = now - timedelta(hours=24)
    start = parse_time_or_default(from_ts, default_from)
    end = parse_time_or_default(to_ts, now)
    if start >= end:
        return error_response('Invalid time range', 'From timestamp must be before to timestamp', 'INVALID_TIME_RANGE')
    step = int(step or 10800)
    points = make_time_range(start, end, step)
    tids = towers_included_list(tower_ids)
    key_map = {tid: normalize_tid_key(tid) for tid in tids}
    data = []
    for ts in points:
        row = {'timestamp': ts.isoformat()}
        for tid in tids:
            t = SIM.towers.get(tid)
            key = key_map[tid]
            if not t:
                row[key] = 0
            else:
                # simulate occasional congestion events
                base = t.get('congestion', 0)
                row[key] = 1 if (base == 1 and random.random() < 0.8) or (random.random() < 0.03) else 0
        data.append(row)
    metadata = {'from': start.isoformat(), 'to': end.isoformat(), 'step': step, 'points': len(points), 'towers_included': tids}
    return {'data': data, 'metadata': metadata}


@app.get('/live/summary')
async def live_summary():
    """Aggregate /live/towers, /live/metrics and /live/alerts and call summarizer. If Groq not configured, return simple generated summary."""
    # gather current state
    towers_resp = await live_towers()
    metrics_resp = await live_metrics()
    alerts_resp = await live_alerts()

    # Build SummarizeRequest-like payload
    payload = {
        'towers': towers_resp,
        'latest_metrics': metrics_resp,
        'alerts': alerts_resp
    }

    # Try to POST to existing summarize handler (call function) and return its response if possible
    try:
        # call summarize function directly
        sreq = SummarizeRequest(**payload)
        res = await summarize(sreq)
        # summarize returns dict-like
        if isinstance(res, JSONResponse):
            return res
        return {'summary': res.get('summary') if isinstance(res, dict) else res}
    except Exception as e:
        # fallback simple summary
        num_towers = len(SIM.towers)
        num_alerts = len(SIM.alerts)
        ts = datetime.now(timezone.utc).isoformat()
        summary = f"# Network Situation Summary\n\n{num_towers} towers monitored, {num_alerts} active alerts. Latest metrics timestamp: {metrics_resp.get('timestamp') if isinstance(metrics_resp, dict) else ''}\n"
        return {'summary': summary, 'generated_at': ts, 'data_sources': {'towers': num_towers, 'alerts': num_alerts, 'metrics_timestamp': metrics_resp.get('timestamp') if isinstance(metrics_resp, dict) else ''}}


@app.get('/live/metrics/speed/hourly')
async def speed_hourly(date: Optional[str] = None, tower_id: Optional[str] = None):
    """Returns average download/upload speeds aggregated by hour for the specified date."""
    if date:
        try:
            day = dateutil_parser.isoparse(date).date()
        except Exception:
            return error_response('Invalid date', 'Could not parse date parameter', 'INVALID_DATE')
    else:
        day = datetime.now(timezone.utc).date()

    towers = [tower_id] if tower_id else list(SIM.towers.keys())
    data = []
    # simulate 24 hourly averages
    for h in range(24):
        # aggregate across towers
        downs = []
        ups = []
        for tid in towers:
            t = SIM.towers.get(tid)
            if not t: continue
            base_down = t['download_speed']
            base_up = t['upload_speed']
            # hour-of-day effect
            modifier = 1.0 + 0.15 * math.sin((h / 24.0) * 2 * math.pi)
            downs.append(round(max(0.1, base_down * modifier * random.uniform(0.85, 1.15)),2))
            ups.append(round(max(0.1, base_up * modifier * random.uniform(0.85, 1.15)),2))
        avg_down = round(sum(downs)/len(downs),2) if downs else 0.0
        avg_up = round(sum(ups)/len(ups),2) if ups else 0.0
        data.append({'hour': f"{h:02d}:00", 'download_speed': avg_down, 'upload_speed': avg_up})

    metadata = {'date': str(day), 'towers_included': towers}
    return {'data': data, 'metadata': metadata}


@app.get('/live/metrics/congestion/timeseries')
async def congestion_timeseries(from_ts: Optional[str] = None, to_ts: Optional[str] = None, step: Optional[int] = 3600):
    now = datetime.now(timezone.utc)
    default_from = now - timedelta(hours=24)
    start = parse_time_or_default(from_ts, default_from)
    end = parse_time_or_default(to_ts, now)
    if start >= end:
        return error_response('Invalid time range', 'From timestamp must be before to timestamp', 'INVALID_TIME_RANGE')
    step = int(step or 3600)
    points = make_time_range(start, end, step)
    data = []
    for ts in points:
        # pick sample towers congestion and total users
        users = sum(int(t['users_connected']) for t in SIM.towers.values())
        # congestion if many towers currently congested; add some time variation
        current_congestion = sum(1 for t in SIM.towers.values() if t.get('congestion',0)==1)
        noise = 1 if random.random() < 0.05 else 0
        congest_flag = 1 if (current_congestion > 0 and random.random() < 0.6) or noise else 0
        data.append({'timestamp': ts.isoformat(), 'users_connected': users, 'congestion': congest_flag})
    metadata = {'from': start.isoformat(), 'to': end.isoformat(), 'step': step, 'points': len(points)}
    return {'data': data, 'metadata': metadata}


@app.get('/live/metrics/events/timeseries')
async def events_timeseries(from_ts: Optional[str] = None, to_ts: Optional[str] = None, step: Optional[int] = 7200):
    now = datetime.now(timezone.utc)
    default_from = now - timedelta(hours=24)
    start = parse_time_or_default(from_ts, default_from)
    end = parse_time_or_default(to_ts, now)
    if start >= end:
        return error_response('Invalid time range', 'From timestamp must be before to timestamp', 'INVALID_TIME_RANGE')
    step = int(step or 7200)
    points = make_time_range(start, end, step)
    data = []
    for ts in points:
        # simulate event counts by sampling current alerts and random noise
        critical = sum(1 for a in SIM.alerts.values() if a.get('priority')=='P1')
        warning = sum(1 for a in SIM.alerts.values() if a.get('priority')!='P1')
        # spread counts over time with randomness
        c = max(0, int(random.gauss(critical/3 if critical else 0, 0.5)))
        w = max(0, int(random.gauss(warning/2 if warning else 0, 0.8)))
        data.append({'timestamp': ts.isoformat(), 'critical_events': c, 'warning_events': w})
    metadata = {'from': start.isoformat(), 'to': end.isoformat(), 'step': step, 'points': len(points)}
    return {'data': data, 'metadata': metadata}

@app.get('/live/alerts', response_model=List[Alert])
async def live_alerts():
    # Build alerts list and restrict to the Alert schema fields to ensure OpenAPI compliance
    out = []
    for a in SIM.alerts.values():
        out.append({
            'alert_id': a.get('alert_id'),
            'tower_id': a.get('tower_id'),
            'location': a.get('location'),
            'priority': a.get('priority'),
            'alert_type': a.get('alert_type'),
            'description': a.get('description'),
            'duration_min': int(a.get('duration_min', 0)),
            'action_required': a.get('action_required', '')
        })
    return out


@app.post('/summarize', response_model=SummarizeResponse)
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
    # Convert Pydantic models to plain dicts first to avoid JSON serialization errors
    payload_dict = payload.dict()
    towers_json = json.dumps(payload_dict.get('towers', []), indent=2, ensure_ascii=False, default=str)
    metrics_json = json.dumps(payload_dict.get('latest_metrics', {}), indent=2, ensure_ascii=False, default=str)
    alerts_json = json.dumps(payload_dict.get('alerts', []), indent=2, ensure_ascii=False, default=str)

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

            # Return typed response
            return {"summary": summary_text}

    except APIError as e:
        # groq API error
        raise HTTPException(status_code=502, detail=f'Groq API error: {str(e)}')
    except Exception as e:
        # Generic error
        raise HTTPException(status_code=500, detail=f'Internal error: {str(e)}')

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=True)
