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


def _last_n_entries(obj, n: int = 7):
    """Return the last n entries for lists or for dicts with common list fields.

    - If obj is a dict and contains a 'data' or 'alerts' key with a list, return that list's last n items.
    - If obj is a list, return its last n items.
    - Otherwise return obj unchanged.
    """
    if isinstance(obj, dict):
        for key in ('data', 'alerts'):
            if key in obj and isinstance(obj[key], list):
                return obj[key][-n:]
        return obj
    if isinstance(obj, list):
        return obj[-n:]
    return obj


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
async def live_summary(from_ts: Optional[str] = None, to_ts: Optional[str] = None, step: Optional[int] = 3600, tower_ids: Optional[str] = None):
    """Aggregate data over a requested time window and call the LLM summarizer.

    Query parameters (optional): from_ts, to_ts, step, tower_ids
    If GROQ_API_KEY is not configured, falls back to a short generated summary.
    """

    now = datetime.now(timezone.utc)
    default_from = now - timedelta(hours=6)
    start = parse_time_or_default(from_ts, default_from)
    end = parse_time_or_default(to_ts, now)
    if start >= end:
        return error_response('Invalid time range', 'From timestamp must be before to timestamp', 'INVALID_TIME_RANGE')

    step = int(step or 3600)

    # Gather timeseries snapshots by calling existing helpers
    try:
        latency_ts = await towers_latency_timeseries(from_ts=start.isoformat(), to_ts=end.isoformat(), step=step, tower_ids=tower_ids)
        speed_ts = await towers_speed_timeseries(from_ts=start.isoformat(), to_ts=end.isoformat(), step=step, tower_ids=tower_ids)
        users_ts = await towers_users_timeseries(from_ts=start.isoformat(), to_ts=end.isoformat(), step=step)
        congestion_ts = await towers_congestion_timeseries(from_ts=start.isoformat(), to_ts=end.isoformat(), step=step, tower_ids=tower_ids)
        events_ts = await events_timeseries(from_ts=start.isoformat(), to_ts=end.isoformat(), step=step)
        towers_resp = await live_towers()
        alerts_resp = await live_alerts()
    except Exception as e:
        num_towers = len(SIM.towers)
        num_alerts = len(SIM.alerts)
        ts = datetime.now(timezone.utc).isoformat()
        summary = f"# Network Situation Summary\n\n{num_towers} towers monitored, {num_alerts} active alerts.\n"
        return {'summary': summary, 'generated_at': ts, 'data_sources_error': str(e)}

    # Build prompt with embedded JSON for the LLM
    towers_json = json.dumps(towers_resp, indent=2, ensure_ascii=False, default=str)
    # Reduce timeseries size to the last 7 points to keep prompts short
    latency_payload = _last_n_entries(latency_ts)
    speed_payload = _last_n_entries(speed_ts)
    users_payload = _last_n_entries(users_ts)
    congestion_payload = _last_n_entries(congestion_ts)
    events_payload = _last_n_entries(events_ts)

    latency_json = json.dumps(latency_payload if not isinstance(latency_payload, dict) else latency_payload, indent=2, ensure_ascii=False, default=str)
    speed_json = json.dumps(speed_payload if not isinstance(speed_payload, dict) else speed_payload, indent=2, ensure_ascii=False, default=str)
    users_json = json.dumps(users_payload if not isinstance(users_payload, dict) else users_payload, indent=2, ensure_ascii=False, default=str)
    congestion_json = json.dumps(congestion_payload if not isinstance(congestion_payload, dict) else congestion_payload, indent=2, ensure_ascii=False, default=str)
    events_json = json.dumps(events_payload if not isinstance(events_payload, dict) else events_payload, indent=2, ensure_ascii=False, default=str)

    prompt = f"""
You are a telecom network operations assistant. Produce a concise operational summary for the time window {start.isoformat()} to {end.isoformat()}.

Return sections: 1) Situation Overview, 2) Key Signals (bullets), 3) Probable Cause, 4) Recommended Actions.

### Towers (current snapshot)
{towers_json}

### Latency timeseries (per-tower)
{latency_json}

### Speed timeseries (per-tower)
{speed_json}

### Users timeseries (per-tower)
{users_json}

### Congestion timeseries (per-tower)
{congestion_json}

### Events timeseries
{events_json}

Focus on correlated failures, load spikes, latency waves, speed degradation, congestion, backhaul issues, and SLA risk. Keep it short and actionable.
"""

    # If no GROQ key is configured, return a short synthetic summary instead of calling the model
    if not os.environ.get('GROQ_API_KEY'):
        num_towers = len(SIM.towers)
        num_alerts = len(SIM.alerts)
        ts = datetime.now(timezone.utc).isoformat()
        summary = f"# Network Situation Summary\n\n{num_towers} towers monitored, {num_alerts} active alerts.\n"
        return {'summary': summary, 'generated_at': ts, 'data_sources': {'towers': num_towers, 'alerts': num_alerts}}

    try:
        summary_text = await _call_groq(prompt)
        return {
            'summary': summary_text,
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'from': start.isoformat(),
            'to': end.isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        num_towers = len(SIM.towers)
        num_alerts = len(SIM.alerts)
        ts = datetime.now(timezone.utc).isoformat()
        summary = f"# Network Situation Summary\n\n{num_towers} towers monitored, {num_alerts} active alerts.\n"
        return {
            'summary': summary,
            'generated_at': ts,
            'data_sources': {'towers': num_towers, 'alerts': num_alerts},
            'llm_error': str(e)
        }


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


##########################
# Per-tower drill-down API
##########################


def _map_status_numeric(status_str: str) -> int:
    s = (status_str or '').lower()
    if s in ('good', 'healthy', 'ok'):
        return 1
    if s in ('warning', 'degraded'):
        return 2
    if s in ('critical', 'down', 'failed'):
        return 3
    return 2


@app.get('/api/towers/{tower_id}/status')
async def tower_status(tower_id: str):
    t = SIM.towers.get(tower_id)
    if not t:
        raise HTTPException(status_code=404, detail='Tower not found')
    now = datetime.now(timezone.utc)
    # Simulate uptime and last_maintenance if not present
    uptime_hours = int(random.randint(24, 24 * 90))
    last_maint = (now - timedelta(hours=random.randint(24, 24 * 90))).isoformat()
    resp = {
        'tower_id': t['tower_id'],
        'location_name': t.get('location_name', ''),
        'latitude': t.get('latitude'),
        'longitude': t.get('longitude'),
        'status': _map_status_numeric(t.get('status', 'Warning')),
        'uptime_hours': uptime_hours,
        'last_maintenance': last_maint,
        'backhaul_type': 'fiber' if random.random() < 0.7 else 'microwave',
        'supported_technologies': ['5G', 'LTE'],
        'timestamp': now.isoformat()
    }
    return resp


@app.get('/api/towers/{tower_id}/metrics/current')
async def tower_metrics_current(tower_id: str):
    t = SIM.towers.get(tower_id)
    if not t:
        raise HTTPException(status_code=404, detail='Tower not found')
    now = datetime.now(timezone.utc)
    congestion_percent = round(100.0 * (1.0 if t.get('congestion', 0) else random.random() * 0.4), 1)
    health_score = max(0.0, min(100.0, 100.0 - (t.get('latency', 0) / 5.0) + (t.get('download_speed', 0) / 5.0) * 0.2))
    return {
        'tower_id': t['tower_id'],
        'timestamp': now.isoformat(),
        'latency_ms': float(t.get('latency', 0.0)),
        'download_speed_mbps': float(t.get('download_speed', 0.0)),
        'upload_speed_mbps': float(t.get('upload_speed', 0.0)),
        'active_users': int(t.get('users_connected', 0)),
        'congestion_percent': round(congestion_percent, 1),
        'health_score': round(health_score, 1)
    }


@app.get('/api/towers/{tower_id}/metrics/latency/timeseries')
async def tower_latency_timeseries(tower_id: str, from_ts: Optional[str] = None, to_ts: Optional[str] = None, step: Optional[int] = 60):
    t = SIM.towers.get(tower_id)
    if not t:
        raise HTTPException(status_code=404, detail='Tower not found')
    now = datetime.now(timezone.utc)
    default_from = now - timedelta(hours=6)
    start = parse_time_or_default(from_ts, default_from)
    end = parse_time_or_default(to_ts, now)
    if start >= end:
        return error_response('Invalid time range', 'From timestamp must be before to timestamp', 'INVALID_TIME_RANGE')
    step = int(step or 60)
    points = make_time_range(start, end, step)
    data = []
    base = float(t.get('latency', 20.0))
    for ts in points:
        # small random walk around base, occasional spike
        val = base * random.uniform(0.9, 1.2)
        if random.random() < 0.01:
            val *= random.uniform(1.5, 3.0)
        data.append({'timestamp': ts.isoformat(), 'value': round(val, 2)})
    return {'tower_id': tower_id, 'metric': 'latency_ms', 'data': data}


@app.get('/api/towers/{tower_id}/metrics/users-congestion/timeseries')
async def tower_users_congestion_timeseries(tower_id: str, from_ts: Optional[str] = None, to_ts: Optional[str] = None, step: Optional[int] = 60):
    t = SIM.towers.get(tower_id)
    if not t:
        raise HTTPException(status_code=404, detail='Tower not found')
    now = datetime.now(timezone.utc)
    default_from = now - timedelta(hours=6)
    start = parse_time_or_default(from_ts, default_from)
    end = parse_time_or_default(to_ts, now)
    if start >= end:
        return error_response('Invalid time range', 'From timestamp must be before to timestamp', 'INVALID_TIME_RANGE')
    step = int(step or 60)
    points = make_time_range(start, end, step)
    data = []
    base_users = int(t.get('users_connected', 100))
    for ts in points:
        users = max(0, int(base_users * random.uniform(0.7, 1.4)))
        # congestion if users spike or existing congestion flag
        congestion_active = 1 if (t.get('congestion', 0) == 1 and random.random() < 0.8) or (users > base_users * 1.2) else 0
        data.append({'timestamp': ts.isoformat(), 'active_users': users, 'congestion_active': congestion_active})
    return {'tower_id': tower_id, 'data': data}


@app.get('/api/towers/{tower_id}/metrics/packet-loss/timeseries')
async def tower_packet_loss_timeseries(tower_id: str, from_ts: Optional[str] = None, to_ts: Optional[str] = None, step: Optional[int] = 60):
    t = SIM.towers.get(tower_id)
    if not t:
        raise HTTPException(status_code=404, detail='Tower not found')
    now = datetime.now(timezone.utc)
    default_from = now - timedelta(hours=6)
    start = parse_time_or_default(from_ts, default_from)
    end = parse_time_or_default(to_ts, now)
    if start >= end:
        return error_response('Invalid time range', 'From timestamp must be before to timestamp', 'INVALID_TIME_RANGE')
    step = int(step or 60)
    points = make_time_range(start, end, step)
    data = []
    for ts in points:
        # small baseline loss with occasional spikes
        up = round(random.uniform(0.0, 1.2), 2)
        down = round(random.uniform(0.0, 1.0), 2)
        if random.random() < 0.02:
            up = round(up + random.uniform(1.0, 4.0), 2)
            down = round(down + random.uniform(0.5, 3.0), 2)
        data.append({'timestamp': ts.isoformat(), 'uplink_loss_percent': up, 'downlink_loss_percent': down})
    return {'tower_id': tower_id, 'data': data}


@app.get('/api/towers/{tower_id}/metrics/backhaul/timeseries')
async def tower_backhaul_timeseries(tower_id: str, from_ts: Optional[str] = None, to_ts: Optional[str] = None, step: Optional[int] = 60):
    t = SIM.towers.get(tower_id)
    if not t:
        raise HTTPException(status_code=404, detail='Tower not found')
    now = datetime.now(timezone.utc)
    default_from = now - timedelta(hours=6)
    start = parse_time_or_default(from_ts, default_from)
    end = parse_time_or_default(to_ts, now)
    if start >= end:
        return error_response('Invalid time range', 'From timestamp must be before to timestamp', 'INVALID_TIME_RANGE')
    step = int(step or 60)
    points = make_time_range(start, end, step)
    data = []
    for ts in points:
        delay = round(random.uniform(5.0, 30.0) * (1.0 if t.get('backhaul_state','Normal')=='Normal' else random.uniform(1.5,3.0)),2)
        jitter = round(random.uniform(0.5, 4.0),2)
        errors = round(random.uniform(0.0, 0.2) * (1.0 if t.get('backhaul_state','Normal')=='Normal' else random.uniform(1.0,5.0)),3)
        data.append({'timestamp': ts.isoformat(), 'delay_ms': delay, 'jitter_ms': jitter, 'errors_per_second': errors})
    return {'tower_id': tower_id, 'data': data}


@app.get('/api/towers/{tower_id}/metrics/handover/timeseries')
async def tower_handover_timeseries(tower_id: str, from_ts: Optional[str] = None, to_ts: Optional[str] = None, step: Optional[int] = 60):
    t = SIM.towers.get(tower_id)
    if not t:
        raise HTTPException(status_code=404, detail='Tower not found')
    now = datetime.now(timezone.utc)
    default_from = now - timedelta(hours=6)
    start = parse_time_or_default(from_ts, default_from)
    end = parse_time_or_default(to_ts, now)
    if start >= end:
        return error_response('Invalid time range', 'From timestamp must be before to timestamp', 'INVALID_TIME_RANGE')
    step = int(step or 60)
    points = make_time_range(start, end, step)
    data = []
    for ts in points:
        attempts = max(0, int(random.gauss(100, 30)))
        failures = max(0, int(attempts * random.uniform(0.0, 0.05)))
        success_rate = 100.0 * (1.0 - (failures / attempts)) if attempts > 0 else 100.0
        data.append({'timestamp': ts.isoformat(), 'success_rate_percent': round(success_rate,2), 'attempts': attempts, 'failures': failures})
    return {'tower_id': tower_id, 'data': data}


@app.get('/api/towers/{tower_id}/alerts')
async def tower_alerts(tower_id: str, limit: Optional[int] = 50, status: Optional[str] = 'active'):
    # SIM.alerts contains active alerts only; support basic filtering
    all_alerts = [a for a in SIM.alerts.values() if a.get('tower_id') == tower_id]
    if status and status != 'all':
        # only support 'active' in this simulator; others map to empty
        if status == 'resolved':
            all_alerts = []
    limit = int(limit or 50)
    out = []
    for a in all_alerts[:limit]:
        out.append({
            'alert_id': a.get('alert_id'),
            'priority': 1 if a.get('priority')=='P1' else (2 if a.get('priority')=='P2' else 3),
            'alert_type': a.get('alert_type'),
            'description': a.get('description'),
            'started_at': (datetime.now(timezone.utc) - timedelta(minutes=a.get('duration_min', 0))).isoformat(),
            'duration_minutes': int(a.get('duration_min', 0)),
            'status': 'active',
            'action_required': a.get('action_required', '')
        })
    return {'tower_id': tower_id, 'alerts': out}


@app.get('/api/towers/{tower_id}/analysis/root-cause')
async def tower_root_cause(tower_id: str, from_ts: Optional[str] = None, to_ts: Optional[str] = None):
    # Use timeseries data across the requested window and call the LLM to produce a focused root-cause-style summary.
    t = SIM.towers.get(tower_id)
    if not t:
        raise HTTPException(status_code=404, detail='Tower not found')
    now = datetime.now(timezone.utc)
    to_dt = parse_time_or_default(to_ts, now)
    from_dt = parse_time_or_default(from_ts, now - timedelta(hours=6))

    # Collect relevant timeseries for this tower
    try:
        latency = await tower_latency_timeseries(tower_id, from_ts=from_dt.isoformat(), to_ts=to_dt.isoformat(), step=60)
        users_cong = await tower_users_congestion_timeseries(tower_id, from_ts=from_dt.isoformat(), to_ts=to_dt.isoformat(), step=60)
        packet = await tower_packet_loss_timeseries(tower_id, from_ts=from_dt.isoformat(), to_ts=to_dt.isoformat(), step=60)
        backhaul = await tower_backhaul_timeseries(tower_id, from_ts=from_dt.isoformat(), to_ts=to_dt.isoformat(), step=60)
        alerts = await tower_alerts(tower_id)
    except Exception as e:
        # revert to synthetic analysis if any collection fails
        summary = 'Insufficient data to perform LLM-based analysis; returning synthetic result.'
        # return insights as an id-keyed dict for easier consumption
        insights = {'error': str(e)}
        return {
            'tower_id': tower_id,
            'analysis_timestamp': now.isoformat(),
            'time_window': {'from': from_dt.isoformat(), 'to': to_dt.isoformat()},
            'summary': summary,
            'insights': insights
        }

    # Build prompt
    try:
        t_json = json.dumps(t, indent=2, ensure_ascii=False, default=str)
        # Keep only the last 7 entries for each timeseries to reduce prompt size
        latency_payload = _last_n_entries(latency)
        users_payload = _last_n_entries(users_cong)
        packet_payload = _last_n_entries(packet)
        backhaul_payload = _last_n_entries(backhaul)
        alerts_payload = _last_n_entries(alerts)

        latency_json = json.dumps(latency_payload if not isinstance(latency_payload, dict) else latency_payload, indent=2, ensure_ascii=False, default=str)
        users_json = json.dumps(users_payload if not isinstance(users_payload, dict) else users_payload, indent=2, ensure_ascii=False, default=str)
        packet_json = json.dumps(packet_payload if not isinstance(packet_payload, dict) else packet_payload, indent=2, ensure_ascii=False, default=str)
        backhaul_json = json.dumps(backhaul_payload if not isinstance(backhaul_payload, dict) else backhaul_payload, indent=2, ensure_ascii=False, default=str)
        alerts_json = json.dumps(alerts_payload if not isinstance(alerts_payload, dict) else alerts_payload, indent=2, ensure_ascii=False, default=str)

        prompt = f"""
You are a telecom network operations assistant. Provide a short root-cause style analysis for tower {tower_id} over the window {from_dt.isoformat()} to {to_dt.isoformat()}.

Include: Situation (1-2 sentences), Key signals (bullets), Most likely root cause, Suggested next actions.

### Tower
{t_json}

### Latency timeseries
{latency_json}

### Users / congestion timeseries
{users_json}

### Packet loss timeseries
{packet_json}

### Alerts
{alerts_json}
"""

        # If no GROQ key is configured, return a short synthetic analysis instead of calling the model
        if not os.environ.get('GROQ_API_KEY'):
            # lightweight synthetic analysis
            summary = 'LLM not configured; returning synthetic root-cause hint. Check tower metrics and backhaul state.'
            insights = []
            if t.get('backhaul_state','Normal') != 'Normal':
                insights = {'backhaul': 'backhaul_state != Normal'}
            elif t.get('congestion',0) == 1:
                insights = {'congestion': f"users_connected={t.get('users_connected',0)}"}
            else:
                insights = {'noise': 'no alerts, slight latency variance'}
            return {
                'tower_id': tower_id,
                'analysis_timestamp': now.isoformat(),
                'time_window_from': from_dt.isoformat(), 
                'time_window_to': to_dt.isoformat(),
                'summary': summary,
                'insights': insights
            }

        summary_text = await _call_groq(prompt)
        return [{'tower_id': tower_id, 'analysis_timestamp': now.isoformat(), 'time_window': {'from': from_dt.isoformat(), 'to': to_dt.isoformat()}, 'summary': summary_text}]
    except HTTPException:
        raise
    except Exception as e:
        # fallback synthetic
        summary = 'LLM summarization failed; returning synthetic analysis.'
        insights = {'error': str(e)}
        return [{
            'tower_id': tower_id,
            'analysis_timestamp': now.isoformat(),
            'time_window_from': from_dt.isoformat(), 
            'time_window_to': to_dt.isoformat(),
            'summary': summary,
            'insights': insights
        }]


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


async def _call_groq(prompt: str) -> str:
    """Helper to call the Groq async client and return the model content string.

    Raises HTTPException on errors so callers can surface appropriate HTTP responses.
    """
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

            summary_text = None
            try:
                summary_text = resp.choices[0].message.content
            except Exception:
                summary_text = getattr(resp.choices[0].message, 'content', None)

            if not summary_text:
                raise HTTPException(status_code=500, detail='No summary returned from model')
            return summary_text

    except APIError as e:
        raise HTTPException(status_code=502, detail=f'Groq API error: {str(e)}')
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Internal error: {str(e)}')

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=True)
