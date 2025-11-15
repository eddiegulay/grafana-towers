#!/usr/bin/env python3
"""Quick test: build a sample SummarizeRequest (using models) and run the same serialization used in /summarize endpoint.
This avoids calling Groq and only verifies JSON serialization works for pydantic models."""
from models import Tower, LatestMetrics, TowerMetrics, GlobalEvents, Alert, SummarizeRequest
from datetime import datetime, timezone
import json

# create few towers
 towers = [
     Tower(tower_id='Tower-1', latitude=59.33, longitude=18.06, users_connected=200, latency=45.2, download_speed=50.1, upload_speed=10.2, location_name='A', status='Good', performance_delta=0.0),
     Tower(tower_id='Tower-5', latitude=59.34, longitude=18.05, users_connected=600, latency=120.5, download_speed=20.3, upload_speed=5.1, location_name='B', status='Critical', performance_delta=-40.0),
 ]

per_tower = {
    'Tower-1': TowerMetrics(users_connected=200, latency=45.2, download_speed=50.1, upload_speed=10.2, congestion=0),
    'Tower-5': TowerMetrics(users_connected=600, latency=120.5, download_speed=20.3, upload_speed=5.1, congestion=1),
}

latest = LatestMetrics(timestamp=datetime.now(timezone.utc), per_tower=per_tower, global_events=GlobalEvents(critical_events=1, warning_events=0))

alerts = [Alert(alert_id='ALT-1', tower_id='Tower-5', location='B', priority='P1', alert_type='Network Degradation', description='High latency', duration_min=15, action_required='Dispatch')]

req = SummarizeRequest(towers=towers, latest_metrics=latest, alerts=alerts)

# serialization as in the endpoint
payload_dict = req.dict()
try:
    towers_json = json.dumps(payload_dict.get('towers', []), indent=2, ensure_ascii=False, default=str)
    metrics_json = json.dumps(payload_dict.get('latest_metrics', {}), indent=2, ensure_ascii=False, default=str)
    alerts_json = json.dumps(payload_dict.get('alerts', []), indent=2, ensure_ascii=False, default=str)
    print('Serialization OK')
    print('--- towers_json ---')
    print(towers_json)
except Exception as e:
    print('Serialization FAILED:', e)
