import asyncio
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from datetime import datetime, timezone
from pathlib import Path
from simulator.state import SimulatorState

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

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=True)
