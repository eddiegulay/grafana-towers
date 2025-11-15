import random
import math
import time
from datetime import datetime, timezone
from .weather import roll_weather, apply_weather_impact
from .backhaul import maybe_degrade_corridors

# helper haversine for proximity
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2*R*math.asin(math.sqrt(a))*1000.0

async def tick_update(state):
    """Perform one simulator tick: update towers, propagate waves, produce alerts."""
    timestamp = datetime.now(timezone.utc).isoformat()
    # time-of-day load effect (assume local Stockholm timezone ~UTC+1/+2; use hour from UTC for simplicity)
    utc_hour = datetime.utcnow().hour
    business = 8 <= utc_hour <= 18
    events = []

    # backhaul events may alter towers
    corridor_events = maybe_degrade_corridors(state)
    if corridor_events:
        events.extend(corridor_events)

    # per-tower updates
    MAX_LATENCY = 2000.0
    # decay factor for multipliers each tick (towards 1.0)
    MULTIPLIER_DECAY = 0.80

    for t in state.towers.values():
        # weather roll
        old_weather = t.get('weather', 'Clear')
        t['weather'] = roll_weather(old_weather)

        # baseline user fluctuation
        base = int(t.get('users_connected', 0))
        jitter = int(base * random.uniform(-0.05, 0.05))
        t['users_connected'] = max(0, base + jitter)

        # business hours increase
        if business:
            t['users_connected'] += int(random.uniform(10, 60))

        # occasional local spike
        if random.random() < 0.02:
            spike = int(t['users_connected'] * random.uniform(0.2, 0.6))
            t['users_connected'] += spike
            # register event and bump latency multiplier rather than raw latency
            t.setdefault('latency_multiplier', t.get('latency_multiplier', 1.0))
            t['latency_multiplier'] *= random.uniform(1.15, 1.5)
            events.append({'tower': t['tower_id'], 'type': 'spike', 'amount': spike})

        # ensure base values exist (from generators)
        latency_base = float(t.get('latency_base', max(5.0, t.get('latency', 20.0))))
        download_base = float(t.get('download_base', t.get('download_speed', 50.0)))
        upload_base = float(t.get('upload_base', t.get('upload_speed', 10.0)))

        # decaying multipliers: move multiplier towards 1.0 to simulate recovery
        t['latency_multiplier'] = max(0.5, 1.0 + (t.get('latency_multiplier', 1.0) - 1.0) * MULTIPLIER_DECAY)
        t['download_multiplier'] = max(0.3, 1.0 + (t.get('download_multiplier', 1.0) - 1.0) * MULTIPLIER_DECAY)
        t['upload_multiplier'] = max(0.2, 1.0 + (t.get('upload_multiplier', 1.0) - 1.0) * MULTIPLIER_DECAY)

        # apply weather effects (adjust multipliers)
        apply_weather_impact(t)

        # users effect: higher users increase latency and reduce speeds (applied as factors)
        users_factor = 1.0 + (t['users_connected'] / 10000.0)

        # compute derived metrics from bases and multipliers
        raw_latency = latency_base * t['latency_multiplier'] * users_factor
        raw_download = download_base * t['download_multiplier'] * (1.0 - min(0.8, t['users_connected'] / 2000.0))
        raw_upload = upload_base * t['upload_multiplier'] * (1.0 - min(0.7, t['users_connected'] / 3000.0))

        # clamp and round
        t['latency'] = round(float(max(1.0, min(MAX_LATENCY, raw_latency))), 2)
        t['download_speed'] = round(float(max(0.1, raw_download)), 2)
        t['upload_speed'] = round(float(max(0.05, raw_upload)), 2)

        # performance delta relative to baseline 100
        baseline = 100.0
        perf = (t['download_speed'] / max(1.0, 50.0)) * 100.0
        t['performance_delta'] = round(perf - baseline, 2)

        # congestion logic
        if t['users_connected'] > 800 or t['latency'] > 200 or t.get('backhaul_state') != 'Normal':
            t['congestion'] = 1
        else:
            t['congestion'] = 0

        # status
        if t['congestion'] == 1 or t['performance_delta'] < -50:
            t['status'] = 'Critical'
        elif t['performance_delta'] < -20 or t['latency'] > 120:
            t['status'] = 'Warning'
        else:
            t['status'] = 'Good'

    # propagate latency waves: for any spike or corridor event, propagate to nearby towers
    spikes = [e for e in events if e.get('type') in ('spike','degraded','failing')]
    if spikes:
        for e in spikes:
            if 'tower' in e:
                center = state.towers.get(e['tower'])
                if not center: continue
                for oth in state.towers.values():
                    if oth['tower_id'] == center['tower_id']: continue
                    d = haversine(center['latitude'], center['longitude'], oth['latitude'], oth['longitude'])
                    if d < 3000: # within 3km
                        increase = random.uniform(1.05, 1.35)
                        oth.setdefault('latency_multiplier', oth.get('latency_multiplier', 1.0))
                        oth.setdefault('download_multiplier', oth.get('download_multiplier', 1.0))
                        oth['latency_multiplier'] *= increase
                        oth['download_multiplier'] *= random.uniform(0.85, 0.98)
            elif 'corridor' in e:
                for tid in e['corridor']:
                    base = state.towers.get(tid)
                    if not base: continue
                    for oth in state.towers.values():
                        d = haversine(base['latitude'], base['longitude'], oth['latitude'], oth['longitude'])
                        if d < 8000:
                            oth.setdefault('latency_multiplier', oth.get('latency_multiplier', 1.0))
                            oth['latency_multiplier'] *= random.uniform(1.1, 1.5)

    # alerts generation/cleanup
    new_alerts = {}
    for tid, t in state.towers.items():
        if t['status'] == 'Critical':
            aid = f"ALT-{tid}-{int(time.time())}"
            new_alerts[aid] = {
                'alert_id': aid,
                'tower_id': tid,
                'location': t['location_name'],
                'priority': 'P1',
                'alert_type': 'Network Degradation',
                'description': f"Latency {t['latency']}ms, download {t['download_speed']} Mbps",
                'duration_min': random.randint(1,60),
                'action_required': 'Immediate technician dispatch'
            }
    # keep existing alerts that are still relevant (merge)
    state.alerts = {**state.alerts, **new_alerts}

    state.last_timestamp = timestamp
    # rotate out old alerts occasionally
    keys = list(state.alerts.keys())
    for k in keys:
        if random.random() < 0.05:
            del state.alerts[k]

    # return an event summary
    return {
        'timestamp': timestamp,
        'events': events,
        'alert_count': len(state.alerts)
    }
