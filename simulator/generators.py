import random
from typing import List, Dict

def generate_initial_tower_state(static_towers: List[dict]) -> Dict[str, dict]:
    towers = {}
    for t in static_towers:
        base_users = random.randint(100, 800)
        latency = random.uniform(20, 80)
        download = random.uniform(10, 150)
        upload = max(2.0, download * 0.3)
        towers[t['tower_id']] = {
            'tower_id': t['tower_id'],
            'latitude': t['latitude'],
            'longitude': t['longitude'],
            'users_connected': base_users,
            'latency': latency,
            'download_speed': round(download,2),
            'upload_speed': round(upload,2),
            'location_name': t.get('location_name',''),
            'status': 'Good',
            'weather': 'Clear',
            'backhaul_state': 'Normal',
            'performance_delta': 0.0,
            'congestion': 0
        }
    return towers
