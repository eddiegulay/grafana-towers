from pydantic import BaseModel
from typing import Literal, Dict, Any

class TowerLive(BaseModel):
    tower_id: str
    latitude: float
    longitude: float
    users_connected: int
    latency: float
    download_speed: float
    upload_speed: float
    location_name: str
    status: Literal['Good','Warning','Critical']
    weather: Literal['Clear','Rain','Snow']
    backhaul_state: Literal['Normal','Degraded','Failing']
    performance_delta: float

class MetricsSnapshot(BaseModel):
    timestamp: str
    per_tower: Dict[str, Dict[str, Any]]
    global_events: Dict[str, int]

class AlertItem(BaseModel):
    alert_id: str
    tower_id: str
    location: str
    priority: Literal['P1','P2','P3']
    alert_type: str
    description: str
    duration_min: int
    action_required: str
