from pydantic import BaseModel
from typing import Dict, List
from datetime import datetime


class Tower(BaseModel):
    tower_id: str
    latitude: float
    longitude: float
    users_connected: int
    latency: float
    download_speed: float
    upload_speed: float
    location_name: str
    status: str
    performance_delta: float


class TowerMetrics(BaseModel):
    users_connected: int
    latency: float
    download_speed: float
    upload_speed: float
    congestion: int


class GlobalEvents(BaseModel):
    critical_events: int
    warning_events: int


class LatestMetrics(BaseModel):
    timestamp: datetime
    per_tower: Dict[str, TowerMetrics]
    global_events: GlobalEvents


class Alert(BaseModel):
    alert_id: str
    tower_id: str
    location: str
    priority: str
    alert_type: str
    description: str
    duration_min: int
    action_required: str


class SummarizeRequest(BaseModel):
    towers: List[Tower]
    latest_metrics: LatestMetrics
    alerts: List[Alert]


class SummarizeResponse(BaseModel):
    summary: str
