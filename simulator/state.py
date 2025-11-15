import json
import asyncio
import time
from typing import Dict, List
from pathlib import Path
from .generators import generate_initial_tower_state

class SimulatorState:
    def __init__(self, data_path: str, tick_seconds: float = 5.0):
        self.data_path = Path(data_path)
        self.tick_seconds = tick_seconds
        self._load_static()
        self.towers = generate_initial_tower_state(self.static_towers)
        self.corridors = self.static_corridors
        self.alerts: Dict[str, dict] = {}
        self._tick_task = None
        self._running = False
        self.last_timestamp = time.time()

    def _load_static(self):
        with open(self.data_path, 'r') as f:
            d = json.load(f)
        self.static_towers = d.get('towers', [])
        self.static_corridors = d.get('corridors', [])

    async def start(self):
        if self._running:
            return
        self._running = True
        self._tick_task = asyncio.create_task(self._loop())

    async def stop(self):
        self._running = False
        if self._tick_task:
            self._tick_task.cancel()
            with suppress_cancel():
                await self._tick_task

    async def _loop(self):
        from .behaviors import tick_update
        while self._running:
            try:
                await tick_update(self)
            except Exception:
                # keep simulator alive in face of errors
                import traceback
                traceback.print_exc()
            await asyncio.sleep(self.tick_seconds)

from contextlib import contextmanager

@contextmanager
def suppress_cancel():
    try:
        yield
    except asyncio.CancelledError:
        pass
