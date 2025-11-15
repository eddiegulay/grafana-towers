#!/usr/bin/env python3
"""Run a few simulator ticks and validate generated metric ranges.

Usage: python3 scripts/validate_sim.py
"""
import asyncio
import math
from simulator.state import SimulatorState
from simulator.behaviors import tick_update

async def run_checks():
    s = SimulatorState('data/towers.json', tick_seconds=1.0)
    problems = []

    # run a few ticks to let multipliers apply and decay
    for i in range(6):
        await tick_update(s)

    for tid, t in s.towers.items():
        lat = t.get('latency')
        down = t.get('download_speed')
        up = t.get('upload_speed')
        if lat is None or not math.isfinite(lat) or lat < 0 or lat > 10000:
            problems.append(f"Tower {tid} invalid latency: {lat}")
        if down is None or not math.isfinite(down) or down < 0 or down > 10000:
            problems.append(f"Tower {tid} invalid download: {down}")
        if up is None or not math.isfinite(up) or up < 0 or up > 10000:
            problems.append(f"Tower {tid} invalid upload: {up}")

    if problems:
        print("Validation FAILED:")
        for p in problems:
            print(' -', p)
        return 1
    else:
        print("Validation passed: all tower metrics within sane ranges")
        for tid, t in s.towers.items():
            print(f"{tid}: latency={t['latency']}ms, down={t['download_speed']}Mbps, up={t['upload_speed']}Mbps, status={t['status']}")
        return 0

if __name__ == '__main__':
    code = asyncio.run(run_checks())
    raise SystemExit(code)
