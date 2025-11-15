import random

def maybe_degrade_corridors(state):
    # small chance of corridor degradation
    events = []
    for corridor in state.corridors:
        if random.random() < 0.03:
            # degrade for a while
            for tid in corridor:
                t = state.towers.get(tid)
                if not t: continue
                t['backhaul_state'] = 'Degraded'
                # apply effects via multipliers so they can decay later
                t.setdefault('latency_multiplier', t.get('latency_multiplier', 1.0))
                t.setdefault('download_multiplier', t.get('download_multiplier', 1.0))
                t['latency_multiplier'] *= random.uniform(1.2, 1.6)
                t['download_multiplier'] *= random.uniform(0.6, 0.9)
                events.append({'corridor': corridor, 'type': 'degraded'})
        elif random.random() < 0.01:
            for tid in corridor:
                t = state.towers.get(tid)
                if not t: continue
                t['backhaul_state'] = 'Failing'
                t.setdefault('latency_multiplier', t.get('latency_multiplier', 1.0))
                t.setdefault('download_multiplier', t.get('download_multiplier', 1.0))
                t['latency_multiplier'] *= random.uniform(1.6, 2.0)
                t['download_multiplier'] *= random.uniform(0.3, 0.6)
                events.append({'corridor': corridor, 'type': 'failing'})
        else:
            # small chance to recover
            if random.random() < 0.05:
                for tid in corridor:
                    t = state.towers.get(tid)
                    if not t: continue
                    t['backhaul_state'] = 'Normal'
    return events
