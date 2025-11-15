import random

WEATHERS = ['Clear','Rain','Snow']

def roll_weather(current: str) -> str:
    # small chance to change; snow rarer
    if random.random() < 0.2:
        r = random.random()
        if r < 0.6:
            return 'Clear'
        elif r < 0.9:
            return 'Rain'
        else:
            return 'Snow'
    return current

def apply_weather_impact(tower: dict):
    w = tower.get('weather','Clear')
    if w == 'Snow':
        # latency +10-20%, speed -10-25%
        tower['latency'] *= (1.0 + random.uniform(0.10,0.20))
        tower['download_speed'] *= (1.0 - random.uniform(0.10,0.25))
        tower['upload_speed'] *= (1.0 - random.uniform(0.10,0.25))
    elif w == 'Rain':
        tower['latency'] *= (1.0 + random.uniform(0.05,0.10))
        tower['download_speed'] *= (1.0 - random.uniform(0.05,0.10))
        tower['upload_speed'] *= (1.0 - random.uniform(0.05,0.10))
    # Clear -> no change
