import pandas as pd
import numpy as np
import time

np.random.seed(42)
rows = []
for _ in range(1000):
    soil_moisture = np.random.uniform(10, 60)
    soil_temp = np.random.uniform(10, 35)
    soil_ph = np.random.uniform(5.5, 8.0)
    tank_level = np.random.uniform(0, 100)
    ambient_humidity = np.random.uniform(20, 90)
    ambient_temp = np.random.uniform(10, 40)
    rain_next_48h = np.random.uniform(0, 20)
    # Simple logic for target
    irrigate = int(
        soil_moisture < 30 and
        rain_next_48h < 5 and
        tank_level > 10 and
        6.0 <= soil_ph <= 7.5
    )
    # Add a realistic timestamp (current time minus random offset)
    timestamp = int(time.time()) - np.random.randint(0, 60*60*24*365)
    rows.append([
        soil_moisture, soil_temp, soil_ph, tank_level,
        ambient_humidity, ambient_temp, rain_next_48h, irrigate, timestamp
    ])

df = pd.DataFrame(rows, columns=[
    "soil_moisture", "soil_temp", "soil_ph", "tank_level",
    "ambient_humidity", "ambient_temp", "rain_next_48h", "irrigate", "timestamp"
])
df.to_csv("fabricated_irrigation_data.csv", index=False)