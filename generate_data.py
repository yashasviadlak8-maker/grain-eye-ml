import numpy as np, pandas as pd
rng = np.random.default_rng(42)
N = 3000

crop = rng.choice(["wheat", "rice", "maize"], N)
# ASSUMED humidity limits per crop. Placeholder values, replace with literature/lab values.
hum_limit = pd.Series(crop).map({"wheat": 65, "rice": 70, "maize": 68}).values

temp = rng.uniform(15, 42, N)            # DHT22, deg C
hum = rng.uniform(35, 95, N)             # DHT22, % RH
gas = rng.uniform(400, 3000, N)          # MQ-135 raw ADC (ESP32, 0-4095)
gas_rise = rng.normal(0, 150, N)         # change in gas reading over last 6 h

hum_excess = np.clip((hum - hum_limit) / (95 - hum_limit), 0, 1)
temp_excess = np.clip((temp - 25) / 17, 0, 1)
gas_norm = (gas - 400) / 2600
rise_norm = np.clip(gas_rise / 400, 0, 1)

score = (0.35 * hum_excess + 0.25 * temp_excess + 0.25 * gas_norm
         + 0.15 * rise_norm + rng.normal(0, 0.05, N))
risk = pd.cut(score, [-9, 0.25, 0.45, 9], labels=["Low", "Moderate", "High"])

df = pd.DataFrame({"crop": crop, "temperature_C": temp.round(1), "humidity_pct": hum.round(1),
                   "gas_raw": gas.round(0), "gas_rise_6h": gas_rise.round(0), "risk": risk})
df.to_csv("data/grain_synthetic.csv", index=False)
print(df.risk.value_counts())
