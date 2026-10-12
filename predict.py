import sys, joblib, pandas as pd
model = joblib.load("model/grain_rf.pkl")
crop = sys.argv[1]
t, h, g, r = map(float, sys.argv[2:6])
x = pd.DataFrame([[{"wheat": 0, "rice": 1, "maize": 2}[crop], t, h, g, r]],
                 columns=["crop_code", "temperature_C", "humidity_pct", "gas_raw", "gas_rise_6h"])
print("Predicted risk:", model.predict(x)[0])
for c, p in zip(model.classes_, model.predict_proba(x)[0]):
    print(f"  {c}: {p:.2f}")
