# GRAIN EYE: ML Risk Classifier

Random Forest classifier that predicts grain storage spoilage risk (Low / Moderate / High) from DHT22 + MQ-135 sensor readings.

## Features
crop type, temperature (C), humidity (%), MQ-135 raw gas reading, 6-hour gas trend

## Run
    pip install numpy pandas scikit-learn matplotlib joblib
    python3 generate_data.py     # creates data/grain_synthetic.csv
    python3 train.py             # trains, evaluates, saves model + plots
    python3 predict.py wheat 36 85 2600 300   # crop temp humidity gas gas_rise

## Results (synthetic data, 3000 rows, 80/20 stratified split)
See results/metrics.txt, results/confusion_matrix.png, results/feature_importance.png

## Important
The dataset is synthetic. Risk labels come from a rule-based formula with added noise, and the crop humidity limits are placeholder assumptions. These results validate the ML pipeline, not real spoilage prediction. Next step: retrain on real DHT22/MQ-135 logs with lab-verified labels (`python3 train.py data/real.csv`).
