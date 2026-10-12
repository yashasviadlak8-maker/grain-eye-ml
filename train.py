import sys, joblib, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

path = sys.argv[1] if len(sys.argv) > 1 else "data/grain_synthetic.csv"
df = pd.read_csv(path)
df["crop_code"] = df["crop"].map({"wheat": 0, "rice": 1, "maize": 2})
FEATURES = ["crop_code", "temperature_C", "humidity_pct", "gas_raw", "gas_rise_6h"]
X, y = df[FEATURES], df["risk"]

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

rf = RandomForestClassifier(n_estimators=200, random_state=42).fit(Xtr, ytr)
pred = rf.predict(Xte)

labels = ["Low", "Moderate", "High"]
report = classification_report(yte, pred, labels=labels)
cv_rf = cross_val_score(RandomForestClassifier(n_estimators=200, random_state=42), X, y, cv=5).mean()
cv_dt = cross_val_score(DecisionTreeClassifier(random_state=42), X, y, cv=5).mean()

text = (f"Data file: {path}\nRows: {len(df)}\nTrain/Test: {len(Xtr)}/{len(Xte)}\n\n{report}\n"
        f"5-fold CV accuracy  Random Forest: {cv_rf:.3f}\n5-fold CV accuracy  Decision Tree: {cv_dt:.3f}\n")
print(text)
open("results/metrics.txt", "w").write(text)

ConfusionMatrixDisplay(confusion_matrix(yte, pred, labels=labels), display_labels=labels).plot(cmap="Blues")
plt.title("Confusion matrix (rows = actual, cols = predicted)")
plt.savefig("results/confusion_matrix.png", dpi=150, bbox_inches="tight"); plt.close()

pd.Series(rf.feature_importances_, index=FEATURES).sort_values().plot.barh(title="Feature importance")
plt.savefig("results/feature_importance.png", dpi=150, bbox_inches="tight"); plt.close()

joblib.dump(rf, "model/grain_rf.pkl")
print("Saved model and plots.")
