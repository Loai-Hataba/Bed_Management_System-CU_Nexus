# bed_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# 1. Load the synthetic data
df = pd.read_csv("synthetic_bedlogs.csv")

# ===============================
# Feature Engineering
# ===============================
df['log_time'] = pd.to_datetime(df['log_time'])
df['hour'] = df['log_time'].dt.hour
df['dayofweek'] = df['log_time'].dt.dayofweek

# Encode patient_id (categorical -> integer)
df['patient_id_enc'] = df['patient_id'].astype('category').cat.codes

# Shift On_bed to create prediction target (next hour)
df = df.sort_values(['Bed_id', 'log_time'])
df['On_bed_next'] = df.groupby('Bed_id')['On_bed'].shift(-1)

# Drop rows where target is NaN (end of sequence)
df = df.dropna(subset=['On_bed_next'])
df['On_bed_next'] = df['On_bed_next'].astype(int)

# ===============================
# Supervised Model → Bed Vacancy Prediction
# ===============================
features = ['Bed_id', 'hour', 'dayofweek', 'patient_id_enc', 'On_bed']
X = df[features]
y = df['On_bed_next']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print("=== Vacancy Prediction Model ===")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

joblib.dump(rf, "bed_vacancy_predictor.pkl")

# ===============================
# Unsupervised Model → Outlier Detection
# ===============================
iso_features = ['Bed_id', 'hour', 'dayofweek', 'patient_id_enc', 'On_bed']
iso = IsolationForest(contamination=0.02, random_state=42)
df['anomaly'] = iso.fit_predict(df[iso_features])

print("=== Outlier Counts ===")
print(df['anomaly'].value_counts())

joblib.dump(iso, "bed_anomaly_detector.pkl")

print("✅ Models trained and saved")
