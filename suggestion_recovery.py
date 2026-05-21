import pandas as pd
import joblib

from tensorflow.keras.models import load_model

# ==============================
# LOAD MODELS
# ==============================

xgb_model = joblib.load("xgb_model.pkl")

dnn_model = load_model("dnn_model.h5")

print("✅ Models Loaded")

# ==============================
# LOAD DATASET
# ==============================

df = pd.read_csv("processed_intrusion_data.csv")

# Take one sample
sample = df.drop("attack_detected", axis=1).iloc[0:1]

# ==============================
# PREDICTIONS
# ==============================

xgb_pred = xgb_model.predict(sample)

dnn_pred = (dnn_model.predict(sample) > 0.5).astype(int)

# ==============================
# HYBRID DECISION
# ==============================

final_prediction = 1 if (
    xgb_pred[0] == 1 and dnn_pred[0][0] == 1
) else 0

# ==============================
# OUTPUT
# ==============================

print("\n🔷 XGBoost Prediction:", xgb_pred[0])

print("🔷 DNN Prediction:", dnn_pred[0][0])

print("\n🔥 FINAL HYBRID PREDICTION:", final_prediction)

# ==============================
# SUGGESTION & RECOVERY
# ==============================

if final_prediction == 1:

    print("\n🚨 Intrusion Detected")

    print("\n📌 Security Suggestions:")

    print("✔ Change account password")

    print("✔ Enable multi-factor authentication")

    print("✔ Monitor suspicious traffic")

    print("✔ Scan system for malware")

    print("\n🔄 Recovery Actions:")

    print("✔ Blocking suspicious IP address")

    print("✔ Terminating malicious session")

    print("✔ Restoring secure connection")

else:

    print("\n✅ Normal Traffic Detected")

    print("✔ No recovery needed")