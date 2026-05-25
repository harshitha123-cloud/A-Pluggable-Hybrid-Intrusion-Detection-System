import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
df = pd.read_csv("cybersecurity_intrusion_data.csv")

# Cleaning
df.drop(columns=["session_id"], inplace=True)
df.drop_duplicates(inplace=True)
df.fillna(0, inplace=True)

# Encode categorical columns
encoders = {}

for col in ["protocol_type", "encryption_used", "browser_type"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# Save encoders
joblib.dump(encoders, "encoders.pkl")

# Features and target
X = df.drop("attack_detected", axis=1)
y = df["attack_detected"]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler
joblib.dump(scaler, "scaler.pkl")

# Save processed dataset
processed_df = pd.DataFrame(X_scaled, columns=X.columns)
processed_df["attack_detected"] = y

processed_df.to_csv("processed_intrusion_data.csv", index=False)

print("Preprocessing Completed")