import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# ==============================
# LOAD DATASET
# ==============================

df = pd.read_csv("cybersecurity_intrusion_data.csv")

print("✅ Dataset Loaded")
print(df.head())

# ==============================
# CLEANING
# ==============================

# Remove unwanted column
df = df.drop(columns=["session_id"])

# Remove duplicates
df = df.drop_duplicates()

# Fill missing values
df = df.fillna(0)

# Remove extra spaces in column names
df.columns = df.columns.str.strip()

print("\n✅ Dataset Cleaned")

# ==============================
# ENCODING
# ==============================

categorical_cols = [
    "protocol_type",
    "encryption_used",
    "browser_type"
]

encoders = {}

for col in categorical_cols:

    # Convert entire column to string
    df[col] = df[col].astype(str)

    le = LabelEncoder()

    df[col] = le.fit_transform(df[col])

    encoders[col] = le

print("\n✅ Encoding Completed")
# ==============================
# SPLIT FEATURES & TARGET
# ==============================

X = df.drop("attack_detected", axis=1)
y = df["attack_detected"]

# ==============================
# FEATURE SCALING
# ==============================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# Save scaler
joblib.dump(scaler, "scaler.pkl")

# ==============================
# SAVE CLEANED DATASET
# ==============================

processed_df = pd.DataFrame(X_scaled, columns=X.columns)

processed_df["attack_detected"] = y.values

processed_df.to_csv("processed_intrusion_data.csv", index=False)

print("\n✅ Preprocessing Completed")
print("✅ Processed dataset saved as processed_intrusion_data.csv")

print("\nFinal Shape:", processed_df.shape)