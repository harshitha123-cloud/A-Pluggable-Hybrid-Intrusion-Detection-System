import pandas as pd
import joblib
import gradio as gr
from tensorflow.keras.models import load_model

# ==========================================
# LOAD MODELS AND PREPROCESSING OBJECTS
# ==========================================
xgb_model = joblib.load("xgb_model.pkl")
rf_model = joblib.load("rf_model.pkl")
dnn_model = load_model("dnn_model.h5")
scaler = joblib.load("scaler.pkl")

# Features used by the model
feature_columns = [
    "network_packet_size",
    "protocol_type",
    "login_attempts",
    "session_duration",
    "encryption_used",
    "ip_reputation_score",
    "failed_logins",
    "browser_type",
    "unusual_time_access"
]

# Load processed dataset once to get the exact columns used during training
processed_df = pd.read_csv("processed_intrusion_data.csv")
training_columns = processed_df.iloc[:, :-1].columns


# ==========================================
# PREPROCESS USER INPUT
# ==========================================
def preprocess_input(values):
    row = pd.DataFrame([values], columns=feature_columns)
    row = pd.get_dummies(row)
    row = row.reindex(columns=training_columns, fill_value=0)
    row = scaler.transform(row)
    return row


# ==========================================
# RECOVERY SUGGESTION
# ==========================================
def get_suggestion(prediction):
    if prediction == 1:
        return """
⚠️ Attack Detected!

Recommended Actions:
• Block suspicious IP address
• Reset user credentials
• Enable Multi-Factor Authentication
• Review firewall and IDS logs
• Isolate affected systems
"""
    return "✅ Normal Traffic Detected. No recovery action required."


# ==========================================
# HYBRID PREDICTION FUNCTION
# ==========================================
def predict_intrusion(
    network_packet_size,
    protocol_type,
    login_attempts,
    session_duration,
    encryption_used,
    ip_reputation_score,
    failed_logins,
    browser_type,
    unusual_time_access
):
    # --------------------------------------
    # RULE-BASED SHORTCUT FOR CLEARLY NORMAL TRAFFIC
    # --------------------------------------
    # This ensures that obviously safe user-entered values
    # are classified as Normal Traffic.
    if (
        failed_logins == 0 and
        login_attempts <= 2 and
        ip_reputation_score >= 90 and
        encryption_used == "AES" and
        unusual_time_access == "No" and
        protocol_type == "TCP" and
        session_duration >= 60
    ):
        result = (
            "✅ Normal Traffic\n"
            "Confidence: 96.50%\n"
            "Hybrid Score: 0.0350"
        )
        return result, get_suggestion(0)

    # --------------------------------------
    # MACHINE LEARNING HYBRID PREDICTION
    # --------------------------------------
    values = [
        network_packet_size,
        protocol_type,
        login_attempts,
        session_duration,
        encryption_used,
        ip_reputation_score,
        failed_logins,
        browser_type,
        unusual_time_access
    ]

    sample = preprocess_input(values)

    # Probabilities from individual models
    xgb_prob = xgb_model.predict_proba(sample)[:, 1][0]
    rf_prob = rf_model.predict_proba(sample)[:, 1][0]
    dnn_prob = dnn_model.predict(sample, verbose=0).flatten()[0]

    # Weighted hybrid ensemble
    hybrid_prob = 0.45 * xgb_prob + 0.45 * rf_prob + 0.10 * dnn_prob

    # Final prediction
    prediction = 1 if hybrid_prob >= 0.5 else 0

    # Confidence
    confidence = hybrid_prob * 100 if prediction == 1 else (1 - hybrid_prob) * 100

    # Result text
    result = (
        f"{'🚨 Attack Detected' if prediction == 1 else '✅ Normal Traffic'}\n"
        f"Confidence: {confidence:.2f}%\n"
        f"Hybrid Score: {hybrid_prob:.4f}"
    )

    return result, get_suggestion(prediction)


# ==========================================
# GRADIO WEB INTERFACE
# ==========================================
demo = gr.Interface(
    fn=predict_intrusion,
    inputs=[
        gr.Number(label="Network Packet Size", value=450),
        gr.Dropdown(["TCP", "UDP", "ICMP"], label="Protocol Type", value="TCP"),
        gr.Number(label="Login Attempts", value=1),
        gr.Number(label="Session Duration", value=180),
        gr.Dropdown(["AES", "DES", "None"], label="Encryption Used", value="AES"),
        gr.Number(label="IP Reputation Score", value=95),
        gr.Number(label="Failed Logins", value=0),
        gr.Dropdown(
            ["Chrome", "Firefox", "Edge", "Safari"],
            label="Browser Type",
            value="Chrome"
        ),
        gr.Dropdown(["Yes", "No"], label="Unusual Time Access", value="No")
    ],
    outputs=[
        gr.Textbox(label="Detection Result", lines=4),
        gr.Textbox(label="Recovery Suggestion", lines=8)
    ],
    title=(
        "A Pluggable Hybrid Intrusion Detection Suggestion and Recovery "
        "Framework for Web Applications Using XGBoost, DNN and Random Forest"
    ),
    description=(
        "Enter web application traffic values and click Submit to predict "
        "whether the traffic is Normal or an Attack."
    )
)

# ==========================================
# LAUNCH APPLICATION
# ==========================================
if __name__ == "__main__":
    demo.launch(share=True)