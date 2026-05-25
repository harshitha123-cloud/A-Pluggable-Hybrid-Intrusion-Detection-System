import pandas as pd
import joblib
import gradio as gr
from tensorflow.keras.models import load_model

# Load models
xgb_model = joblib.load("xgb_model.pkl")
rf_model = joblib.load("rf_model.pkl")
dnn_model = load_model("dnn_model.h5")
scaler = joblib.load("scaler.pkl")

# Feature columns
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

# Training columns
processed_df = pd.read_csv("processed_intrusion_data.csv")
training_columns = processed_df.iloc[:, :-1].columns


def preprocess_input(values):
    row = pd.DataFrame([values], columns=feature_columns)
    row = pd.get_dummies(row)
    row = row.reindex(columns=training_columns, fill_value=0)
    row = scaler.transform(row)
    return row


def get_suggestion(pred):
    if pred == 1:
        return """
⚠️ Attack Detected!

Recommended Actions:
• Block suspicious IP address
• Reset user credentials
• Enable Multi-Factor Authentication
• Review IDS / Firewall logs
• Isolate affected systems
"""
    return "✅ Normal Traffic Detected. No recovery action required."


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

    # Model probabilities
    xgb_prob = xgb_model.predict_proba(sample)[0][1]
    rf_prob = rf_model.predict_proba(sample)[0][1]
    dnn_prob = float(dnn_model.predict(sample, verbose=0)[0][0])

    # Hybrid Ensemble
    hybrid_prob = (
        0.45 * xgb_prob +
        0.45 * rf_prob +
        0.10 * dnn_prob
    )

    prediction = 1 if hybrid_prob >= 0.5 else 0

    confidence = (
        hybrid_prob * 100
        if prediction == 1
        else (1 - hybrid_prob) * 100
    )

    result = (
        f"{'🚨 Attack Detected' if prediction else '✅ Normal Traffic'}\n"
        f"Confidence: {confidence:.2f}%\n"
        f"Hybrid Score: {hybrid_prob:.4f}"
    )

    return result, get_suggestion(prediction)


demo = gr.Interface(
    fn=predict_intrusion,

    inputs=[
        gr.Number(label="Network Packet Size"),

        gr.Dropdown(
            ["TCP", "UDP", "ICMP"],
            label="Protocol Type",
             ),

        gr.Number(label="Login Attempts"),

        gr.Number(label="Session Duration"),

        gr.Dropdown(
            ["AES", "DES"],
            label="Encryption Used",
        ),

        gr.Number(
            label="IP Reputation Score (0 - 1)",
        ),

        gr.Number(label="Failed Logins"),

        gr.Dropdown(
            ["Chrome", "Firefox", "Edge", "Safari", "Unknown"],
            label="Browser Type",
            ),

        gr.Dropdown(
            [0, 1],
            label="Unusual Time Access",
        )
    ],

    outputs=[
        gr.Textbox(label="Detection Result", lines=4),
        gr.Textbox(label="Recovery Suggestion", lines=8)
    ],

    title="A Pluggable Hybrid Intrusion Detection Suggestion and Recovery Framework for Web Applications",

    description="Enter network traffic values and click Submit to predict whether the traffic is Normal or an Attack."
)

if __name__ == "__main__":
    demo.launch(share=True)