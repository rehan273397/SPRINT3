

import sys
import re
import json
import joblib
import torch
import pandas as pd
import streamlit as st

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.lstm_model import PhishGuardLSTM


# =====================================================
# CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="PHISHGUARD | Threat Intelligence",
    page_icon="🛡️",
    layout="wide"
)

MODEL_DIR = ROOT / "model_artifacts"
REPORT_DIR = ROOT / "reports"

DEVICE = torch.device("cpu")
MAX_LENGTH = 128

torch.set_num_threads(2)

MODEL_NAMES = [
    "Baseline",
    "LSTM",
    "DistilBERT"
]

SAMPLES = {
    "Legitimate example": (
        "Hello team, our weekly project meeting has been moved "
        "to Thursday at 10 AM. Please review the attached agenda "
        "before the meeting. Regards, Project Management."
    ),
    "Phishing example": (
        "URGENT: Your bank account will be suspended today. "
        "Verify your password and account details immediately "
        "using http://secure-login.example.test to avoid losing access."
    )
}


# =====================================================
# CUSTOM DESIGN
# =====================================================

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(ellipse at 50% 0%, #14243c 0%, #080d18 55%);
    color: #e6edf7;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

[data-testid="stSidebar"] {
    background: #0b1220;
    border-right: 1px solid #26354b;
}

.ph-hero {
    background: linear-gradient(120deg, #111d31, #142b42);
    border: 1px solid #29415d;
    border-radius: 18px;
    padding: 30px;
    margin-bottom: 25px;
}

.ph-eyebrow {
    color: #39d9bd;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 3px;
}

.ph-heading {
    font-size: 38px;
    font-weight: 850;
    color: #f3f7ff;
    margin: 8px 0;
}

.ph-subtitle {
    color: #aabbd0;
    font-size: 15px;
}

.ph-card {
    background: #111c2c;
    border: 1px solid #293b52;
    border-radius: 15px;
    padding: 20px;
    margin: 8px 0;
}

.ph-small {
    color: #91a6bd;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
}

.ph-value {
    color: #edf5ff;
    font-size: 25px;
    font-weight: 800;
    margin-top: 8px;
}

.ph-muted {
    color: #91a6bd;
    font-size: 13px;
}

.ph-danger {
    background: #351923;
    border: 1px solid #a63c55;
    border-radius: 14px;
    padding: 20px;
}

.ph-safe {
    background: #102f2b;
    border: 1px solid #287c68;
    border-radius: 14px;
    padding: 20px;
}

.ph-warning {
    color: #f5c56b;
}

.ph-section {
    color: #edf5ff;
    font-weight: 800;
    font-size: 20px;
    margin-top: 12px;
}

div.stButton > button {
    border-radius: 10px;
    min-height: 45px;
    font-weight: 750;
    border: 1px solid #2a8e99;
}

div.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #087e91, #12a890);
    color: white;
    border: none;
}

div[data-testid="stMetric"] {
    background: #111c2c;
    border: 1px solid #293b52;
    padding: 16px;
    border-radius: 12px;
}

div[data-testid="stMetricLabel"] {
    color: #9db1c8;
}

hr {
    border-color: #293b52;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# MODEL LOADING
# =====================================================

@st.cache_resource
def load_baseline():
    vectorizer = joblib.load(
        MODEL_DIR / "tfidf_vectorizer.joblib"
    )

    model = joblib.load(
        MODEL_DIR / "logistic_regression.joblib"
    )

    return vectorizer, model


@st.cache_resource
def load_lstm():
    tokenizer = joblib.load(
        MODEL_DIR / "lstm_tokenizer.joblib"
    )

    model = PhishGuardLSTM(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=64,
        hidden_dim=64,
        num_layers=1,
        dropout=0.3
    )

    model.load_state_dict(
        torch.load(
            MODEL_DIR / "lstm_model.pt",
            map_location=DEVICE,
            weights_only=True
        )
    )

    model.eval()

    return tokenizer, model


@st.cache_resource
def load_distilbert():

    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification
    )

    model_path = MODEL_DIR / "distilbert_phishguard"

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path
    )

    model.to(DEVICE)
    model.eval()

    return tokenizer, model


# =====================================================
# PREDICTION ENGINE
# =====================================================

def predict_baseline(text):

    vectorizer, model = load_baseline()

    features = vectorizer.transform([text])
    probabilities = model.predict_proba(features)[0]

    phishing_index = list(model.classes_).index(1)

    return float(probabilities[phishing_index])


def predict_lstm(text):

    tokenizer, model = load_lstm()

    sequence = tokenizer.encode(
        text,
        max_length=MAX_LENGTH
    )

    inputs = torch.tensor(
        [sequence],
        dtype=torch.long
    )

    with torch.no_grad():
        output = model(inputs)
        probability = torch.sigmoid(output).item()

    return float(probability)


def predict_distilbert(text):

    tokenizer, model = load_distilbert()

    encoded = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**encoded)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )

    return float(probabilities[0][1].item())


def predict(text, model_name):

    if model_name == "Baseline":
        probability = predict_baseline(text)

    elif model_name == "LSTM":
        probability = predict_lstm(text)

    elif model_name == "DistilBERT":
        probability = predict_distilbert(text)

    else:
        raise ValueError("Unknown model.")

    return {
        "model": model_name,
        "phishing_probability": probability,
        "prediction": (
            "Phishing" if probability >= 0.5
            else "Legitimate"
        )
    }


# =====================================================
# SIMPLE TEXT INDICATORS
# =====================================================

def inspect_text(text):

    indicators = []

    lowered = text.lower()

    if re.search(r"https?://|www\.", lowered):
        indicators.append(
            ("Link detected", "The message contains a web address.")
        )

    urgency_words = [
        "urgent", "immediately", "act now",
        "suspended", "expires today", "limited time"
    ]

    if any(word in lowered for word in urgency_words):
        indicators.append(
            ("Urgency language", "The message uses urgency-related wording.")
        )

    credentials = [
        "password", "otp", "verification code",
        "pin", "login details", "account details"
    ]

    if any(word in lowered for word in credentials):
        indicators.append(
            (
                "Credential-related wording",
                "The message mentions passwords, codes or account details."
            )
        )

    financial_words = [
        "bank", "payment", "credit card",
        "refund", "wire transfer", "billing"
    ]

    if any(word in lowered for word in financial_words):
        indicators.append(
            (
                "Financial wording",
                "The message refers to financial activity."
            )
        )

    return indicators


# =====================================================
# EVALUATION DATA
# =====================================================

def load_evaluation():

    path = REPORT_DIR / "final_test_results.json"

    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


# =====================================================
# HEADER
# =====================================================

st.markdown("""
<div class="ph-hero">
    <div class="ph-eyebrow">
        CYBERSECURITY • MACHINE LEARNING • NLP
    </div>
    <div class="ph-heading">🛡️ PHISHGUARD</div>
    <div class="ph-subtitle">
        Intelligent phishing message detection and analysis
    </div>
</div>
""", unsafe_allow_html=True)


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.markdown("## 🛡️ PHISHGUARD")

    st.caption("THREAT ANALYSIS CONSOLE")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Message Scanner",
            "Model Lab",
            "How It Works"
        ]
    )

    st.divider()

    st.markdown("### SYSTEM")

    st.success("Models loaded on demand")

    st.caption("Inference device: CPU")
    st.caption("Classification: Binary")
    st.caption("0 = Legitimate")
    st.caption("1 = Phishing")

    st.divider()

    st.caption("PHISHGUARD • Academic project")


# =====================================================
# PAGE 1: SCANNER
# =====================================================

if page == "Message Scanner":

    st.markdown(
        '<div class="ph-section">Message Inspection Center</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Submit a suspicious email or message to inspect it "
        "with your trained detection models."
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        mode = st.radio(
            "Detection mode",
            [
                "Compare all three models",
                "Use one model"
            ],
            horizontal=True
        )

    with col2:
        if mode == "Use one model":

            selected_model = st.selectbox(
                "Select model",
                MODEL_NAMES
            )

        else:
            st.info(
                "All three trained models will analyze the same message."
            )
            selected_model = None

    st.divider()

    st.markdown("### 01 / Input Message")

    st.caption(
        "Paste the complete message. You can also load a demonstration example."
    )

    def apply_example():
        choice = st.session_state.get("sample_choice")

        if choice in SAMPLES:
            st.session_state["message_input"] = SAMPLES[choice]

    st.session_state.setdefault("message_input", "")

    st.selectbox(
        "Load a sample message",
        ["Choose an example"] + list(SAMPLES.keys()),
        key="sample_choice",
        on_change=apply_example
    )

    message = st.text_area(
        "Message content",
        height=210,
        placeholder=(
            "Paste the email, SMS or suspicious message here..."
        ),
        key="message_input",
        label_visibility="collapsed"
    )

    st.caption(
        f"Characters entered: {len(message)}"
    )

    scan_button = st.button(
        "⚡ ANALYZE MESSAGE",
        type="primary",
        use_container_width=True
    )

    if scan_button:

        if not message.strip():

            st.warning("Enter a message before starting analysis.")

        else:

            results = []

            models_to_run = (
                MODEL_NAMES
                if mode == "Compare all three models"
                else [selected_model]
            )

            try:

                with st.spinner(
                    "Analyzing message with trained models..."
                ):

                    for model_name in models_to_run:
                        results.append(
                            predict(message.strip(), model_name)
                        )

                st.session_state["last_scan"] = {
                    "message": message.strip(),
                    "results": results
                }

            except Exception as error:

                st.error(f"Analysis failed: {error}")
                st.exception(error)

    # Display previous or newest scan
    scan = st.session_state.get("last_scan")

    if scan:

        st.divider()

        st.markdown("## 02 / Detection Results")

        results = scan["results"]

        phishing_count = sum(
            result["prediction"] == "Phishing"
            for result in results
        )

        legitimate_count = len(results) - phishing_count

        if len(results) == 1:

            result = results[0]

            if result["prediction"] == "Phishing":

                st.markdown("""
                <div class="ph-danger">
                    <div class="ph-small">MODEL CLASSIFICATION</div>
                    <div class="ph-value">⚠️ PHISHING DETECTED</div>
                    <div class="ph-muted">
                        The selected model classified this message as phishing.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            else:

                st.markdown("""
                <div class="ph-safe">
                    <div class="ph-small">MODEL CLASSIFICATION</div>
                    <div class="ph-value">✓ LEGITIMATE PREDICTION</div>
                    <div class="ph-muted">
                        The selected model classified this message as legitimate.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.write("")

            st.metric(
                "Phishing probability",
                f"{result['phishing_probability']:.2%}"
            )

            st.progress(result["phishing_probability"])

            st.caption(
                "This is the model's estimated phishing probability, "
                "not a guarantee of safety or a calibrated risk score."
            )

        else:

            st.markdown("### Model consensus")

            c1, c2, c3 = st.columns(3)

            c1.metric("Models tested", len(results))
            c2.metric("Phishing predictions", phishing_count)
            c3.metric("Legitimate predictions", legitimate_count)

            predictions = [
                result["prediction"] for result in results
            ]

            if len(set(predictions)) == 1:

                st.success(
                    f"All {len(results)} models agree: "
                    f"{predictions[0]}."
                )

            else:

                st.warning(
                    "The models disagree. Review the individual "
                    "predictions before drawing a conclusion."
                )

            comparison_rows = []

            for result in results:

                comparison_rows.append({
                    "Model": result["model"],
                    "Prediction": result["prediction"],
                    "Phishing probability": (
                        f"{result['phishing_probability']:.2%}"
                    )
                })

            st.dataframe(
                comparison_rows,
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "Model probabilities are individual outputs, "
                "not a combined ensemble prediction."
            )

        st.divider()

        st.markdown("### 03 / Textual Indicators")

        indicators = inspect_text(scan["message"])

        st.caption(
            "These are simple rule-based text checks, separate from "
            "the machine-learning model's decision."
        )

        if indicators:

            for name, description in indicators:

                st.markdown(
                    f"**⚑ {name}**  \n{description}"
                )

        else:

            st.info(
                "No indicators from the basic rule-based checks were found. "
                "This does not mean the message is safe."
            )

        st.divider()

        st.markdown("### Recommended precautions")

        st.write(
            "Verify the sender through an independently obtained contact "
            "method. Do not share passwords, OTPs or banking details. "
            "Avoid opening unexpected links or attachments."
        )

        if st.button("Clear previous results"):

            st.session_state.pop("last_scan", None)
            st.rerun()


# =====================================================
# PAGE 2: MODEL LAB
# =====================================================

elif page == "Model Lab":

    st.markdown(
        '<div class="ph-section">Model Performance Laboratory</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Compare the performance of the three trained classifiers "
        "using the held-out test dataset."
    )

    results = load_evaluation()

    if results is None:

        st.warning(
            "The final test results file was not found. "
            "Run the final evaluation script first."
        )

    else:

        rows = []

        for name, metrics in results.items():

            rows.append({
                "Model": name,
                "Accuracy": metrics["accuracy"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1 Score": metrics["f1"]
            })

        df = pd.DataFrame(rows).set_index("Model")

        best_model = df["F1 Score"].idxmax()
        best_f1 = df["F1 Score"].max()

        st.success(
            f"Highest test F1 score: {best_model} "
            f"({best_f1:.2%})"
        )

        st.markdown("### Performance comparison")

        st.dataframe(
            df.style.format("{:.2%}"),
            use_container_width=True
        )

        st.markdown("### Performance charts")

        metric = st.selectbox(
            "Select metric",
            [
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score"
            ]
        )

        st.bar_chart(df[[metric]])

        st.divider()

        st.markdown("### Confusion matrices")

        for model_name, metrics in results.items():

            with st.expander(f"{model_name} — Confusion Matrix"):

                matrix = pd.DataFrame(
                    metrics["confusion_matrix"],
                    index=["Actual Legitimate", "Actual Phishing"],
                    columns=["Predicted Legitimate", "Predicted Phishing"]
                )

                st.dataframe(
                    matrix,
                    use_container_width=True
                )

                st.caption(
                    "Rows represent actual labels; columns represent "
                    "model predictions."
                )

        st.info(
            "Accuracy measures overall correctness. Precision measures "
            "how often phishing predictions are correct. Recall measures "
            "how many actual phishing messages are detected. F1 balances "
            "precision and recall."
        )


# =====================================================
# PAGE 3: HOW IT WORKS
# =====================================================

elif page == "How It Works":

    st.markdown(
        '<div class="ph-section">Understanding PHISHGUARD</div>',
        unsafe_allow_html=True
    )

    st.write(
        "PHISHGUARD is a binary text-classification project that "
        "compares traditional machine learning with deep learning "
        "and Transformer-based NLP."
    )

    st.markdown("### How to use the scanner")

    st.markdown("""
    **01 — Provide text**

    Paste an email, SMS, or other message into the scanner.

    **02 — Select the analysis mode**

    Choose one model or compare all three models on the same message.

    **03 — Run the analysis**

    Press the Analyze Message button.

    **04 — Interpret the prediction**

    A phishing probability of 0.80 means that the model assigned an
    80% phishing probability to the input. It is not proof that the
    message is malicious.

    **05 — Verify independently**

    Check the sender, links and claims before taking action.
    """)

    st.divider()

    st.markdown("### The three models")

    st.markdown("""
    **1. TF-IDF + Logistic Regression**

    Converts text into weighted word and phrase features, then uses
    Logistic Regression for classification. This is the baseline model.

    **2. LSTM**

    Uses word embeddings and a Long Short-Term Memory neural network
    to learn patterns across sequences of words.

    **3. DistilBERT**

    Uses a pretrained Transformer model and fine-tuning to classify
    messages based on contextual language representations.
    """)

    st.divider()

    st.markdown("### Dataset and labels")

    st.write("""
    PHISHGUARD uses the DIFrauD phishing dataset from Hugging Face.

    Its classification labels are:

    - **0 — Legitimate**
    - **1 — Phishing**

    The original dataset was split into training, validation and test
    sets. The test set is used for final model evaluation.
    """)

    st.divider()

    st.markdown("### Important limitations")

    st.warning("""
    PHISHGUARD is an academic detection prototype, not a complete
    email security gateway.

    It analyzes the text supplied to it. It does not verify the sender,
    inspect actual website destinations, analyze attachments, or
    establish whether a link is safe.

    A legitimate prediction does not guarantee that a message is safe.
    A phishing prediction should be independently investigated.
    """)


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#8297ae; font-size:12px;">
        PHISHGUARD • INTELLIGENT PHISHING MESSAGE DETECTION<br>
        Machine Learning | NLP | Deep Learning | Transformers
    </div>
    """,
    unsafe_allow_html=True
)
