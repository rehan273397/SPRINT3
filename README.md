# PHISHGUARD — Intelligent Phishing Message Detection System

## Project Overview

PHISHGUARD is a machine-learning-based application designed to identify potentially fraudulent phishing messages. It analyzes the text of emails and messages and classifies them as either **Legitimate** or **Phishing**.

The project explores and compares three machine-learning and deep-learning approaches:

- **Logistic Regression with TF-IDF:** Identifies suspicious patterns and keywords in messages.
- **LSTM (Long Short-Term Memory):** Learns sequential patterns and relationships between words.
- **DistilBERT:** Uses contextual language understanding through a pretrained Transformer model.

The models are trained and evaluated using the DIFrauD phishing dataset from Hugging Face.

PHISHGUARD includes an interactive Streamlit dashboard where users can submit messages for analysis, compare model predictions, view phishing probabilities, and examine model performance through evaluation metrics and confusion matrices.

**Project Goal:** Demonstrate how Natural Language Processing, Deep Learning, and Transformer architectures can be applied to phishing message detection.

**Project Type:** Academic Sprint 03 — Deep Learning, NLP & Transformers.

**Note:** PHISHGUARD is an educational prototype. It analyzes message text and does not independently verify sender identities, actual URL destinations, or attachments.
