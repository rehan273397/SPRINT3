
from datetime import datetime
from io import StringIO
import html

import joblib
import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# GHOSTTRACE - beginner-friendly security evidence analyzer
# ---------------------------------------------------------

st.set_page_config(
    page_title="GHOSTTRACE | Security Analyzer",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "models/baseline_tfidf_logreg_clean.joblib"
MAX_ROWS = 5000
APP_VERSION = "2.0"


# ---------------------------------------------------------
# Security categories and beginner-friendly explanations
# ---------------------------------------------------------

SECURITY_INFO = {
    "A01_broken_access_control": {
        "name": "Broken Access Control",
        "short": "Someone may be able to access information or actions they should not.",
        "details": (
            "An application may fail to enforce what a user is allowed to see or do. "
            "For example, changing an account ID might expose another user's records."
        ),
        "checks": [
            "Check whether the server verifies permissions for every sensitive request.",
            "In an authorized test environment, check whether users can access only their own records.",
            "Review role and tenant restrictions on server-side operations.",
        ],
    },
    "A02_cryptographic_failures": {
        "name": "Cryptographic Failures",
        "short": "Sensitive information may not be adequately protected.",
        "details": (
            "Sensitive data could be exposed because of weak encryption, unsafe storage, "
            "poor key management, or transmission without suitable protection."
        ),
        "checks": [
            "Review how sensitive data is encrypted in transit and at rest.",
            "Check that secrets and cryptographic keys are not exposed in source code.",
            "Review cryptographic algorithms and key-management practices.",
        ],
    },
    "A03_injection": {
        "name": "Injection",
        "short": "Untrusted input may be interpreted as a command or query.",
        "details": (
            "Injection risks can occur when an application combines untrusted input "
            "with database queries, operating-system commands, or other interpreters."
        ),
        "checks": [
            "Review database access for parameterized queries.",
            "Check how untrusted input is passed to interpreters and commands.",
            "Review input handling and context-appropriate output encoding.",
        ],
    },
    "A04_insecure_design": {
        "name": "Insecure Design",
        "short": "A security control may be missing from the application's design.",
        "details": (
            "The application may lack appropriate safeguards, abuse-case analysis, "
            "or limits for sensitive workflows."
        ),
        "checks": [
            "Review the security requirements for the affected feature.",
            "Consider abuse cases, rate limits, and failure scenarios.",
            "Check that important controls exist in the design, not just the interface.",
        ],
    },
    "A05_security_misconfiguration": {
        "name": "Security Misconfiguration",
        "short": "A security setting may be missing, unsafe, or incorrectly configured.",
        "details": (
            "Examples can include unnecessarily exposed services, unsafe default settings, "
            "excessive permissions, or overly detailed error messages."
        ),
        "checks": [
            "Review application and server configuration against a secure baseline.",
            "Check default accounts, permissions, and unnecessary services.",
            "Review security headers, error handling, and environment settings.",
        ],
    },
    "A06_vulnerable_and_outdated_components": {
        "name": "Vulnerable and Outdated Components",
        "short": "A software dependency may be outdated or contain a known vulnerability.",
        "details": (
            "Applications depend on libraries, frameworks, and other components. "
            "An affected version may introduce a known security risk."
        ),
        "checks": [
            "Identify the exact dependency and version.",
            "Check trusted vendor advisories and relevant CVE records.",
            "Confirm whether the vulnerable component and affected functionality are actually present.",
        ],
    },
    "A07_identification_and_authentication_failures": {
        "name": "Identification and Authentication Failures",
        "short": "Login, identity verification, or session handling may be weak.",
        "details": (
            "Possible issues include weak authentication controls, unsafe session handling, "
            "or account-recovery workflows that do not adequately verify users."
        ),
        "checks": [
            "Review login, account recovery, and multi-factor authentication controls.",
            "Check session expiration, logout, and session renewal behaviour.",
            "Review protections against automated login attempts.",
        ],
    },
    "A08_software_and_data_integrity_failures": {
        "name": "Software and Data Integrity Failures",
        "short": "The application may not adequately verify software or data integrity.",
        "details": (
            "This category can involve unsafe updates, untrusted software components, "
            "or data that is accepted without suitable integrity verification."
        ),
        "checks": [
            "Review how updates and software packages are obtained and verified.",
            "Check integrity controls for sensitive data and build artifacts.",
            "Review trust decisions involving third-party code or serialized data.",
        ],
    },
    "A09_security_logging_and_monitoring_failures": {
        "name": "Security Logging and Monitoring Failures",
        "short": "Important security events may not be recorded or investigated effectively.",
        "details": (
            "Insufficient logging or alerting can make attacks difficult to detect, "
            "investigate, or respond to."
        ),
        "checks": [
            "Review logging for authentication failures and sensitive actions.",
            "Check whether alerts exist for relevant suspicious activity.",
            "Verify that logs are protected and do not expose secrets.",
        ],
    },
    "A10_server_side_request_forgery": {
        "name": "Server-Side Request Forgery (SSRF)",
        "short": "A server may be tricked into making requests to an unintended destination.",
        "details": (
            "An application that fetches URLs or other remote resources may be at risk "
            "if it does not sufficiently restrict destinations and network access."
        ),
        "checks": [
            "Review features that fetch URLs or remote resources supplied by users.",
            "Check destination allowlists and validation after URL parsing and redirects.",
            "Review network controls that restrict access to internal services.",
        ],
    },
}

VERIFICATION_OPTIONS = [
    "Unreviewed",
    "Under investigation",
    "Confirmed by reviewer",
    "False positive",
    "Unable to determine",
]


# ---------------------------------------------------------
# Model loading and helper functions
# ---------------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def get_category_info(label):
    if label in SECURITY_INFO:
        return SECURITY_INFO[label]

    # Graceful fallback if a model label is not in our guide.
    return {
        "name": str(label).replace("_", " ").replace("-", " ").title(),
        "short": "The model matched this finding to a security category.",
        "details": (
            "This category was predicted by the trained model. "
            "Review the original evidence before drawing a conclusion."
        ),
        "checks": [
            "Review the original evidence and its source.",
            "Compare the finding with trusted security documentation.",
            "Have a qualified reviewer assess whether the issue is applicable.",
        ],
    }


def predict_text(model, text):
    prediction = model.predict([text])[0]
    confidence = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([text])[0]
        confidence = float(max(probabilities))

    return str(prediction), confidence


def readable_confidence(value):
    if value is None:
        return "Not available"

    if value >= 0.75:
        level = "Higher"
    elif value >= 0.45:
        level = "Moderate"
    else:
        level = "Lower"

    return f"{level} ({value:.1%})"


def create_report_text(results, report_title="GHOSTTRACE Security Analysis"):
    output = [
        report_title,
        "=" * len(report_title),
        f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "IMPORTANT: These are machine-learning predictions, not proof of vulnerabilities.",
        "Human review and appropriate authorized testing are required.",
        "",
    ]

    for index, row in results.reset_index(drop=True).iterrows():
        label = str(row.get("predicted_label", "Unknown"))
        info = get_category_info(label)
        evidence = str(row.get("evidence_text", ""))
        confidence = row.get("model_confidence", None)
        status = str(row.get("review_status", "Unreviewed"))
        notes = str(row.get("reviewer_notes", ""))

        output.extend([
            f"Finding {index + 1}: {info['name']}",
            "-" * 40,
            f"Summary: {info['short']}",
            f"Evidence: {evidence}",
            f"Model confidence: {readable_confidence(confidence) if pd.notna(confidence) else 'Not available'}",
            f"Review status: {status}",
            f"Reviewer notes: {notes}",
            "Suggested review steps:",
        ])

        output.extend(f"  - {step}" for step in info["checks"])
        output.append("")

    return "\n".join(output)


def prepare_download_frame(frame):
    export = frame.copy()

    if "predicted_label" in export.columns:
        export["predicted_category"] = export["predicted_label"].apply(
            lambda value: get_category_info(value)["name"]
        )
        export["plain_english_summary"] = export["predicted_label"].apply(
            lambda value: get_category_info(value)["short"]
        )

    return export


def show_prediction(model, evidence, key_prefix):
    label, confidence = predict_text(model, evidence)
    info = get_category_info(label)

    st.success("Analysis complete")
    st.subheader(info["name"])
    st.write(info["short"])

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Model confidence", readable_confidence(confidence))
    with col2:
        st.metric("Review status", "Unreviewed")

    st.caption(
        "Confidence describes how strongly the model favours its prediction. "
        "It does not measure the probability that a real vulnerability exists."
    )

    with st.expander("What does this mean?", expanded=True):
        st.write(info["details"])

    with st.expander("What should a reviewer check next?"):
        for step in info["checks"]:
            st.markdown(f"- {step}")

    with st.expander("Technical details"):
        st.code(label)
        st.write("Predictions can be wrong. Verify the evidence independently.")

    st.markdown("#### Record your review")
    review_status = st.selectbox(
        "Review status",
        VERIFICATION_OPTIONS,
        index=0,
        key=f"{key_prefix}_status",
    )
    notes = st.text_area(
        "Reviewer notes (optional)",
        placeholder="Record what was checked or what evidence is still needed.",
        key=f"{key_prefix}_notes",
    )

    result = pd.DataFrame([{
        "evidence_text": evidence,
        "predicted_label": label,
        "model_confidence": confidence,
        "review_status": review_status,
        "reviewer_notes": notes,
        "analyzed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }])

    csv_data = prepare_download_frame(result).to_csv(index=False).encode("utf-8-sig")
    report_text = create_report_text(result)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download result as CSV",
            data=csv_data,
            file_name="ghosttrace_finding.csv",
            mime="text/csv",
            key=f"{key_prefix}_csv",
            use_container_width=True,
        )
    with col2:
        st.download_button(
            "Download readable report",
            data=report_text.encode("utf-8"),
            file_name="ghosttrace_report.txt",
            mime="text/plain",
            key=f"{key_prefix}_report",
            use_container_width=True,
        )


# ---------------------------------------------------------
# Friendly interface
# ---------------------------------------------------------

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.7rem; padding-bottom: 3rem;}
    div[data-testid="stMetric"] {
        background: rgba(128,128,128,0.08);
        border: 1px solid rgba(128,128,128,0.18);
        padding: 14px;
        border-radius: 12px;
    }
    .gt-subtitle {color: #9CA3AF; margin-top: -8px;}
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("# 🛡️ GHOSTTRACE")
    st.caption("Security Evidence Analyzer")
    st.divider()
    st.markdown("### How it works")
    st.markdown(
        "1. Provide security-related text.\n"
        "2. AI suggests a category.\n"
        "3. Review the explanation.\n"
        "4. Record your review or export a report."
    )
    st.divider()
    st.caption("AI predictions need human review.")
    st.caption(f"Version {APP_VERSION}")

st.title("Security made easier")
st.markdown(
    '<p class="gt-subtitle">Understand security findings without needing machine-learning expertise.</p>',
    unsafe_allow_html=True,
)

try:
    model = load_model()
except Exception as exc:
    st.error("GHOSTTRACE could not load its trained model.")
    st.markdown(
        "Check that the model file exists at "
        "`models/baseline_tfidf_logreg_clean.joblib` and that your virtual "
        "environment has the required dependencies."
    )
    with st.expander("Technical error details"):
        st.code(str(exc))
    st.stop()

st.success("AI engine ready")

analyze_tab, batch_tab, learn_tab, about_tab = st.tabs([
    "🔎 Analyze evidence",
    "📁 Analyze a file",
    "📘 Learn cybersecurity",
    "ℹ️ About",
])


# ---------------------------------------------------------
# Single finding analysis
# ---------------------------------------------------------

with analyze_tab:
    st.subheader("What would you like to check?")
    st.write(
        "Paste a security report, audit note, or finding from a system "
        "you own or are authorized to assess."
    )

    with st.expander("Try an example"):
        st.code(
            "A user changes the account_id in a request and can view "
            "another user's private profile.",
            language=None,
        )
        st.caption(
            "This is an illustrative example, not a real vulnerability report."
        )

    evidence = st.text_area(
        "Paste the security evidence here",
        height=160,
        placeholder=(
            "Example: The application exposes private account details "
            "when a user changes an identifier..."
        ),
        key="single_evidence",
    )

    if st.button("Analyze evidence", type="primary", use_container_width=True):
        if not evidence.strip():
            st.warning("Enter some evidence before starting the analysis.")
        elif len(evidence.strip()) < 12:
            st.warning(
                "This description is very short. Add more context for a more useful prediction."
            )
        else:
            st.session_state["last_evidence"] = evidence.strip()
            st.session_state["last_analysis_id"] = (
                st.session_state.get("last_analysis_id", 0) + 1
            )

    if "last_evidence" in st.session_state:
        st.divider()
        show_prediction(
            model,
            st.session_state["last_evidence"],
            key_prefix=f"single_{st.session_state.get('last_analysis_id', 0)}",
        )


# ---------------------------------------------------------
# CSV batch analysis
# ---------------------------------------------------------

with batch_tab:
    st.subheader("Analyze multiple findings")
    st.write(
        "Upload a CSV file with a column named `evidence_text`. "
        "Each row should contain one security-related description."
    )

    st.download_button(
        "Download example CSV template",
        data=pd.DataFrame({
            "evidence_text": [
                "A user can view another user's records by changing an account identifier.",
                "The application uses an outdated dependency with a published security advisory.",
            ]
        }).to_csv(index=False).encode("utf-8-sig"),
        file_name="ghosttrace_template.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Maximum 5,000 rows. Files should contain an evidence_text column.",
        key="batch_upload",
    )

    if uploaded_file is not None:
        try:
            uploaded_file.seek(0)

            # Try common encodings and automatically detect the delimiter.
            input_df = None
            read_errors = []

            for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
                try:
                    uploaded_file.seek(0)
                    candidate = pd.read_csv(
                        uploaded_file,
                        sep=None,
                        engine="python",
                        encoding=encoding,
                        on_bad_lines="error",
                    )
                    if len(candidate.columns) > 1 or len(candidate) > 0:
                        input_df = candidate
                        break
                except Exception as exc:
                    read_errors.append(str(exc))

            if input_df is None:
                st.error("Could not read this CSV. Check that it is a valid, non-empty CSV file.")
                with st.expander("Technical details"):
                    st.code("\n".join(read_errors[-3:]))

        except Exception as exc:
            st.error(f"Could not read this CSV: {exc}")
            input_df = None

        if input_df is not None:
            # Remove accidental spaces from column headings.
            input_df.columns = [
                str(column).strip() for column in input_df.columns
            ]

            if len(input_df.columns) == 0:
                st.error("The CSV does not contain any columns.")
            elif len(input_df) == 0:
                st.warning("The CSV contains headers but no data rows.")
            elif len(input_df) > MAX_ROWS:
                st.error(
                    f"This file has {len(input_df):,} rows. "
                    f"The current limit is {MAX_ROWS:,} rows."
                )
            else:
                st.write(f"**{len(input_df):,} records found**")
                st.write("Preview of your uploaded file:")
                st.dataframe(input_df.head(10), use_container_width=True)

                # Look for common names used for text/evidence columns.
                aliases = {
                    "evidence_text", "evidence", "text", "description",
                    "finding", "details", "attack", "attack_description",
                    "context", "message", "content", "conversation",
                    "conversations", "report", "issue", "vulnerability",
                }

                matching_columns = [
                    column for column in input_df.columns
                    if column.strip().lower().replace(" ", "_") in aliases
                ]

                default_column = (
                    matching_columns[0]
                    if matching_columns
                    else input_df.columns[0]
                )

                evidence_column = st.selectbox(
                    "Which column contains the security descriptions?",
                    options=list(input_df.columns),
                    index=list(input_df.columns).index(default_column),
                    help=(
                        "Choose the column containing the actual finding or "
                        "security description, not an ID or category label."
                    ),
                    key=f"evidence_column_{uploaded_file.name}",
                )

                # Normalize the chosen column for the existing pipeline.
                input_df["evidence_text"] = (
                    input_df[evidence_column].fillna("").astype(str).str.strip()
                )

                if st.button(
                    "Analyze all records",
                    type="primary",
                    key="run_batch",
                    use_container_width=True,
                ):
                    texts = input_df["evidence_text"]
                    valid_mask = texts.str.len() >= 12

                    if not valid_mask.any():
                        st.warning(
                            "No usable descriptions found. Choose the correct "
                            "column and make sure it contains text at least "
                            "12 characters long."
                        )
                    else:
                        progress = st.progress(0, text="Analyzing evidence...")
                        valid_texts = texts[valid_mask].tolist()

                        try:
                            predictions = model.predict(valid_texts)

                            if hasattr(model, "predict_proba"):
                                confidences = model.predict_proba(
                                    valid_texts
                                ).max(axis=1)
                            else:
                                confidences = [None] * len(valid_texts)

                            results = input_df.copy()
                            results["predicted_label"] = ""
                            results["model_confidence"] = None
                            results["review_status"] = "Unreviewed"
                            results["reviewer_notes"] = ""
                            results["analyzed_at"] = (
                                datetime.now()
                                .astimezone()
                                .isoformat(timespec="seconds")
                            )

                            results.loc[valid_mask, "predicted_label"] = list(
                                predictions
                            )
                            results.loc[valid_mask, "model_confidence"] = list(
                                confidences
                            )
                            results.loc[
                                ~valid_mask, "review_status"
                            ] = "Needs better evidence"

                            results["predicted_category"] = (
                                results["predicted_label"].apply(
                                    lambda value: (
                                        get_category_info(value)["name"]
                                        if value else "Not analyzed"
                                    )
                                )
                            )
                            results["plain_english_summary"] = (
                                results["predicted_label"].apply(
                                    lambda value: (
                                        get_category_info(value)["short"]
                                        if value
                                        else "Add a longer description before analysis."
                                    )
                                )
                            )

                            progress.progress(100, text="Analysis complete")
                            st.session_state["batch_results"] = results
                            st.session_state["batch_source_name"] = uploaded_file.name

                            st.success(
                                f"Analyzed {int(valid_mask.sum()):,} records. "
                                f"{int((~valid_mask).sum()):,} need better evidence."
                            )

                        except Exception as exc:
                            progress.empty()
                            st.error(f"Analysis failed: {exc}")
                            with st.expander("Technical details"):
                                st.code(str(exc))


    # Display and export batch results
    if "batch_results" in st.session_state:
        results = st.session_state["batch_results"].copy()

        st.divider()
        st.subheader("Your results")

        analyzed = results[
            results["predicted_label"].astype(str).str.len() > 0
        ]

        if not analyzed.empty:
            st.metric("Analyzed findings", len(analyzed))

            category_counts = (
                analyzed["predicted_label"]
                .map(lambda label: get_category_info(label)["name"])
                .value_counts()
            )

            st.markdown("#### Predicted categories")
            st.bar_chart(category_counts)

        st.caption(
            "Review status and notes are for human review. "
            "AI predictions do not confirm vulnerabilities."
        )

        display_columns = [
            "evidence_text",
            "predicted_category",
            "plain_english_summary",
            "model_confidence",
            "review_status",
            "reviewer_notes",
        ]

        display_columns = [
            column for column in display_columns
            if column in results.columns
        ]

        edited = st.data_editor(
            results[display_columns],
            use_container_width=True,
            hide_index=True,
            disabled=[
                column for column in display_columns
                if column not in ("review_status", "reviewer_notes")
            ],
            key="batch_results_editor",
        )

        for column in ("review_status", "reviewer_notes"):
            if column in edited.columns:
                results[column] = edited[column].values

        st.session_state["batch_results"] = results

        export_df = prepare_download_frame(results)

        csv_data = export_df.to_csv(index=False).encode("utf-8-sig")
        report_data = create_report_text(results).encode("utf-8")

        col1, col2 = st.columns(2)

        with col1:
            st.download_button(
                "Download results (CSV)",
                data=csv_data,
                file_name="ghosttrace_batch_results.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_batch_csv",
            )

        with col2:
            st.download_button(
                "Download readable report",
                data=report_data,
                file_name="ghosttrace_batch_report.txt",
                mime="text/plain",
                use_container_width=True,
                key="download_batch_report",
            )


# ---------------------------------------------------------
# Built-in cybersecurity learning guide
# ---------------------------------------------------------

with learn_tab:
    st.subheader("Cybersecurity in plain English")
    st.write(
        "Choose a topic to understand what it means and what a reviewer can check. "
        "These explanations are educational, not proof of a vulnerability."
    )

    topic_names = [info["name"] for info in SECURITY_INFO.values()]
    topic_labels = list(SECURITY_INFO.keys())
    selected_name = st.selectbox("Choose a security topic", topic_names)
    selected_label = next(
        label for label in topic_labels
        if SECURITY_INFO[label]["name"] == selected_name
    )
    info = SECURITY_INFO[selected_label]

    st.markdown(f"### {info['name']}")
    st.info(info["short"])
    st.write(info["details"])

    st.markdown("#### What can a reviewer check?")
    for step in info["checks"]:
        st.markdown(f"- {step}")

    st.markdown("#### Example")
    examples = {
        "A01_broken_access_control":
            "A customer can see another customer's invoice by changing an identifier.",
        "A02_cryptographic_failures":
            "Sensitive information is stored without appropriate protection.",
        "A03_injection":
            "Untrusted input is combined unsafely with a database query.",
        "A04_insecure_design":
            "A sensitive workflow has no suitable limit or abuse prevention.",
        "A05_security_misconfiguration":
            "A production service exposes a debug page containing internal details.",
        "A06_vulnerable_and_outdated_components":
            "An application uses a dependency version affected by a known security advisory.",
        "A07_identification_and_authentication_failures":
            "An account recovery process does not adequately verify the user.",
        "A08_software_and_data_integrity_failures":
            "An update process accepts a package without verifying its authenticity.",
        "A09_security_logging_and_monitoring_failures":
            "Repeated failed logins are not recorded or surfaced for review.",
        "A10_server_side_request_forgery":
            "A URL-fetching feature does not adequately restrict where the server can connect.",
    }
    st.code(examples[selected_label], language=None)

    st.caption(
        "The OWASP Top 10:2021 is a general awareness document. "
        "A predicted category is not a complete security assessment."
    )


# ---------------------------------------------------------
# About, limitations, and methodology
# ---------------------------------------------------------

with about_tab:
    st.subheader("About GHOSTTRACE")
    st.write(
        "GHOSTTRACE is an experimental AI-assisted tool that classifies "
        "security-related text into likely OWASP Top 10:2021 categories."
    )

    st.markdown("#### What it can do")
    st.markdown(
        "- Classify supplied security descriptions.\n"
        "- Explain predicted categories in plain English.\n"
        "- Analyze CSV files containing multiple findings.\n"
        "- Export predictions and reviewer notes."
    )

    st.markdown("#### What it cannot do")
    st.markdown(
        "- It does not scan websites or networks in this version.\n"
        "- It does not exploit systems or confirm vulnerabilities.\n"
        "- It may misclassify vague, unusual, or out-of-distribution evidence.\n"
        "- Its confidence score is not the probability that a vulnerability is real."
    )

    st.markdown("#### How the AI works")
    st.write(
        "The current model uses TF-IDF to turn text into numerical features "
        "and Logistic Regression to select a likely category. The model was "
        "trained and evaluated on a limited experimental dataset."
    )

    st.markdown("#### Responsible use")
    st.write(
        "Only analyze information you are authorized to handle. Verify findings "
        "with trusted documentation and appropriate testing. Keep sensitive "
        "evidence out of reports unless sharing is authorized."
    )

    st.markdown("#### Data and privacy")
    st.write(
        "This app processes uploaded CSV data in the running Streamlit session. "
        "It does not intentionally send your evidence to an external analysis API. "
        "Avoid uploading secrets or personal information. Downloaded reports are "
        "saved by your browser when you choose to download them."
    )

