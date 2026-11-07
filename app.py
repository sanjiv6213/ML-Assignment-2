
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Room Occupancy | ML Dashboard",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

    /* Main application */
    .stApp {
        background-color: #0E1117;
        color: #E6EDF3;
    }

    /* Main content width */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #151B26;
        border-right: 1px solid #263244;
    }

    /* Main title */
    .main-title {
        font-size: 34px;
        font-weight: 750;
        color: #F0F4FA;
        margin-bottom: 3px;
    }

    .subtitle {
        font-size: 14px;
        color: #9BA9BB;
        margin-bottom: 25px;
    }

    /* Section headings */
    .section-title {
        font-size: 21px;
        font-weight: 650;
        color: #E6EDF3;
        margin-top: 12px;
        margin-bottom: 15px;
    }

    /* KPI cards */
    div[data-testid="stMetric"] {
        background-color: #192333;
        border: 1px solid #29384C;
        padding: 18px;
        border-radius: 12px;
    }

    div[data-testid="stMetricLabel"] {
        color: #A9B8CA;
        font-size: 13px;
    }

    div[data-testid="stMetricValue"] {
        color: #63B3ED;
        font-weight: 700;
    }

    /* Information cards */
    .info-card {
        background-color: #192333;
        border: 1px solid #29384C;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 12px;
    }

    .info-label {
        color: #9BA9BB;
        font-size: 13px;
    }

    .info-value {
        color: #E6EDF3;
        font-size: 23px;
        font-weight: 700;
        margin-top: 5px;
    }

    /* Divider */
    hr {
        border-color: #29384C;
    }

    /* Buttons */
    .stButton > button {
        background-color: #2563EB;
        color: white;
        border-radius: 8px;
        border: none;
    }

    .stButton > button:hover {
        background-color: #1D4ED8;
        color: white;
    }

    /* Dataframes */
    [data-testid="stDataFrame"] {
        border: 1px solid #29384C;
        border-radius: 10px;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        color: #A9B8CA;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #63B3ED;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #718096;
        font-size: 12px;
        padding-top: 25px;
    }

</style>
""", unsafe_allow_html=True)


# ==========================================================
# CONFIGURATION
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

TARGET = "Room_Occupancy_Count"

FEATURES = [
    "S1_Temp",
    "S2_Temp",
    "S3_Temp",
    "S4_Temp",
    "S1_Light",
    "S2_Light",
    "S3_Light",
    "S4_Light",
    "S1_Sound",
    "S2_Sound",
    "S3_Sound",
    "S4_Sound",
    "S5_CO2",
    "S5_CO2_Slope",
    "S6_PIR",
    "S7_PIR"
]

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib"
}

CLASS_NAMES = {
    0: "Empty",
    1: "One Person",
    2: "Two People",
    3: "Three People"
}

CLASS_ORDER = [0, 1, 2, 3]


# ==========================================================
# LOAD MODELS
# ==========================================================

@st.cache_resource
def load_models():

    loaded_models = {}

    for model_name, file_name in MODEL_FILES.items():

        file_path = os.path.join(MODEL_DIR, file_name)

        if os.path.isfile(file_path):
            loaded_models[model_name] = joblib.load(file_path)

    return loaded_models


try:
    models = load_models()

except Exception as e:
    st.error(f"Unable to load saved models: {e}")
    st.stop()


if len(models) != 5:

    st.error("Some trained models are missing.")

    st.write("Loaded models:", list(models.keys()))

    if os.path.isdir(MODEL_DIR):
        st.write("Available files:", os.listdir(MODEL_DIR))

    st.stop()


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.markdown("## 🏢 Occupancy AI")

    st.caption("Machine Learning Evaluation Platform")

    st.divider()

    st.markdown("### Dataset Upload")

    uploaded_file = st.file_uploader(
        "Upload your test dataset",
        type=["csv"],
        help="CSV must contain 16 sensor features and the occupancy target."
    )

    st.caption("Required format: CSV")

    st.divider()

    st.markdown("### Model Selection")

    selected_model_name = st.selectbox(
        "Choose a model",
        list(models.keys())
    )

    st.divider()

    st.markdown("### Project Information")

    st.caption("Dataset: UCI Room Occupancy Estimation")

    st.caption("Models: 5 Classification Algorithms")

    st.caption("Input Features: 16 Sensor Variables")

    st.caption("Target: Room Occupancy Count")

    st.divider()

    st.markdown("### Model Status")

    st.success("All 5 models loaded successfully")


# ==========================================================
# HEADER
# ==========================================================

st.markdown(
    '<div class="main-title">Room Occupancy Estimation</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Intelligent occupancy analysis using machine learning '
    '| Sensor-based classification'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ==========================================================
# UPLOAD VALIDATION
# ==========================================================

if uploaded_file is None:

    st.markdown("## Welcome to the Occupancy Dashboard")

    st.write(
        "Upload your test dataset using the sidebar to explore "
        "model performance, occupancy patterns and evaluation results."
    )

    st.info(
        "Your CSV must contain all 16 sensor features and "
        "the Room_Occupancy_Count target column."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Available Models", "5")
    col2.metric("Sensor Features", "16")
    col3.metric("Occupancy Classes", "4")

    st.markdown("### Available Classification Models")

    model_columns = st.columns(5)

    for i, name in enumerate(models.keys()):
        with model_columns[i]:
            st.markdown(
                f"""
                <div class="info-card">
                    <div class="info-label">Model {i + 1}</div>
                    <div style="font-size:15px;font-weight:600;">
                        {name}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.stop()


# ==========================================================
# READ AND VALIDATE DATA
# ==========================================================

try:

    data = pd.read_csv(uploaded_file)

except Exception as e:

    st.error(f"Unable to read the uploaded CSV: {e}")
    st.stop()


required_columns = FEATURES + [TARGET]

missing_columns = [
    col for col in required_columns
    if col not in data.columns
]

if missing_columns:

    st.error(
        "Missing required columns: "
        + ", ".join(missing_columns)
    )

    st.stop()


if data.empty:

    st.error("The uploaded dataset is empty.")
    st.stop()


X = data[FEATURES].copy()
y = data[TARGET].copy()


if X.isnull().any().any() or y.isnull().any():

    st.error("The dataset contains missing values.")
    st.stop()


try:

    X = X.apply(pd.to_numeric, errors="raise")

    y_numeric = pd.to_numeric(y, errors="raise")

    if not np.isfinite(y_numeric.to_numpy()).all():
        raise ValueError("Target contains non-finite values.")

    if not np.equal(
        y_numeric.to_numpy(),
        np.floor(y_numeric.to_numpy())
    ).all():
        raise ValueError("Target values must be integers.")

    y = y_numeric.astype(int)

except Exception as e:

    st.error(f"Invalid numerical values: {e}")
    st.stop()


if not np.isfinite(X.to_numpy()).all():

    st.error("Sensor features contain infinite values.")
    st.stop()


if not y.isin(CLASS_ORDER).all():

    st.error("Occupancy target must contain only 0, 1, 2 or 3.")
    st.stop()


# ==========================================================
# EVALUATION FUNCTION
# ==========================================================

def evaluate_model(model, X, y):

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)

    accuracy = accuracy_score(y, predictions)

    precision = precision_score(
        y,
        predictions,
        labels=CLASS_ORDER,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        labels=CLASS_ORDER,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        labels=CLASS_ORDER,
        average="macro",
        zero_division=0
    )

    mcc = matthews_corrcoef(y, predictions)

    # AUC requires all four actual classes in the uploaded data.
    if (
        set(y.unique()) == set(CLASS_ORDER)
        and list(model.classes_) == CLASS_ORDER
    ):

        auc = roc_auc_score(
            y,
            probabilities,
            labels=CLASS_ORDER,
            multi_class="ovr",
            average="macro"
        )

    else:
        auc = np.nan

    return {
        "predictions": predictions,
        "probabilities": probabilities,
        "Accuracy": accuracy,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "MCC": mcc
    }


# ==========================================================
# EVALUATE ALL MODELS
# ==========================================================

with st.spinner("Evaluating all five machine learning models..."):

    all_results = {}

    try:

        for name, model in models.items():

            all_results[name] = evaluate_model(
                model,
                X,
                y
            )

    except Exception as e:

        st.error(f"Model evaluation failed: {e}")
        st.stop()


# ==========================================================
# DATASET OVERVIEW
# ==========================================================

st.markdown('<div class="section-title">Dataset Overview</div>',
            unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Records", f"{len(data):,}")

col2.metric("Sensor Features", len(FEATURES))

col3.metric("Occupancy Classes", y.nunique())

col4.metric("Selected Model", selected_model_name)


# ==========================================================
# NAVIGATION TABS
# ==========================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "🔍 Model Evaluation",
    "⚖️ Model Comparison",
    "📁 Dataset Explorer"
])


# ==========================================================
# TAB 1 - OVERVIEW
# ==========================================================

with tab1:

    st.markdown(
        '<div class="section-title">Occupancy Distribution</div>',
        unsafe_allow_html=True
    )

    distribution = (
        y.value_counts()
        .reindex(CLASS_ORDER, fill_value=0)
        .rename(index=CLASS_NAMES)
        .reset_index()
    )

    distribution.columns = ["Occupancy", "Records"]

    col1, col2 = st.columns([1.3, 1])

    with col1:

        st.markdown("#### Records by Occupancy")

        st.bar_chart(
            distribution.set_index("Occupancy"),
            use_container_width=True
        )

    with col2:

        st.markdown("#### Occupancy Summary")

        for class_id, class_name in CLASS_NAMES.items():

            count = int((y == class_id).sum())

            percentage = count / len(y) * 100

            st.markdown(
                f"""
                <div class="info-card">
                    <div class="info-label">{class_name}</div>
                    <div class="info-value">{count:,}</div>
                    <div class="info-label">{percentage:.2f}% of records</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.divider()

    st.markdown("#### Selected Model Performance")

    result = all_results[selected_model_name]

    c1, c2, c3 = st.columns(3)

    c1.metric("Accuracy", f"{result['Accuracy']:.4f}")
    c2.metric(
        "AUC",
        f"{result['AUC']:.4f}" if not np.isnan(result["AUC"]) else "N/A"
    )
    c3.metric("Precision", f"{result['Precision']:.4f}")

    c4, c5, c6 = st.columns(3)

    c4.metric("Recall", f"{result['Recall']:.4f}")
    c5.metric("F1 Score", f"{result['F1 Score']:.4f}")
    c6.metric("MCC", f"{result['MCC']:.4f}")


# ==========================================================
# TAB 2 - MODEL EVALUATION
# ==========================================================

with tab2:

    st.markdown(
        '<div class="section-title">Detailed Model Evaluation</div>',
        unsafe_allow_html=True
    )

    st.caption(f"Currently selected: {selected_model_name}")

    result = all_results[selected_model_name]

    predictions = result["predictions"]

    # Metrics
    c1, c2, c3 = st.columns(3)

    c1.metric("Accuracy", f"{result['Accuracy']:.4f}")
    c2.metric(
        "AUC",
        f"{result['AUC']:.4f}" if not np.isnan(result["AUC"]) else "N/A"
    )
    c3.metric("Precision", f"{result['Precision']:.4f}")

    c4, c5, c6 = st.columns(3)

    c4.metric("Recall", f"{result['Recall']:.4f}")
    c5.metric("F1 Score", f"{result['F1 Score']:.4f}")
    c6.metric("MCC", f"{result['MCC']:.4f}")

    st.divider()

    # Confusion matrix
    st.markdown("#### Confusion Matrix")

    cm = confusion_matrix(
        y,
        predictions,
        labels=CLASS_ORDER
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    fig.patch.set_facecolor("#192333")
    ax.set_facecolor("#192333")

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=list(CLASS_NAMES.values()),
        yticklabels=list(CLASS_NAMES.values()),
        linewidths=0.5,
        linecolor="#29384C",
        ax=ax
    )

    ax.set_xlabel("Predicted Occupancy", color="white")
    ax.set_ylabel("Actual Occupancy", color="white")
    ax.tick_params(colors="white")

    plt.tight_layout()

    st.pyplot(fig, use_container_width=True)

    plt.close(fig)

    st.divider()

    # Classification report
    st.markdown("#### Classification Report")

    report = classification_report(
        y,
        predictions,
        labels=CLASS_ORDER,
        target_names=list(CLASS_NAMES.values()),
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df.round(4),
        use_container_width=True
    )


# ==========================================================
# TAB 3 - MODEL COMPARISON
# ==========================================================

with tab3:

    st.markdown(
        '<div class="section-title">Model Performance Comparison</div>',
        unsafe_allow_html=True
    )

    comparison_rows = []

    for name, result in all_results.items():

        comparison_rows.append({
            "Model": name,
            "Accuracy": result["Accuracy"],
            "AUC": result["AUC"],
            "Precision": result["Precision"],
            "Recall": result["Recall"],
            "F1 Score": result["F1 Score"],
            "MCC": result["MCC"]
        })

    comparison_df = pd.DataFrame(comparison_rows)

    st.dataframe(
        comparison_df.round(4),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        label="⬇ Download Comparison Results",
        data=comparison_df.to_csv(index=False).encode("utf-8"),
        file_name="model_comparison_results.csv",
        mime="text/csv"
    )

    st.divider()

    st.markdown("#### Performance Across Models")

    selected_metric = st.selectbox(
        "Choose a metric to compare",
        [
            "Accuracy",
            "AUC",
            "Precision",
            "Recall",
            "F1 Score",
            "MCC"
        ]
    )

    chart_data = comparison_df[
        ["Model", selected_metric]
    ].set_index("Model")

    st.bar_chart(
        chart_data,
        use_container_width=True
    )

    # Winner based on F1 score
    winner = comparison_df.loc[
        comparison_df["F1 Score"].idxmax()
    ]

    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-label">Highest F1 Score on Uploaded Dataset</div>
            <div class="info-value">{winner['Model']}</div>
            <div class="info-label">
                F1 Score: {winner['F1 Score']:.4f}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.caption(
        "Comparison metrics are calculated using the currently uploaded "
        "dataset. The models are loaded from saved files and are not retrained."
    )


# ==========================================================
# TAB 4 - DATASET EXPLORER
# ==========================================================

with tab4:

    st.markdown(
        '<div class="section-title">Dataset Explorer</div>',
        unsafe_allow_html=True
    )

    st.markdown("#### Uploaded Dataset Preview")

    st.dataframe(
        data.head(20),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.markdown("#### Sensor Feature Statistics")

    st.dataframe(
        X.describe().round(3),
        use_container_width=True
    )

    st.divider()

    st.markdown("#### Occupancy Class Distribution")

    class_distribution = (
        y.value_counts()
        .reindex(CLASS_ORDER, fill_value=0)
        .rename(index=CLASS_NAMES)
        .rename_axis("Occupancy")
        .reset_index(name="Count")
    )

    st.dataframe(
        class_distribution,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        label="⬇ Download Uploaded Dataset",
        data=data.to_csv(index=False).encode("utf-8"),
        file_name="uploaded_occupancy_data.csv",
        mime="text/csv"
    )


# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.markdown(
    """
    <div class="footer">
        Room Occupancy Estimation | Machine Learning Assignment<br>
        Logistic Regression • Decision Tree • KNN • Naive Bayes • SVM
    </div>
    """,
    unsafe_allow_html=True
)
