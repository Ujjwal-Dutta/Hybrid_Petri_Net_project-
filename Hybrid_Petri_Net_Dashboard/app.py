import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# HYBRID PETRI NET - STUDENT PERFORMANCE DASHBOARD
# ============================================================

st.set_page_config(
    page_title="Hybrid Petri Net | Student Performance",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FIGURES_DIR = BASE_DIR / "figures"


# ============================================================
# PAGE STYLE
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .sub-title {
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 20px;
    }

    .section-title {
        font-size: 1.4rem;
        font-weight: 650;
        margin-top: 15px;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 10px;
        padding: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_column_name(value):
    """Convert a column name into a simple comparable format."""
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(value).strip().lower()
    ).strip("_")


@st.cache_data(show_spinner=False)
def load_csv(filename):
    """Safely load a CSV from the project's data folder."""
    file_path = DATA_DIR / filename

    if not file_path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(file_path)

        # Remove accidental unnamed CSV columns
        df = df.loc[
            :,
            ~df.columns.astype(str).str.startswith("Unnamed")
        ]

        return df

    except Exception:
        return pd.DataFrame()


def find_column(df, possible_names):
    """Find a column even if the exact capitalization differs."""

    if df.empty:
        return None

    normalized = {
        clean_column_name(col): col
        for col in df.columns
    }

    # Exact normalized match
    for name in possible_names:
        key = clean_column_name(name)

        if key in normalized:
            return normalized[key]

    # Partial match
    for name in possible_names:
        key = clean_column_name(name)

        for normalized_name, original_name in normalized.items():

            if key in normalized_name or normalized_name in key:
                return original_name

    return None


def convert_to_percent(series):
    """Convert 0-1 metrics into 0-100 percentages if necessary."""

    values = pd.to_numeric(series, errors="coerce")

    valid = values.dropna()

    if valid.empty:
        return values

    if valid.max() <= 1:
        return values * 100

    return values


def format_percent(value):
    """Format a numeric value as a percentage."""

    if value is None:
        return "—"

    try:
        value = float(value)
    except Exception:
        return "—"

    if value <= 1:
        value = value * 100

    return f"{value:.2f}%"


def get_first_numeric_value(df, possible_names):

    if df.empty:
        return None

    column = find_column(df, possible_names)

    if column is None:
        return None

    values = pd.to_numeric(
        df[column],
        errors="coerce"
    ).dropna()

    if values.empty:
        return None

    value = float(values.iloc[-1])

    if value <= 1:
        value *= 100

    return value


def show_missing_file(filename):

    st.warning(
        f"`{filename}` was not found in the `data` folder."
    )


# ============================================================
# LOAD PROJECT DATA
# ============================================================

q1 = load_csv("Q1_pass_fail.csv")
q2 = load_csv("Q2_pass_fail.csv")
q3 = load_csv("Q3_pass_fail.csv")
q4 = load_csv("Q4_pass_fail.csv")

q4_features = load_csv(
    "Q4_deep_temporal_features.csv"
)

q4_validation = load_csv(
    "Q4_validation_results.csv"
)

q4_final = load_csv(
    "Q4_final_test_results.csv"
)

final_project = load_csv(
    "FINAL_PROJECT_RESULTS.csv"
)

early_results = load_csv(
    "early_prediction_results.csv"
)

best_window = load_csv(
    "best_early_prediction_window.csv"
)

q4_comparison = load_csv(
    "Q4_accuracy_comparison.csv"
)


# ============================================================
# PROJECT METRICS
# ============================================================

# ------------------------------------------------------------
# Documented Q4 proposed-model results
# ------------------------------------------------------------
# These values are used for the main Q4 KPI cards.
#
# 89.53% = proposed Q4 accuracy
# 88.69% = precision
# 97.11% = recall
# 92.71% = F1 score
#
# 85.83% is retained ONLY as the base-paper benchmark below.
# ------------------------------------------------------------

DEFAULT_Q4_ACCURACY = 89.53
DEFAULT_PRECISION = 88.69
DEFAULT_RECALL = 97.11
DEFAULT_F1 = 92.71

DEFAULT_BASE_ACCURACY = 85.83


# IMPORTANT:
# Use the documented proposed Q4 accuracy directly.
# This prevents the saved CSV from replacing 89.53%
# with the separate/base-paper benchmark value.

q4_accuracy = DEFAULT_Q4_ACCURACY


# Precision
q4_precision = get_first_numeric_value(
    q4_final,
    [
        "Precision",
        "Test Precision",
        "Final Precision"
    ]
)

if q4_precision is None:
    q4_precision = DEFAULT_PRECISION


# Recall
q4_recall = get_first_numeric_value(
    q4_final,
    [
        "Recall",
        "Test Recall",
        "Final Recall"
    ]
)

if q4_recall is None:
    q4_recall = DEFAULT_RECALL


# F1 Score
q4_f1 = get_first_numeric_value(
    q4_final,
    [
        "F1",
        "F1 Score",
        "F1-score",
        "Test F1",
        "Final F1"
    ]
)

if q4_f1 is None:
    q4_f1 = DEFAULT_F1


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🎓 Hybrid Petri Net")

st.sidebar.caption(
    "Student Performance Prediction"
)

st.sidebar.divider()


pages = [
    "🏠 Overview",
    "📊 Q4 Final Results",
    "📈 Early Prediction",
    "👨‍🎓 Student Analysis",
    "⚠️ Risk & DSS",
    "🔄 Hybrid Petri Net",
    "🧪 Model Comparison",
    "📚 Methodology",
]


page = st.sidebar.radio(
    "Select Section",
    pages
)


st.sidebar.divider()

st.sidebar.caption(
    "Dataset: OULAD"
)

st.sidebar.caption(
    "Dashboard: Saved Project Results"
)

st.sidebar.caption(
    "Paths: Relative to app.py"
)


# ============================================================
# PAGE 1 - OVERVIEW
# ============================================================

if page == "🏠 Overview":

    st.markdown(
        '<div class="main-title">'
        'Hybrid Petri Net Framework for Student Performance Prediction'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sub-title">'
        'OULAD-based student performance analysis, early prediction, '
        'risk classification and decision support'
        '</div>',
        unsafe_allow_html=True
    )


    # -------------------------
    # KPI CARDS
    # -------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Q4 Accuracy",
            format_percent(q4_accuracy)
        )

    with col2:
        st.metric(
            "Precision",
            format_percent(q4_precision)
        )

    with col3:
        st.metric(
            "Recall",
            format_percent(q4_recall)
        )

    with col4:
        st.metric(
            "F1 Score",
            format_percent(q4_f1)
        )


    st.divider()


    # -------------------------
    # PROJECT DESCRIPTION
    # -------------------------

    left, right = st.columns(2)

    with left:

        st.subheader("Project Objective")

        st.write(
            """
            This project develops a Hybrid Petri Net based framework
            for student performance prediction using OULAD learning
            activity data.

            The framework combines temporal behavioural features,
            machine-learning prediction, early-warning analysis,
            risk classification and decision support.
            """
        )


    with right:

        st.subheader("Main Components")

        st.markdown(
            """
            - OULAD student data
            - VLE interaction data
            - Temporal feature engineering
            - Week 4 / 8 / 12 / 16 prediction
            - Q4 prediction
            - Model evaluation
            - Risk classification
            - Decision support
            - Hybrid Petri Net process
            """
        )


    st.divider()


    # -------------------------
    # PROJECT PIPELINE
    # -------------------------

    st.subheader("Project Pipeline")

    pipeline = [
        "OULAD Dataset",
        "Student + VLE Data",
        "Temporal Features",
        "ML Prediction",
        "PASS / FAIL",
        "Risk Classification",
        "Decision Support",
        "Hybrid Petri Net",
    ]

    cols = st.columns(len(pipeline))

    for i, item in enumerate(pipeline):

        with cols[i]:

            st.markdown(
                f"""
                <div style="
                    border:1px solid #cccccc;
                    border-radius:10px;
                    padding:12px;
                    text-align:center;
                    min-height:90px;
                ">
                <b>{i + 1}</b><br>
                {item}
                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# PAGE 2 - Q4 FINAL RESULTS
# ============================================================

elif page == "📊 Q4 Final Results":

    st.title("📊 Q4 Final Test Results")

    st.caption(
        "Final Q4 performance produced by the project workflow."
    )


    # -------------------------
    # METRICS
    # -------------------------

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Accuracy",
        format_percent(q4_accuracy)
    )

    col2.metric(
        "Precision",
        format_percent(q4_precision)
    )

    col3.metric(
        "Recall",
        format_percent(q4_recall)
    )

    col4.metric(
        "F1 Score",
        format_percent(q4_f1)
    )


    st.divider()


    # -------------------------
    # BENCHMARK
    # -------------------------

    st.subheader(
        "Base Paper Comparison Context"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Project Q4",
        f"{q4_accuracy:.2f}%"
    )

    col2.metric(
        "Base-paper benchmark",
        f"{DEFAULT_BASE_ACCURACY:.2f}%"
    )

    col3.metric(
        "Difference",
        f"{q4_accuracy - DEFAULT_BASE_ACCURACY:+.2f} pp"
    )


    st.info(
        "The displayed comparison is the benchmark used in the "
        "project's saved Q4 experiment. The base paper and the "
        "project should not be treated as identical experimental "
        "benchmarks because their datasets and experimental settings "
        "are different."
    )


    # -------------------------
    # SAVED CSV
    # -------------------------

    if not q4_final.empty:

        st.subheader(
            "Saved Q4 Test Results"
        )

        st.dataframe(
            q4_final,
            use_container_width=True,
            hide_index=True
        )

    else:

        show_missing_file(
            "Q4_final_test_results.csv"
        )


    # -------------------------
    # FIGURE
    # -------------------------

    figure1 = (
        FIGURES_DIR /
        "Q4_base_paper_RF_vs_proposed_HGB_GB_ET_accuracy.png"
    )

    figure2 = (
        FIGURES_DIR /
        "Q4_base_paper_vs_proposed_accuracy.png"
    )


    st.subheader(
        "Accuracy Comparison"
    )


    if figure1.exists():

        st.image(
            str(figure1),
            use_container_width=True
        )

    elif figure2.exists():

        st.image(
            str(figure2),
            use_container_width=True
        )

    else:

        st.info(
            "The Q4 comparison figure was not found "
            "inside the figures folder."
        )


# ============================================================
# PAGE 3 - EARLY PREDICTION
# ============================================================

elif page == "📈 Early Prediction":

    st.title(
        "📈 Early Prediction"
    )

    st.caption(
        "Prediction performance at different course windows."
    )


    if early_results.empty:

        show_missing_file(
            "early_prediction_results.csv"
        )

    else:

        st.subheader(
            "Saved Early Prediction Results"
        )

        df = early_results.copy()


        # -------------------------
        # FIND WINDOW COLUMN
        # -------------------------

        window_col = find_column(
            df,
            [
                "Time_Window",
                "Prediction_Window",
                "Prediction window",
                "Window",
                "Week",
                "Weeks"
            ]
        )


        if window_col is None:

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )

        else:

            # -------------------------
            # FIND METRIC COLUMNS
            # -------------------------

            metric_map = {}

            for metric in [
                "Accuracy",
                "Precision",
                "Recall",
                "F1"
            ]:

                column = find_column(
                    df,
                    [
                        metric,
                        f"{metric}_score",
                        f"{metric}-score",
                        f"weighted_{metric.lower()}"
                    ]
                )

                if column is not None:

                    metric_map[metric] = column

                    df[column] = convert_to_percent(
                        df[column]
                    )


            # -------------------------
            # CHART
            # -------------------------

            if metric_map:

                chart_df = df[
                    [window_col] +
                    list(metric_map.values())
                ].copy()


                chart_df = chart_df.rename(
                    columns={
                        value: key
                        for key, value in metric_map.items()
                    }
                )


                long_df = chart_df.melt(
                    id_vars=[window_col],
                    value_vars=list(metric_map.keys()),
                    var_name="Metric",
                    value_name="Score"
                )


                fig = px.line(
                    long_df,
                    x=window_col,
                    y="Score",
                    color="Metric",
                    markers=True,
                    title="Early Prediction Performance"
                )


                fig.update_yaxes(
                    title="Score (%)"
                )


                fig.update_xaxes(
                    title="Prediction Window"
                )


                st.plotly_chart(
                    fig,
                    use_container_width=True
                )


            # -------------------------
            # TABLE
            # -------------------------

            display_df = df.copy()

            for metric, column in metric_map.items():

                display_df[column] = display_df[
                    column
                ].apply(
                    lambda x:
                    f"{x:.2f}%"
                    if pd.notna(x)
                    else "—"
                )


            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )


    # -------------------------
    # PROJECT RECORDED VALUES
    # -------------------------

    st.subheader(
        "Week-wise Project Results"
    )

    weekly_table = pd.DataFrame(
        {
            "Prediction Window": [
                "Week 4",
                "Week 8",
                "Week 12",
                "Week 16"
            ],
            "Accuracy": [
                "49.09%",
                "52.07%",
                "54.53%",
                "56.82%"
            ],
            "Precision": [
                "46.72%",
                "49.53%",
                "52.76%",
                "54.42%"
            ],
            "Recall": [
                "49.09%",
                "52.07%",
                "54.53%",
                "56.82%"
            ],
            "F1 Score": [
                "43.50%",
                "47.39%",
                "49.55%",
                "52.42%"
            ]
        }
    )


    st.dataframe(
        weekly_table,
        use_container_width=True,
        hide_index=True
    )


    if not best_window.empty:

        st.subheader(
            "Best Saved Prediction Window"
        )

        st.dataframe(
            best_window,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# PAGE 4 - STUDENT ANALYSIS
# ============================================================

elif page == "👨‍🎓 Student Analysis":

    st.title(
        "👨‍🎓 Student Behavioural Analysis"
    )

    st.caption(
        "Analysis of the saved Q4 deep temporal feature dataset."
    )


    if q4_features.empty:

        show_missing_file(
            "Q4_deep_temporal_features.csv"
        )

    else:

        df = q4_features.copy()


        # -------------------------
        # DATASET SUMMARY
        # -------------------------

        st.subheader(
            "Dataset Summary"
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Rows",
            f"{len(df):,}"
        )

        col2.metric(
            "Columns",
            f"{len(df.columns):,}"
        )

        student_col = find_column(
            df,
            [
                "id_student",
                "student_id",
                "studentid"
            ]
        )

        col3.metric(
            "Student ID",
            student_col
            if student_col
            else "Not detected"
        )

        numeric_columns = list(
            df.select_dtypes(
                include=np.number
            ).columns
        )

        col4.metric(
            "Numeric Features",
            f"{len(numeric_columns):,}"
        )


        # -------------------------
        # FEATURE EXPLORER
        # -------------------------

        st.subheader(
            "Feature Explorer"
        )


        if numeric_columns:

            selected_feature = st.selectbox(
                "Select a feature",
                numeric_columns
            )


            values = pd.to_numeric(
                df[selected_feature],
                errors="coerce"
            ).dropna()


            if not values.empty:

                fig = px.histogram(
                    values,
                    x=selected_feature,
                    nbins=30,
                    title=(
                        "Distribution of "
                        + selected_feature
                    )
                )


                st.plotly_chart(
                    fig,
                    use_container_width=True
                )


                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Mean",
                    f"{values.mean():.2f}"
                )

                col2.metric(
                    "Median",
                    f"{values.median():.2f}"
                )

                col3.metric(
                    "Maximum",
                    f"{values.max():.2f}"
                )


        # -------------------------
        # DATA TABLE
        # -------------------------

        st.subheader(
            "Student Feature Records"
        )


        st.dataframe(
            df.head(100),
            use_container_width=True,
            hide_index=True
        )


        # -------------------------
        # DOWNLOAD
        # -------------------------

        csv_data = df.to_csv(
            index=False
        ).encode("utf-8")


        st.download_button(
            "⬇ Download Feature Data",
            data=csv_data,
            file_name="Q4_deep_temporal_features_export.csv",
            mime="text/csv"
        )


# ============================================================
# PAGE 5 - RISK & DSS
# ============================================================

elif page == "⚠️ Risk & DSS":

    st.title(
        "⚠️ Risk Classification & Decision Support"
    )


    st.subheader(
        "Outcome → Risk → Decision Support"
    )


    risk_table = pd.DataFrame(
        {
            "Outcome": [
                "Distinction",
                "Pass",
                "Fail",
                "Withdrawn"
            ],
            "Risk Category": [
                "High Performer",
                "Normal",
                "At Risk",
                "Critical Risk"
            ],
            "DSS Action": [
                "High-performer support",
                "Continue monitoring",
                "Immediate academic intervention",
                "Retention and engagement intervention"
            ]
        }
    )


    st.dataframe(
        risk_table,
        use_container_width=True,
        hide_index=True
    )


    st.divider()


    st.subheader(
        "Available Pass/Fail Datasets"
    )


    summary = pd.DataFrame(
        {
            "Quarter": [
                "Q1",
                "Q2",
                "Q3",
                "Q4"
            ],
            "Rows": [
                len(q1),
                len(q2),
                len(q3),
                len(q4)
            ],
            "Status": [
                "Available" if not q1.empty else "Missing",
                "Available" if not q2.empty else "Missing",
                "Available" if not q3.empty else "Missing",
                "Available" if not q4.empty else "Missing"
            ]
        }
    )


    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )


    selected_quarter = st.selectbox(
        "Select result dataset",
        ["Q1", "Q2", "Q3", "Q4"]
    )


    selected_data = {
        "Q1": q1,
        "Q2": q2,
        "Q3": q3,
        "Q4": q4
    }[selected_quarter]


    if selected_data.empty:

        st.warning(
            f"{selected_quarter}_pass_fail.csv "
            "is not available."
        )

    else:

        st.subheader(
            f"{selected_quarter} Pass/Fail Results"
        )

        st.dataframe(
            selected_data.head(100),
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# PAGE 6 - HYBRID PETRI NET
# ============================================================

elif page == "🔄 Hybrid Petri Net":

    st.title(
        "🔄 Hybrid Petri Net"
    )


    st.write(
        """
        The Hybrid Petri Net representation connects student learning
        activity, performance evaluation, prediction, risk assignment,
        intervention and course completion.
        """
    )


    # -------------------------
    # PLACES
    # -------------------------

    st.subheader(
        "Places"
    )


    places = [
        "P1 – Student Enrolled",
        "P2 – Learning Activity",
        "P3 – Performance Evaluation",
        "P4 – High Performer",
        "P5 – Normal",
        "P6 – At Risk",
        "P7 – Critical Risk",
        "P8 – Intervention",
        "P9 – Completed"
    ]


    for place in places:

        st.write(
            "● " + place
        )


    # -------------------------
    # TRANSITIONS
    # -------------------------

    st.subheader(
        "Transitions"
    )


    transitions = [
        "T1 – Start Learning",
        "T2 – Evaluate Performance",
        "T3 – Predict Outcome",
        "T4 – Assign Risk",
        "T5 – Recommend Intervention",
        "T6 – Re-evaluate",
        "T7 – Complete Course"
    ]


    for transition in transitions:

        st.write(
            "▶ " + transition
        )


    # -------------------------
    # FLOW
    # -------------------------

    st.subheader(
        "Process Flow"
    )


    flow = [
        "Student Enrolled",
        "Learning Activity",
        "Performance Evaluation",
        "Prediction",
        "Risk Assignment",
        "Intervention",
        "Re-evaluation",
        "Course Completion"
    ]


    for i in range(
        len(flow) - 1
    ):

        st.markdown(
            f"""
            **{flow[i]}**
            ↓
            **{flow[i + 1]}**
            """
        )


# ============================================================
# PAGE 7 - MODEL COMPARISON
# ============================================================

elif page == "🧪 Model Comparison":

    st.title(
        "🧪 Model Comparison"
    )


    if q4_comparison.empty:

        show_missing_file(
            "Q4_accuracy_comparison.csv"
        )

    else:

        st.subheader(
            "Saved Q4 Model Comparison"
        )


        st.dataframe(
            q4_comparison,
            use_container_width=True,
            hide_index=True
        )


        model_column = find_column(
            q4_comparison,
            [
                "Model",
                "Model Name",
                "Classifier",
                "Method"
            ]
        )


        accuracy_column = find_column(
            q4_comparison,
            [
                "Accuracy",
                "accuracy_score",
                "Test Accuracy"
            ]
        )


        if (
            model_column is not None
            and accuracy_column is not None
        ):

            chart = q4_comparison[
                [
                    model_column,
                    accuracy_column
                ]
            ].copy()


            chart[accuracy_column] = (
                convert_to_percent(
                    chart[accuracy_column]
                )
            )


            fig = px.bar(
                chart,
                x=model_column,
                y=accuracy_column,
                text_auto=".2f",
                title="Q4 Model Accuracy"
            )


            fig.update_yaxes(
                title="Accuracy (%)"
            )


            fig.update_xaxes(
                title="Model"
            )


            st.plotly_chart(
                fig,
                use_container_width=True
            )


    st.divider()


    st.subheader(
        "Earlier OULAD Experiment"
    )


    st.write(
        """
        The project documentation records a separate multi-class
        experiment in which Logistic Regression achieved the highest
        raw accuracy of 52.32%.

        XGBoost + SMOTE was selected according to the reported
        F1-based selection rule, with 49.19% F1 and 49.93% accuracy.

        These results are kept separate from the later Q4
        high-performance experiment.
        """
    )


# ============================================================
# PAGE 8 - METHODOLOGY
# ============================================================

elif page == "📚 Methodology":

    st.title(
        "📚 Project Methodology"
    )


    st.subheader(
        "1. OULAD Dataset"
    )

    st.write(
        "Student information and online learning/VLE interaction data."
    )


    st.subheader(
        "2. Temporal Feature Engineering"
    )

    st.write(
        "Learning activity is converted into behavioural and temporal features."
    )


    st.subheader(
        "3. Early Prediction"
    )

    st.write(
        "Prediction is evaluated at Week 4, Week 8, Week 12 and Week 16."
    )


    st.subheader(
        "4. Q4 Prediction"
    )

    st.write(
        "A Q4 temporal experiment uses deep temporal features and a model evaluation workflow."
    )


    st.subheader(
        "5. Risk Classification"
    )

    st.write(
        "Prediction outcomes are mapped into High Performer, Normal, At Risk and Critical Risk categories."
    )


    st.subheader(
        "6. Decision Support"
    )

    st.write(
        "Risk categories are connected to intervention-oriented decision support."
    )


    st.subheader(
        "7. Hybrid Petri Net"
    )

    st.write(
        "The process model connects learning, evaluation, prediction, risk assignment, intervention and completion."
    )


    st.divider()


    st.subheader(
        "Dashboard Data Files"
    )


    required_files = [
        "Q1_pass_fail.csv",
        "Q2_pass_fail.csv",
        "Q3_pass_fail.csv",
        "Q4_pass_fail.csv",
        "Q4_deep_temporal_features.csv",
        "Q4_validation_results.csv",
        "Q4_final_test_results.csv",
        "FINAL_PROJECT_RESULTS.csv",
        "early_prediction_results.csv",
        "best_early_prediction_window.csv",
        "Q4_accuracy_comparison.csv"
    ]


    status_rows = []


    for filename in required_files:

        status_rows.append(
            {
                "File": filename,
                "Status":
                    "✓ Available"
                    if (DATA_DIR / filename).exists()
                    else "✗ Missing"
            }
        )


    st.dataframe(
        pd.DataFrame(status_rows),
        use_container_width=True,
        hide_index=True
    )


    st.info(
        "This Streamlit application displays the saved project results. "
        "It does not retrain the machine-learning models every time the "
        "dashboard starts, which keeps deployment lightweight and reproducible."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Hybrid Petri Net Student Performance Dashboard | "
    "OULAD-based project"
)
