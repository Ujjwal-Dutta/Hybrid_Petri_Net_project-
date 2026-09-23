# 🎓 Hybrid Petri Net Framework for Student Performance Prediction

## 📌 Project Overview

This project presents an **OULAD-based Hybrid Petri Net Decision Support System (DSS)** for student performance prediction and early warning.

The system combines:

* 🎓 Student information
* 💻 Virtual Learning Environment (VLE) activity
* ⏱️ Temporal and behavioral feature engineering
* 🤖 Machine Learning classification
* ⚖️ SMOTE-based class balancing
* 🔬 SHAP-based explainability
* 📈 Early prediction at Week 4, Week 8, Week 12 and Week 16
* ⚠️ Student risk classification
* 💡 Intervention-oriented decision support
* 🔄 Hybrid Petri Net process modelling

The project extends the conventional student-performance prediction approach toward a **time-aware, behavior-aware Decision Support System**.

---

## 🎯 Objectives

The main objectives of this project are:

1. 📊 Analyze student academic and online-learning behavior.
2. ⏱️ Generate temporal behavioral features from VLE activity.
3. 🔮 Evaluate student-performance prediction at multiple course stages.
4. 🤖 Compare different machine-learning models.
5. ⚖️ Address class imbalance using SMOTE.
6. 🔬 Provide model explainability using SHAP.
7. ⚠️ Categorize students according to predicted risk.
8. 💡 Connect predictions with decision/intervention support.
9. 🔄 Represent the decision workflow using a Hybrid Petri Net.
10. 🌐 Provide an interactive Streamlit dashboard for project results.

---

## 📚 Dataset

### Open University Learning Analytics Dataset (OULAD)

The project uses the **Open University Learning Analytics Dataset (OULAD)**.

The dataset combines:

* 👨‍🎓 Student information
* 📚 Course information
* 📝 Assessment information
* 💻 Virtual Learning Environment (VLE) interactions
* 📅 Registration information
* 📈 Student learning activity

The project uses behavioral information from online learning activity to construct time-aware features.

---

## 🧠 Methodology

The overall workflow is:

```text
🎓 OULAD Dataset
        ↓
👨‍🎓 Student Information
        +
💻 VLE Interaction Data
        ↓
🧹 Data Preprocessing
        ↓
📊 Behavioral Feature Engineering
        ↓
⏱️ Temporal Feature Windows
        ↓
📅 Week 4 / Week 8 / Week 12 / Week 16
        ↓
🤖 Machine Learning Models
        ↓
⚖️ SMOTE / Ensemble Methods
        ↓
🔬 SHAP Explainability
        ↓
⚠️ Risk Classification
        ↓
💡 Decision Support
        ↓
🔄 Hybrid Petri Net
```

---

## 📈 Early Prediction

The project evaluates early student-performance prediction using behavioral information available at different stages of the course.

| Prediction Window | Accuracy | Precision | Recall | Weighted F1 |
| ----------------- | -------: | --------: | -----: | ----------: |
| 📅 Week 4         |   49.09% |    46.72% | 49.09% |      43.50% |
| 📅 Week 8         |   52.07% |    49.53% | 52.07% |      47.39% |
| 📅 Week 12        |   54.53% |    52.76% | 54.53% |      49.55% |
| 📅 Week 16        |   56.82% |    54.42% | 56.82% |      52.42% |

The Week-16 experiment records the highest measured performance among these four early-prediction windows.

> These values are evaluation results on the corresponding feature windows and should not be interpreted as guaranteed real-world prediction accuracy.

---

## 🤖 Machine Learning

The project evaluates multiple machine-learning approaches, including:

* Logistic Regression
* Decision Tree
* Support Vector Machine (SVM)
* Naive Bayes
* SMOTE-enhanced models
* Hybrid Stacking + SMOTE
* XGBoost + SMOTE

The main notebook experiment reported:

| Model                   | Accuracy | Precision | Recall |     F1 |
| ----------------------- | -------: | --------: | -----: | -----: |
| XGBoost + SMOTE         |   49.93% |    50.87% | 49.93% | 49.19% |
| SVM                     |   51.50% |    48.91% | 51.50% | 48.63% |
| Hybrid Stacking + SMOTE |   47.43% |    48.51% | 47.43% | 47.68% |
| Logistic Regression     |   52.32% |    51.56% | 52.32% | 45.99% |

The project documentation selects **XGBoost + SMOTE according to the F1-based selection criterion**, while Logistic Regression has the highest raw accuracy in that particular multi-class experiment.

---

## 🔬 Explainability

The project includes **SHAP-based explainability** to investigate how model features contribute to predictions.

Behavioral features include measures such as:

* 🖱️ Total clicks
* 📅 Active days
* 📆 Active weeks
* 🌐 Unique VLE sites
* 📊 Average clicks
* 📈 Maximum clicks
* ⏱️ Recent activity
* 🔄 Interaction density

---

## ⚠️ Risk Classification & DSS

The project extends prediction into a Decision Support System.

A conceptual mapping is used between predicted outcomes, risk levels and intervention-oriented actions.

| Outcome        | Risk Category  | DSS Action                            |
| -------------- | -------------- | ------------------------------------- |
| 🏆 Distinction | High Performer | High-performer support                |
| ✅ Pass         | Normal         | Continue monitoring                   |
| ⚠️ Fail        | At Risk        | Academic intervention                 |
| 🚨 Withdrawn   | Critical Risk  | Retention and engagement intervention |

The DSS layer is intended as a decision-support workflow rather than proof of intervention effectiveness in a real institution.

---

## 🔄 Hybrid Petri Net

The Hybrid Petri Net represents the student-performance decision workflow.

### Places

* 🟢 Student Enrolled
* 📚 Learning Activity
* 📊 Performance Evaluation
* 🏆 High Performer
* 🟡 Normal
* 🟠 At Risk
* 🔴 Critical Risk
* 💡 Intervention
* ✅ Completed

### Transitions

* ▶️ Start Learning
* ▶️ Evaluate Performance
* ▶️ Predict Outcome
* ▶️ Assign Risk
* ▶️ Recommend Intervention
* ▶️ Re-evaluate
* ▶️ Complete Course

### Process

```text
Student Enrolled
       ↓
Learning Activity
       ↓
Performance Evaluation
       ↓
Prediction
       ↓
Risk Assignment
       ↓
┌───────────────┬───────────────┐
↓               ↓               ↓
High            Normal          At Risk
Performer                       / Critical
                                ↓
                           Intervention
                                ↓
                           Re-evaluation
                                ↓
                         Learning Activity
                                ↓
                         Course Completion
```

---

## 📊 Q4 High-Performance Dashboard Experiment

The Streamlit dashboard also contains the saved Q4 high-performance experiment.

### Reported Q4 Results

| Metric       |     Result |
| ------------ | ---------: |
| 🎯 Accuracy  | **89.53%** |
| 🎯 Precision | **88.69%** |
| 🎯 Recall    | **97.11%** |
| 🎯 F1 Score  | **92.71%** |

The dashboard presents this as a **separate Q4 experimental result** and does not merge it with the earlier multi-class model-comparison results.

---

## 🖥️ Streamlit Dashboard

The dashboard provides the following sections:

### 🏠 Overview

Project summary, major metrics and workflow.

### 📊 Q4 Final Results

Displays the saved Q4 test results and comparison figures.

### 📈 Early Prediction

Displays Week 4, Week 8, Week 12 and Week 16 performance.

### 👨‍🎓 Student Analysis

Explores the saved deep temporal student features.

### ⚠️ Risk & DSS

Displays risk categories and decision-support mappings.

### 🔄 Hybrid Petri Net

Displays the conceptual process model.

### 🧪 Model Comparison

Displays saved model-comparison results.

### 📚 Methodology

Explains the complete project workflow and available project files.

---

## 📁 Project Structure

```text
Hybrid_Petri_Net/
│
├── 📁 data/
│   ├── best_early_prediction_window.csv
│   ├── early_prediction_results.csv
│   ├── FINAL_PROJECT_RESULTS.csv
│   ├── Q1_pass_fail.csv
│   ├── Q2_pass_fail.csv
│   ├── Q3_pass_fail.csv
│   ├── Q4_deep_temporal_features.csv
│   ├── Q4_final_test_results.csv
│   ├── Q4_pass_fail.csv
│   └── Q4_validation_results.csv
│
├── 📁 figures/
│   ├── Q4_accuracy_comparison.csv
│   ├── Q4_base_paper_RF_vs_proposed_HGB_GB_ET_accuracy.png
│   ├── Q4_base_paper_vs_proposed_accuracy.png
│   └── Q4_base_paper_vs_proposed_deep_temporal_model.png
│
├── 📓 BAI_PROJECT.ipynb
├── 🐍 app.py
├── 📖 README.md
└── 📦 requirements.txt
```

---

## ⚙️ Requirements

The dashboard requires:

```text
streamlit
pandas
numpy
plotly
matplotlib
scikit-learn
```

---

## ▶️ Run the Dashboard Locally

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — Start Streamlit

```bash
streamlit run app.py
```

### Step 3 — Open the dashboard

Streamlit will provide a local address, normally:

```text
http://localhost:8501
```

---

## ☁️ Streamlit Community Cloud Deployment

The project can be deployed through **Streamlit Community Cloud** using the GitHub repository.

The repository should contain:

```text
app.py
requirements.txt
README.md
data/
figures/
```

The Streamlit application reads the project data using relative paths, so the dashboard does not depend on the original Google Drive or Windows/OneDrive location.

---

## 🔗 Live Dashboard

 Streamlit URL here:

```text
https://hybrid-petri-net.streamlit.app/
```

---

## 🛠️ Technologies Used

* 🐍 Python
* 📊 Pandas
* 🔢 NumPy
* 🤖 Scikit-learn
* 📈 Plotly
* 🎈 Streamlit
* ⚖️ SMOTE
* 🔬 SHAP
* 🔄 Hybrid Petri Net
* 🎓 OULAD

---

## 🔬 Research Contribution

The project is positioned as a **methodological extension** of conventional student-performance prediction.

Its main contribution is the integration of:

```text
Behavioral Data
      +
Temporal Prediction
      +
Machine Learning
      +
Explainability
      +
Risk Classification
      +
Decision Support
      +
Hybrid Petri Net
```

Rather than treating accuracy as the only contribution, the project focuses on creating an integrated **early-warning and decision-support workflow**.

---

## ⚠️ Limitations

* 📚 OULAD is a historical dataset.
* ⏱️ The current implementation is not a live real-time deployment.
* 🔄 The Hybrid Petri Net represents a simulated/process DSS workflow.
* 📊 Model performance depends on the selected dataset, target and experimental setup.
* 🧪 Further validation and robustness testing can strengthen the predictive stage.

---

## 👨‍💻 Author

**Ujjwal Kumar Dutta**

M.Tech Student
Interests: Artificial Intelligence, Data and Practical Technology Solutions

---

## 📌 Project Status

**✅ Data Processing Completed**
**✅ Feature Engineering Completed**
**✅ Early Prediction Completed**
**✅ Model Evaluation Completed**
**✅ Risk & DSS Layer Completed**
**✅ Hybrid Petri Net Workflow Completed**
**✅ Streamlit Dashboard Prepared**
**🚀 Deployment Ready**
