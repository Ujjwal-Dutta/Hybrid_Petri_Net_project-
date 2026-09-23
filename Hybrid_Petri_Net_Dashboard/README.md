# Hybrid Petri Net – Streamlit Dashboard

## Project Overview

This project presents a Streamlit dashboard for visualizing the results produced by the Hybrid Petri Net / early-prediction project.

The dashboard is based on the generated CSV result files and figures from `BAI_PROJECT.ipynb`.

## Project Contents

- `BAI_PROJECT.ipynb` – original project notebook.
- `data/` – generated prediction, validation, Q1–Q4, and final-result CSV files.
- `figures/` – generated accuracy-comparison CSV and PNG figures.
- `app.py` – Streamlit dashboard application (to be created).
- `requirements.txt` – Python packages required to run the dashboard.

## Dashboard Scope

The planned dashboard will present:

1. Early prediction results.
2. Best early-prediction window.
3. Q1–Q4 pass/fail results.
4. Q4 validation and final test results.
5. Final project performance.
6. Model/accuracy comparison figures.
7. Interactive visualizations and filters.

## How to Run

After `app.py` is created and the required packages are installed:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The dashboard should be run from the `Hybrid_Petri_Net_Dashboard` project folder.

## Data Note

The dashboard uses the generated project-result files stored in the `data/` and `figures/` folders. The original OULAD raw-data folder is not required for the initial results-visualization dashboard.
