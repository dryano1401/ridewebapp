# RIDE Streamlit App

Radiopharmaceutical Infiltration Dosimetry Estimator (RIDE), ported from the original R Shiny application to Streamlit for GitHub-based hosting.

## What changed from the Shiny version

- Rebuilt as a Python/Streamlit app with a single GitHub-ready entry point: `streamlit_app.py`.
- Preserved the original isotope table, activity-unit handling, uptake-time correction, and three dose estimates.
- Recreated the `PK::biexp` behavior using a two-phase exponential model: `y = a1*exp(-b1*t) + a2*exp(-b2*t)`.
- Added robust input validation and a single-exponential fallback when a two-phase fit is unstable.
- Added an interactive Plotly TAC fit, downloadable CSV output, clearer warnings, and an editable/pasteable table.
- Added CSV/TSV upload with selectable time/count-rate columns.

## Files

```text
streamlit_app.py          # Main Streamlit app
ride_calculations.py      # Calculation and model-fitting functions
requirements.txt          # Python dependencies for Streamlit Community Cloud
.streamlit/config.toml    # Theme/config settings
tests/test_calculations.py# Lightweight calculation smoke tests
```

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy with GitHub + Streamlit Community Cloud

1. Create a new GitHub repository.
2. Upload these files to the repository root.
3. Go to Streamlit Community Cloud and create a new app from the repository.
4. Use `streamlit_app.py` as the app entry point.

## Clinical-use note

This tool is intended for guidance, research, and quality-improvement support only. It should not be used as a standalone diagnostic or treatment decision tool without local clinical review, institutional validation, and appropriate oversight.
