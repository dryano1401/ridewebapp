# RIDE Streamlit App

Radiopharmaceutical Infiltration Dosimetry Estimator (RIDE), ported from the original R Shiny application to Streamlit for GitHub-based hosting.

## What changed from the Shiny version

- Rebuilt as a Python/Streamlit app with a single GitHub-ready entry point: `streamlit_app.py`.
- Preserved the original isotope table, activity-unit handling, uptake-time correction, and three dose estimates.
- Recreated the `PK::biexp` behavior using a two-phase exponential model: `y = a1*exp(-b1*t) + a2*exp(-b2*t)`.
- Added robust input validation and a single-exponential fallback when a two-phase fit is unstable.
- Added an interactive Plotly TAC fit, downloadable CSV output, clearer warnings, and an editable/pasteable table.
- Added CSV/TSV upload with selectable time/count-rate columns.
- Added an automated narrative report that summarizes the case inputs, TAC fit, dose estimates, model quality, warnings, and TAC data used for fitting. Reports can be copied in-app or downloaded as Markdown/TXT.

## Files

```text
streamlit_app.py          # Main Streamlit app
ride_calculations.py      # Calculation and model-fitting functions
ride_report.py            # Textual narrative report generator
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

## Narrative report output

After selecting **Calculate results**, the app now creates a textual report with:

- Case inputs and TAC source
- TAC model and fit quality summary
- Curve-based, complete-infiltration comparison, and physical-decay-only dose estimates
- Automated warnings and edge-case notes
- The TAC data points used for model fitting

The report appears in a copyable text area and can be downloaded as `.md` or `.txt`. The structured results CSV remains available as a separate download.

## Clinical-use note

This tool is intended for guidance, research, and quality-improvement support only. It should not be used as a standalone diagnostic or treatment decision tool without local clinical review, institutional validation, and appropriate oversight.
