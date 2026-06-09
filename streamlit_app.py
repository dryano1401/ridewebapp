from __future__ import annotations

from io import StringIO
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ride_calculations import (
    ISOTOPE_DATA,
    calculate_dose,
    clean_tac_dataframe,
    dose_result_to_dataframe,
    fit_tac_curve,
    _biexp_model,
    _single_exp_model,
)
from ride_report import build_markdown_report, markdown_to_plain_text


DISPLAY_DIGITS = 2


st.set_page_config(
    page_title="RIDE | Radiopharmaceutical Infiltration Dosimetry Estimator",
    page_icon="☢️",
    layout="wide",
)


DEFAULT_TAC = pd.DataFrame(
    {
        "Time": [60.00, 1200.00, 2400.00, 4800.00],
        "CountRate": [0.99, 0.88, 0.78, 0.60],
    }
)


def init_state() -> None:
    if "tac_editor" not in st.session_state:
        st.session_state.tac_editor = DEFAULT_TAC.copy()
    if "last_report_md" not in st.session_state:
        st.session_state.last_report_md = ""
    if "last_results_csv" not in st.session_state:
        st.session_state.last_results_csv = b""


def reset_app() -> None:
    st.session_state.tac_editor = DEFAULT_TAC.copy()
    for key in ["uploaded_df", "tac_editor_widget"]:
        st.session_state.pop(key, None)
    st.session_state.last_report_md = ""
    st.session_state.last_results_csv = b""


def read_uploaded_table(uploaded_file, sep: str) -> pd.DataFrame:
    if uploaded_file is None:
        return pd.DataFrame()
    return pd.read_csv(uploaded_file, sep=sep)


def format_float(value: float | None, digits: int = DISPLAY_DIGITS) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    return f"{value:,.{digits}f}"


def build_fit_plot(clean_df: pd.DataFrame, fit_result) -> go.Figure:
    t = clean_df["Time_min"].to_numpy(dtype=float)
    y = clean_df["CountRate"].to_numpy(dtype=float)
    t_grid = np.linspace(0, max(float(t.max()), 1.0), 250)
    if fit_result.model_name.startswith("Bi"):
        y_grid = _biexp_model(t_grid, fit_result.a1, fit_result.b1, fit_result.a2, fit_result.b2)
    else:
        y_grid = _single_exp_model(t_grid, fit_result.a1, fit_result.b1)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t,
            y=y,
            mode="markers",
            name="Observed TAC",
            marker={"size": 10},
            hovertemplate="Time: %{x:.2f} min<br>Count rate: %{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t_grid,
            y=y_grid,
            mode="lines",
            name=f"{fit_result.model_name} fit",
            hovertemplate="Time: %{x:.2f} min<br>Predicted: %{y:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis_title="Time after measurement start (min)",
        yaxis_title="Relative count rate",
        hovermode="x unified",
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
    )
    fig.update_xaxes(tickformat=".2f")
    fig.update_yaxes(tickformat=".2f")
    return fig


def render_report_downloads(report_md: str, results_csv: bytes) -> None:
    st.markdown("### Narrative report")
    st.write(
        "The report converts the fitted TAC and dose calculations into a concise textual summary that can be copied into a note, QA record, or research worksheet."
    )
    st.text_area("Generated report", report_md, height=420)
    d1, d2 = st.columns(2)
    d1.download_button(
        "Download report TXT",
        data=markdown_to_plain_text(report_md).encode("utf-8"),
        file_name="ride_dosimetry_report.txt",
        mime="text/plain",
        use_container_width=True,
    )
    if results_csv:
        d2.download_button(
            "Download results CSV",
            data=results_csv,
            file_name="ride_dosimetry_results.csv",
            mime="text/csv",
            use_container_width=True,
        )


init_state()

st.title("Radiopharmaceutical Infiltration Dosimetry Estimator (RIDE)")
st.caption("Streamlit port of the original Shiny RIDE application with improved validation, interactive plots, selectable TAC fitting, downloadable results, and GitHub-ready deployment files.")

st.warning(
    "For guidance and research/quality-improvement support only. This tool is not intended to independently diagnose, treat, or replace local clinical review."
)

with st.sidebar:
    st.header("Case inputs")
    isotope = st.selectbox("Isotope", ISOTOPE_DATA["Isotope"].tolist(), index=2)
    units = st.radio("Activity units", ["MBq", "mCi"], horizontal=True)
    injected_activity = st.number_input(
        "Injected activity",
        min_value=0.0,
        value=370.00 if units == "MBq" else 10.00,
        step=1.00,
        format="%.2f",
    )
    infiltration_activity = st.number_input(
        "Measured infiltration activity",
        min_value=0.0,
        value=37.00 if units == "MBq" else 1.00,
        step=0.10,
        format="%.2f",
    )
    uptake_time_min = st.number_input(
        "Uptake time between injection and image (min)",
        min_value=0.0,
        value=60.00,
        step=1.00,
        format="%.2f",
    )
    tac_time_unit = st.radio("TAC time units", ["seconds", "minutes"], horizontal=True)
    fit_model = st.radio(
        "TAC fit model",
        ["Bi-exponential", "Single-exponential"],
        index=0,
        help="Bi-exponential matches the original RIDE approach. Single-exponential directly fits one clearance component.",
    )

    st.divider()
    data_source = st.radio("TAC data source", ["Edit/paste table", "Upload CSV/TSV"], horizontal=False)
    calculate = st.button("Calculate results", type="primary", use_container_width=True)
    show_prior_report = st.checkbox("Keep last report visible", value=True, help="Keeps the most recent narrative report available after download button clicks or minor UI refreshes.")
    st.button("Reset example data", use_container_width=True, on_click=reset_app)

left, right = st.columns([0.42, 0.58], vertical_alignment="top")

with left:
    st.subheader("TAC data")
    st.write("Enter two columns: time and relative count rate. Count rates must be positive.")

    raw_df = pd.DataFrame()
    time_col = "Time"
    count_col = "CountRate"
    sep = ","

    if data_source == "Edit/paste table":
        raw_df = st.data_editor(
            st.session_state.tac_editor,
            num_rows="dynamic",
            use_container_width=True,
            key="tac_editor_widget",
            column_config={
                "Time": st.column_config.NumberColumn("Time", help="Seconds or minutes based on the sidebar selection.", format="%.2f"),
                "CountRate": st.column_config.NumberColumn("Count rate", min_value=0.0, format="%.2f"),
            },
        )
        st.session_state.tac_editor = raw_df
        time_col, count_col = "Time", "CountRate"
    else:
        uploaded_file = st.file_uploader("Upload TAC file", type=["csv", "tsv", "txt"])
        sep_label = st.selectbox("Delimiter", ["Comma", "Tab", "Semicolon"], index=0)
        sep = {"Comma": ",", "Tab": "\t", "Semicolon": ";"}[sep_label]
        try:
            raw_df = read_uploaded_table(uploaded_file, sep)
        except Exception as exc:
            st.error(f"Could not read the uploaded file: {exc}")
            raw_df = pd.DataFrame()

        if not raw_df.empty:
            st.dataframe(raw_df.head(20), use_container_width=True)
            cols = list(raw_df.columns)
            time_col = st.selectbox("Time column", cols, index=0)
            count_col = st.selectbox("Count-rate column", cols, index=1 if len(cols) > 1 else 0)
        else:
            st.info("Upload a CSV/TSV file or switch to the editable table.")

    with st.expander("Isotope conversion table"):
        isotope_display = ISOTOPE_DATA.copy()
        numeric_cols = isotope_display.select_dtypes(include="number").columns
        isotope_display[numeric_cols] = isotope_display[numeric_cols].round(DISPLAY_DIGITS)
        st.dataframe(isotope_display, use_container_width=True, hide_index=True)

with right:
    st.subheader("Curve fit and dose summary")

    clean_df = pd.DataFrame()
    fit_result = None
    dose_result = None

    try:
        if not raw_df.empty:
            clean_df = clean_tac_dataframe(raw_df, time_col, count_col, tac_time_unit)
        if clean_df.empty:
            st.info("Enter TAC data to view the fit and results.")
        else:
            fit_result = fit_tac_curve(clean_df["Time_min"], clean_df["CountRate"], model_type=fit_model)
            st.plotly_chart(build_fit_plot(clean_df, fit_result), use_container_width=True)

            fit_cols = st.columns(4)
            fit_cols[0].metric("Model", fit_result.model_name)
            fit_cols[1].metric("Terminal biological HL", f"{fit_result.terminal_half_life_min:,.2f} min")
            fit_cols[2].metric("R²", f"{fit_result.r_squared:.2f}" if math.isfinite(fit_result.r_squared) else "—")
            fit_cols[3].metric("RMSE", f"{fit_result.rmse:.2f}")

            if fit_result.warning:
                st.info(fit_result.warning)

            if calculate:
                dose_result = calculate_dose(
                    isotope=isotope,
                    units=units,
                    injected_activity=injected_activity,
                    infiltration_activity=infiltration_activity,
                    uptake_time_min=uptake_time_min,
                    biological_terminal_half_life_min=fit_result.terminal_half_life_min,
                )

                if dose_result.warning:
                    st.warning(dose_result.warning)
                if dose_result.complete_infiltration_flag:
                    st.error("Possible complete infiltration: curve-corrected activity exceeded injected activity.")

                r1, r2, r3 = st.columns(3)
                r1.metric("Curve-based dose", f"{dose_result.curve_absorbed_dose_gy:,.2f} Gy")
                r2.metric("Complete-infiltration comparison", f"{format_float(dose_result.complete_absorbed_dose_gy)} Gy")
                r3.metric("Physical-decay-only dose", f"{dose_result.physical_absorbed_dose_gy:,.2f} Gy")

                results_df = dose_result_to_dataframe(dose_result)
                st.dataframe(
                    results_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={"Value": st.column_config.NumberColumn(format="%.2f")},
                )

                details = pd.DataFrame(
                    {
                        "Parameter": [
                            "Isotope",
                            "Units",
                            "Fit model",
                            "Injected activity",
                            "Measured infiltration activity",
                            "Uptake time",
                            "Dose-rate factor used",
                            "Effective correction factor",
                            "Physical correction factor",
                        ],
                        "Value": [
                            dose_result.isotope,
                            dose_result.units,
                            fit_result.model_name,
                            dose_result.injected_activity,
                            dose_result.infiltration_activity,
                            dose_result.uptake_time_min,
                            dose_result.dose_rate_factor_used,
                            dose_result.effective_correction_factor,
                            dose_result.physical_correction_factor,
                        ],
                    }
                )

                export = pd.concat(
                    [
                        results_df.assign(Section="Results"),
                        details.rename(columns={"Parameter": "Metric"}).assign(Units="", Section="Inputs"),
                    ],
                    ignore_index=True,
                    sort=False,
                )
                csv_bytes = export.to_csv(index=False, float_format="%.2f").encode("utf-8")
                source_label = "manual table entry" if data_source == "Edit/paste table" else "uploaded file"
                report_md = build_markdown_report(
                    dose=dose_result,
                    fit=fit_result,
                    tac_df=clean_df,
                    source_label=source_label,
                )
                st.session_state.last_report_md = report_md
                st.session_state.last_results_csv = csv_bytes
                render_report_downloads(report_md, csv_bytes)
            else:
                st.info("Review the fit, then select **Calculate results** in the sidebar.")
                if show_prior_report and st.session_state.last_report_md:
                    render_report_downloads(st.session_state.last_report_md, st.session_state.last_results_csv)
    except Exception as exc:
        st.error(str(exc))

st.divider()
with st.expander("Calculation notes"):
    st.markdown(
        """
        The selected TAC model is used to estimate the biological half-life for the dose calculation.

        Bi-exponential model: `y = a1*exp(-b1*t) + a2*exp(-b2*t)`. The slower fitted component is used as the terminal biological half-life. If this fit is unstable, the app uses a single-exponential fallback.

        Single-exponential model: `y = a*exp(-b*t)`. The fitted half-life is used directly as the biological terminal half-life.

        Effective half-life is calculated as:
        `T_eff = (T_bio × T_phys) / (T_bio + T_phys)`.

        The app reports three estimates: curve-based, complete-infiltration comparison, and physical-decay-only. Inputs and outputs retain the same activity units selected in the sidebar, while absorbed dose is reported in Gy.
        """
    )
