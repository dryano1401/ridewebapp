"""Narrative report generation for the RIDE Streamlit application."""

from __future__ import annotations

from datetime import datetime
import math
from typing import Iterable

import pandas as pd

from ride_calculations import DoseResult, FitResult


def _fmt(value: float | None, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "not available"
    try:
        if not math.isfinite(float(value)):
            return "not available"
    except (TypeError, ValueError):
        return "not available"
    return f"{float(value):,.{digits}f}{suffix}"


def _pct_change(numerator: float | None, denominator: float | None) -> str:
    if numerator is None or denominator is None or denominator == 0:
        return "not available"
    return _fmt((numerator - denominator) / denominator * 100.0, 1, "%")


def _quality_statement(fit: FitResult, n_points: int) -> str:
    if n_points < 4:
        return (
            "The TAC contains fewer than four usable points, so the two-phase fit may be poorly constrained. "
            "Additional time points should be considered if the result will be used for formal reporting."
        )
    if math.isfinite(fit.r_squared) and fit.r_squared >= 0.95:
        return "The fitted model closely followed the entered TAC data based on the reported R²."
    if math.isfinite(fit.r_squared) and fit.r_squared >= 0.85:
        return "The fitted model showed reasonable agreement with the entered TAC data, but visual review of the plot is recommended."
    if math.isfinite(fit.r_squared):
        return "The fitted model showed limited agreement with the entered TAC data; review the input points and fitted curve before relying on the estimate."
    return "Fit quality could not be summarized by R² for this dataset. Visual review of the fitted curve is recommended."


def _dose_statement(dose: DoseResult) -> str:
    pieces: list[str] = []
    pieces.append(
        "The curve-based absorbed dose estimate was "
        f"{_fmt(dose.curve_absorbed_dose_gy, 2, ' Gy')}."
    )
    pieces.append(
        "The physical-decay-only estimate was "
        f"{_fmt(dose.physical_absorbed_dose_gy, 2, ' Gy')}, which is "
        f"{_pct_change(dose.physical_absorbed_dose_gy, dose.curve_absorbed_dose_gy)} relative to the curve-based estimate."
    )
    if dose.complete_absorbed_dose_gy is not None:
        pieces.append(
            "The complete-infiltration comparison estimate was "
            f"{_fmt(dose.complete_absorbed_dose_gy, 2, ' Gy')}."
        )
    else:
        pieces.append("The complete-infiltration comparison estimate was not available for this input combination.")
    if dose.complete_infiltration_flag:
        pieces.append(
            "The curve-corrected initial infiltrated activity exceeded the injected activity, so the curve-based activity was capped at the injected activity."
        )
    return " ".join(pieces)


def build_markdown_report(
    *,
    dose: DoseResult,
    fit: FitResult,
    tac_df: pd.DataFrame,
    source_label: str,
    app_version: str = "RIDE Streamlit",
) -> str:
    """Build a clinician-readable Markdown summary from app results."""
    n_points = len(tac_df)
    time_min = tac_df["Time_min"] if "Time_min" in tac_df else pd.Series(dtype=float)
    count_rate = tac_df["CountRate"] if "CountRate" in tac_df else pd.Series(dtype=float)
    time_range = (
        f"{_fmt(float(time_min.min()), 2)} to {_fmt(float(time_min.max()), 2)} min" if not time_min.empty else "not available"
    )
    count_range = (
        f"{_fmt(float(count_rate.min()), 4)} to {_fmt(float(count_rate.max()), 4)}" if not count_rate.empty else "not available"
    )
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    warnings: list[str] = []
    if fit.warning:
        warnings.append(fit.warning)
    if dose.warning:
        warnings.append(dose.warning)
    if dose.complete_infiltration_flag:
        warnings.append("Possible complete infiltration flag was triggered.")
    warnings_text = "\n".join(f"- {item}" for item in warnings) if warnings else "- No automated warnings were generated."

    tac_table = tac_df[["Time_min", "CountRate"]].rename(
        columns={"Time_min": "Time after imaging start (min)", "CountRate": "Relative count rate"}
    )
    tac_table_md = tac_table.to_markdown(index=False, floatfmt=".4f") if not tac_table.empty else "No TAC data available."

    report = f"""# RIDE Dosimetry Narrative Report

Generated: {now}  
Application: {app_version}

## Case summary

A radiopharmaceutical infiltration dosimetry estimate was generated for **{dose.isotope}** using an injected activity of **{_fmt(dose.injected_activity, 3)} {dose.units}** and a measured infiltration activity of **{_fmt(dose.infiltration_activity, 3)} {dose.units}** at **{_fmt(dose.uptake_time_min, 1)} minutes** after administration. The source TAC data were entered by **{source_label}** and included **{n_points} usable time points** spanning **{time_range}**. Relative count rates ranged from **{count_range}**.

## TAC model summary

The entered time-activity data were fit using a **{fit.model_name}** model. The terminal biological half-life estimated from the fitted clearance curve was **{_fmt(fit.terminal_half_life_min, 2)} minutes**. The isotope physical half-life used in the calculation was **{_fmt(dose.physical_half_life_min, 2)} minutes**, producing an effective half-life of **{_fmt(dose.effective_half_life_min, 2)} minutes**. Fit quality metrics were R² = **{_fmt(fit.r_squared, 3)}** and RMSE = **{_fmt(fit.rmse, 4)}**. {_quality_statement(fit, n_points)}

## Dose estimate summary

{_dose_statement(dose)} The curve-corrected initial infiltrated activity was **{_fmt(dose.curve_initial_activity, 3)} {dose.units}**. The effective correction factor was **{_fmt(dose.effective_correction_factor, 4)}**, and the physical-decay correction factor was **{_fmt(dose.physical_correction_factor, 4)}**.

## Automated checks and warnings

{warnings_text}

## TAC data used for fitting

{tac_table_md}

## Interpretation note

This report is an automated summary of the entered data and fitted model output. It should be reviewed alongside the TAC plot, original images/count measurements, local infiltration assessment, and institutional clinical judgment. The tool is intended for guidance, research, and quality-improvement support and is not a standalone diagnostic or treatment decision system.
"""
    return report.strip() + "\n"


def markdown_to_plain_text(markdown_report: str) -> str:
    """A small Markdown-to-text helper for downloadable TXT reports."""
    lines: list[str] = []
    for line in markdown_report.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            lines.append(stripped[2:].upper())
        elif stripped.startswith("## "):
            lines.append("\n" + stripped[3:].upper())
        elif stripped.startswith("- "):
            lines.append(stripped)
        else:
            lines.append(line.replace("**", ""))
    return "\n".join(lines).strip() + "\n"
