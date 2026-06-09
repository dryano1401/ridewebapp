"""Concise plain-text report generation for the RIDE Streamlit application."""

from __future__ import annotations

from datetime import datetime
import math

import pandas as pd

from ride_calculations import DoseResult, FitResult


def _fmt(value: float | None, digits: int = 2, suffix: str = "") -> str:
    """Format numeric values for a readable report."""
    if value is None:
        return "not available"
    try:
        numeric = float(value)
        if not math.isfinite(numeric):
            return "not available"
    except (TypeError, ValueError):
        return "not available"
    return f"{numeric:,.{digits}f}{suffix}"


def _relative_to_curve(value: float | None, curve_value: float | None) -> str:
    """Return a concise comparison against the curve-based estimate."""
    if value is None or curve_value is None or curve_value == 0:
        return "not available"
    change = (float(value) - float(curve_value)) / float(curve_value) * 100.0
    direction = "higher" if change >= 0 else "lower"
    return f"{abs(change):.1f}% {direction} than curve-based"


def _quality_statement(fit: FitResult, n_points: int) -> str:
    """Summarize fit quality without over-interpreting it."""
    if n_points < 4:
        return "limited TAC points; verify fit visually"
    if math.isfinite(fit.r_squared) and fit.r_squared >= 0.95:
        return "excellent visual/statistical agreement"
    if math.isfinite(fit.r_squared) and fit.r_squared >= 0.85:
        return "reasonable agreement; review plot"
    if math.isfinite(fit.r_squared):
        return "limited agreement; review inputs and fit"
    return "R2 unavailable; review plot"


def _warning_lines(dose: DoseResult, fit: FitResult) -> list[str]:
    warnings: list[str] = []
    if fit.warning:
        warnings.append(fit.warning)
    if dose.warning:
        warnings.append(dose.warning)
    if dose.complete_infiltration_flag:
        warnings.append("Possible complete infiltration flag was triggered.")
    return warnings


def _tac_summary(tac_df: pd.DataFrame) -> tuple[int, str, str]:
    n_points = len(tac_df)
    if tac_df.empty or "Time_min" not in tac_df or "CountRate" not in tac_df:
        return n_points, "not available", "not available"

    time_min = tac_df["Time_min"]
    count_rate = tac_df["CountRate"]
    time_range = f"{_fmt(float(time_min.min()), 2)}-{_fmt(float(time_min.max()), 2)} min"
    count_range = f"{_fmt(float(count_rate.min()), 4)}-{_fmt(float(count_rate.max()), 4)}"
    return n_points, time_range, count_range


def _tac_lines(tac_df: pd.DataFrame) -> list[str]:
    if tac_df.empty or "Time_min" not in tac_df or "CountRate" not in tac_df:
        return ["TAC data: not available"]

    lines = ["TAC points used for fitting:"]
    for _, row in tac_df[["Time_min", "CountRate"]].iterrows():
        lines.append(
            f"  - {_fmt(row['Time_min'], 2)} min: {_fmt(row['CountRate'], 4)} relative count rate"
        )
    return lines


def build_markdown_report(
    *,
    dose: DoseResult,
    fit: FitResult,
    tac_df: pd.DataFrame,
    source_label: str,
    app_version: str = "RIDE Streamlit",
) -> str:
    """Build a concise plain-text report from app results.

    The function name is retained for compatibility with the Streamlit app,
    but the returned report is intentionally plain text so it displays cleanly
    inside a Streamlit text area and can be copied directly into notes.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    n_points, time_range, count_range = _tac_summary(tac_df)
    warnings = _warning_lines(dose, fit)

    lines = [
        "RIDE DOSIMETRY REPORT",
        f"Generated: {now}",
        f"Application: {app_version}",
        "",
        "Summary",
        (
            f"{dose.isotope} infiltration estimate using injected activity "
            f"{_fmt(dose.injected_activity, 3)} {dose.units} and measured infiltration "
            f"activity {_fmt(dose.infiltration_activity, 3)} {dose.units} at "
            f"{_fmt(dose.uptake_time_min, 1)} min post-administration."
        ),
        (
            f"TAC source: {source_label}; {n_points} usable points; "
            f"time range {time_range}; relative count-rate range {count_range}."
        ),
        "",
        "Model and half-life",
        (
            f"Fit model: {fit.model_name}; terminal biological half-life "
            f"{_fmt(fit.terminal_half_life_min, 2)} min; physical half-life "
            f"{_fmt(dose.physical_half_life_min, 2)} min; effective half-life "
            f"{_fmt(dose.effective_half_life_min, 2)} min."
        ),
        (
            f"Fit quality: R2 {_fmt(fit.r_squared, 3)}, RMSE {_fmt(fit.rmse, 4)} "
            f"({_quality_statement(fit, n_points)})."
        ),
        "",
        "Dose estimates",
        f"Curve-based absorbed dose: {_fmt(dose.curve_absorbed_dose_gy, 2)} Gy.",
        (
            f"Physical-decay-only absorbed dose: {_fmt(dose.physical_absorbed_dose_gy, 2)} Gy "
            f"({_relative_to_curve(dose.physical_absorbed_dose_gy, dose.curve_absorbed_dose_gy)})."
        ),
    ]

    if dose.complete_absorbed_dose_gy is not None:
        lines.append(
            f"Complete-infiltration comparison dose: {_fmt(dose.complete_absorbed_dose_gy, 2)} Gy "
            f"({_relative_to_curve(dose.complete_absorbed_dose_gy, dose.curve_absorbed_dose_gy)})."
        )
    else:
        lines.append("Complete-infiltration comparison dose: not available.")

    lines.extend(
        [
            (
                f"Curve-corrected initial infiltrated activity: "
                f"{_fmt(dose.curve_initial_activity, 3)} {dose.units}."
            ),
            (
                f"Correction factors: effective {_fmt(dose.effective_correction_factor, 4)}, "
                f"physical {_fmt(dose.physical_correction_factor, 4)}."
            ),
            "",
            "Automated checks",
        ]
    )

    if warnings:
        lines.extend(f"- {item}" for item in warnings)
    else:
        lines.append("- No automated warnings generated.")

    lines.extend(["", *_tac_lines(tac_df), "", "Note"])
    lines.append(
        "Automated summary for guidance, research, and quality-improvement use. "
        "Review with the TAC plot, source images/counts, local assessment, and institutional clinical judgment."
    )

    return "\n".join(lines).strip() + "\n"


def markdown_to_plain_text(markdown_report: str) -> str:
    """Compatibility helper retained for existing app downloads."""
    return markdown_report.strip() + "\n"
