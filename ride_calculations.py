"""Core calculations for the RIDE Streamlit application.

The original Shiny application used R's PK::biexp() to fit a two-phase
exponential clearance curve, then used the terminal biological half-life to
estimate effective half-life and absorbed dose.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

LN2 = math.log(2.0)


ISOTOPE_DATA = pd.DataFrame(
    {
        "Isotope": [
            "C11",
            "C64",
            "F18",
            "Ga67",
            "Ga68",
            "I123",
            "I131",
            "In111",
            "Lu177",
            "N13",
            "Na22",
            "O15",
            "Rb82",
            "Sm153",
            "Sr89",
            "Tc99m",
            "Y90",
        ],
        # Original Shiny values, in minutes.
        "Physical half-life (min)": [
            20.39,
            762,
            109.77,
            4696.13,
            67.71,
            796.2,
            11549.81,
            4038.77,
            9572.4,
            9.97,
            1367558.64,
            2.04,
            1.27,
            2790,
            72763,
            360.9,
            3846,
        ],
        # Original Shiny dose-rate conversion factor. The app applies /1000
        # after cumulated activity, and multiplies this column by 37 for mCi.
        "Dose-rate factor": [
            0.741,
            0.246,
            0.498,
            0.086,
            1.23,
            0.073,
            0.38,
            0.097,
            0.285,
            0.907,
            0.463,
            1.25,
            1.94,
            0.554,
            1,
            0.039,
            1.46,
        ],
    }
)


@dataclass(frozen=True)
class FitResult:
    model_name: str
    a1: float
    b1: float
    a2: float
    b2: float
    initial_half_life_min: float
    terminal_half_life_min: float
    fitted_y: np.ndarray
    r_squared: float
    rmse: float
    warning: str | None = None


@dataclass(frozen=True)
class DoseResult:
    isotope: str
    units: str
    injected_activity: float
    infiltration_activity: float
    uptake_time_min: float
    physical_half_life_min: float
    biological_terminal_half_life_min: float
    effective_half_life_min: float
    effective_correction_factor: float
    physical_correction_factor: float
    curve_initial_activity: float
    complete_infiltration_flag: bool
    curve_cumulated_activity: float
    complete_cumulated_activity: float | None
    physical_cumulated_activity: float
    curve_absorbed_dose_gy: float
    complete_absorbed_dose_gy: float | None
    physical_absorbed_dose_gy: float
    dose_rate_factor_used: float
    warning: str | None = None


def isotope_lookup(isotope: str) -> tuple[float, float]:
    row = ISOTOPE_DATA.loc[ISOTOPE_DATA["Isotope"] == isotope]
    if row.empty:
        raise ValueError(f"Unknown isotope: {isotope}")
    return float(row.iloc[0]["Physical half-life (min)"]), float(row.iloc[0]["Dose-rate factor"])


def clean_tac_dataframe(df: pd.DataFrame, time_col: str, count_col: str, time_unit: str) -> pd.DataFrame:
    """Return cleaned TAC data with time in minutes and positive count rates."""
    out = pd.DataFrame(
        {
            "Time": pd.to_numeric(df[time_col], errors="coerce"),
            "CountRate": pd.to_numeric(df[count_col], errors="coerce"),
        }
    )
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    out = out[(out["Time"] >= 0) & (out["CountRate"] > 0)].copy()
    if time_unit.lower().startswith("sec"):
        out["Time_min"] = out["Time"] / 60.0
    else:
        out["Time_min"] = out["Time"]
    out = out.sort_values("Time_min").drop_duplicates(subset=["Time_min"], keep="last")
    return out.reset_index(drop=True)


def _biexp_model(t: np.ndarray, a1: float, b1: float, a2: float, b2: float) -> np.ndarray:
    return a1 * np.exp(-b1 * t) + a2 * np.exp(-b2 * t)


def _single_exp_model(t: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * np.exp(-b * t)


def _goodness_of_fit(y: np.ndarray, y_hat: np.ndarray) -> tuple[float, float]:
    residual = y - y_hat
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(np.mean(residual**2)))
    return r2, rmse


def _starting_rate_from_log_linear(t: np.ndarray, y: np.ndarray) -> float:
    if len(t) < 2 or np.allclose(t, t[0]):
        return 1e-3
    try:
        slope, intercept = np.polyfit(t, np.log(y), 1)
        b = max(-float(slope), 1e-9)
        return b
    except Exception:
        return 1e-3


def _fit_single_exponential(
    t: np.ndarray,
    y: np.ndarray,
    *,
    base_b: float,
    y0: float,
    max_t: float,
    warning: str | None = None,
    model_name: str = "Single-exponential",
) -> FitResult:
    """Fit a single-exponential TAC model."""
    try:
        params, _ = curve_fit(
            _single_exp_model,
            t,
            y,
            p0=(y0, max(base_b, 1e-9)),
            bounds=([0.0, 1e-12], [max(y0 * 10, 1.0), 100.0 / max_t]),
            maxfev=50000,
        )
        a, b = map(float, params)
    except Exception:
        b = max(base_b, 1e-9)
        a = float(np.exp(np.mean(np.log(y) + b * t)))
        warning = "Single-exponential optimizer was unstable; used log-linear estimate." if warning is None else warning

    y_hat = _single_exp_model(t, a, b)
    r2, rmse = _goodness_of_fit(y, y_hat)
    half_life = LN2 / b
    return FitResult(
        model_name=model_name,
        a1=a,
        b1=b,
        a2=0.0,
        b2=0.0,
        initial_half_life_min=float(half_life),
        terminal_half_life_min=float(half_life),
        fitted_y=y_hat,
        r_squared=r2,
        rmse=rmse,
        warning=warning,
    )


def fit_tac_curve(
    t_min: Iterable[float],
    count_rate: Iterable[float],
    model_type: str = "Bi-exponential",
) -> FitResult:
    """Fit the selected exponential model and return a terminal half-life.

    Supported model_type values:
    - "Bi-exponential": fits y = a1*exp(-b1*t) + a2*exp(-b2*t). The slower
      component is treated as the terminal biological half-life.
    - "Single-exponential": fits y = a*exp(-b*t) and uses that half-life as the
      biological terminal half-life.

    If the bi-exponential fit is selected but unstable, a single-exponential
    fallback is used.
    """
    t = np.asarray(list(t_min), dtype=float)
    y = np.asarray(list(count_rate), dtype=float)
    mask = np.isfinite(t) & np.isfinite(y) & (t >= 0) & (y > 0)
    t = t[mask]
    y = y[mask]

    if len(t) < 2:
        raise ValueError("At least two positive TAC points are required.")

    order = np.argsort(t)
    t = t[order]
    y = y[order]

    model_normalized = model_type.strip().lower()
    if model_normalized not in {"bi-exponential", "biexponential", "single-exponential", "single exponential"}:
        raise ValueError("Model type must be 'Bi-exponential' or 'Single-exponential'.")

    base_b = _starting_rate_from_log_linear(t, y)
    y0 = max(float(np.max(y)), 1e-9)
    max_t = max(float(np.max(t)), 1.0)

    if model_normalized in {"single-exponential", "single exponential"}:
        return _fit_single_exponential(
            t,
            y,
            base_b=base_b,
            y0=y0,
            max_t=max_t,
            model_name="Single-exponential",
        )

    # Two-phase fits can be underdetermined for the default 4-point input, so use
    # several reasonable starts and select the lowest residual solution.
    starts = [
        (0.7 * y0, max(base_b * 4, 1e-9), 0.3 * y0, max(base_b * 0.5, 1e-10)),
        (0.5 * y0, max(base_b * 2, 1e-9), 0.5 * y0, max(base_b * 0.25, 1e-10)),
        (0.3 * y0, max(base_b * 6, 1e-9), 0.7 * y0, max(base_b, 1e-10)),
    ]
    best = None
    best_sse = np.inf
    last_error: str | None = None
    lower = [0.0, 1e-12, 0.0, 1e-12]
    upper = [max(y0 * 10, 1.0), 100.0 / max_t, max(y0 * 10, 1.0), 100.0 / max_t]

    if len(t) >= 4:
        for start in starts:
            try:
                params, _ = curve_fit(
                    _biexp_model,
                    t,
                    y,
                    p0=start,
                    bounds=(lower, upper),
                    maxfev=50000,
                )
                y_hat = _biexp_model(t, *params)
                sse = float(np.sum((y - y_hat) ** 2))
                if sse < best_sse and np.all(np.isfinite(params)):
                    best = params
                    best_sse = sse
            except Exception as exc:  # pragma: no cover - depends on optimizer path
                last_error = str(exc)

    if best is not None:
        a1, b1, a2, b2 = map(float, best)
        # Sort so component 1 is faster, component 2 is terminal/slower.
        components = sorted([(a1, b1), (a2, b2)], key=lambda pair: pair[1], reverse=True)
        (a_fast, b_fast), (a_slow, b_slow) = components
        y_hat = _biexp_model(t, a_fast, b_fast, a_slow, b_slow)
        r2, rmse = _goodness_of_fit(y, y_hat)
        initial_hl = LN2 / b_fast
        terminal_hl = LN2 / b_slow
        warning = None
        if np.isclose(b_fast, b_slow, rtol=0.05):
            warning = "The two fitted components were very similar; consider whether a single-exponential model is adequate."
        return FitResult(
            model_name="Bi-exponential",
            a1=a_fast,
            b1=b_fast,
            a2=a_slow,
            b2=b_slow,
            initial_half_life_min=float(initial_hl),
            terminal_half_life_min=float(terminal_hl),
            fitted_y=y_hat,
            r_squared=r2,
            rmse=rmse,
            warning=warning,
        )

    fallback_warning = "Bi-exponential fit was unstable; used single-exponential fallback."
    if last_error:
        fallback_warning += f" Optimizer note: {last_error}"
    return _fit_single_exponential(
        t,
        y,
        base_b=base_b,
        y0=y0,
        max_t=max_t,
        warning=fallback_warning,
        model_name="Single-exponential fallback",
    )


def calculate_dose(
    *,
    isotope: str,
    units: str,
    injected_activity: float,
    infiltration_activity: float,
    uptake_time_min: float,
    biological_terminal_half_life_min: float,
) -> DoseResult:
    """Calculate absorbed-dose estimates from the original Shiny logic."""
    if injected_activity <= 0:
        raise ValueError("Injected activity must be greater than zero.")
    if infiltration_activity <= 0:
        raise ValueError("Infiltration activity must be greater than zero.")
    if uptake_time_min <= 0:
        raise ValueError("Uptake time must be greater than zero.")
    if biological_terminal_half_life_min <= 0:
        raise ValueError("Biological terminal half-life must be greater than zero.")

    physical_hl_min, dose_rate_factor = isotope_lookup(isotope)
    if units == "mCi":
        dose_rate_factor *= 37.0

    effective_hl = (biological_terminal_half_life_min * physical_hl_min) / (
        biological_terminal_half_life_min + physical_hl_min
    )
    eff_cf = math.exp(-LN2 / effective_hl * uptake_time_min)
    phys_cf = math.exp(-LN2 / physical_hl_min * uptake_time_min)
    curve_initial_activity = infiltration_activity / eff_cf
    complete_flag = curve_initial_activity > injected_activity

    physical_cumulated = (infiltration_activity / phys_cf) * physical_hl_min * 1.44

    warning = None
    if complete_flag:
        curve_cumulated = injected_activity * effective_hl * 1.44
        warning = (
            "Curve-corrected infiltrated activity exceeds the injected activity; "
            "the curve-based estimate has been capped at the injected activity."
        )
    else:
        curve_cumulated = curve_initial_activity * effective_hl * 1.44

    complete_cumulated: float | None
    if injected_activity > infiltration_activity:
        comp_cum_decay_constant = math.log(injected_activity / infiltration_activity) / uptake_time_min
        complete_cumulated = injected_activity * (LN2 / comp_cum_decay_constant) * 1.44
    else:
        complete_cumulated = None
        extra = (
            " Complete-infiltration comparison cannot be calculated because infiltration activity "
            "is greater than or equal to injected activity."
        )
        warning = extra if warning is None else warning + extra

    curve_abs_dose = curve_cumulated * dose_rate_factor / 1000.0
    complete_abs_dose = None if complete_cumulated is None else complete_cumulated * dose_rate_factor / 1000.0
    physical_abs_dose = physical_cumulated * dose_rate_factor / 1000.0

    return DoseResult(
        isotope=isotope,
        units=units,
        injected_activity=float(injected_activity),
        infiltration_activity=float(infiltration_activity),
        uptake_time_min=float(uptake_time_min),
        physical_half_life_min=float(physical_hl_min),
        biological_terminal_half_life_min=float(biological_terminal_half_life_min),
        effective_half_life_min=float(effective_hl),
        effective_correction_factor=float(eff_cf),
        physical_correction_factor=float(phys_cf),
        curve_initial_activity=float(curve_initial_activity),
        complete_infiltration_flag=bool(complete_flag),
        curve_cumulated_activity=float(curve_cumulated),
        complete_cumulated_activity=None if complete_cumulated is None else float(complete_cumulated),
        physical_cumulated_activity=float(physical_cumulated),
        curve_absorbed_dose_gy=float(curve_abs_dose),
        complete_absorbed_dose_gy=None if complete_abs_dose is None else float(complete_abs_dose),
        physical_absorbed_dose_gy=float(physical_abs_dose),
        dose_rate_factor_used=float(dose_rate_factor),
        warning=warning,
    )


def dose_result_to_dataframe(result: DoseResult) -> pd.DataFrame:
    rows = [
        ("Curve-based absorbed dose", result.curve_absorbed_dose_gy, "Gy"),
        ("Complete-infiltration comparison", result.complete_absorbed_dose_gy, "Gy"),
        ("Physical-decay-only absorbed dose", result.physical_absorbed_dose_gy, "Gy"),
        ("Biological terminal half-life", result.biological_terminal_half_life_min, "min"),
        ("Physical half-life", result.physical_half_life_min, "min"),
        ("Effective half-life", result.effective_half_life_min, "min"),
        ("Curve-corrected initial infiltrated activity", result.curve_initial_activity, result.units),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value", "Units"])
