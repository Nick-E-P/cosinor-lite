"""Unit tests for cosinor analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from cosinor_lite.livecell_cosinor_analysis import (
    CosinorAnalysis,
    _sanitize_xy,
    constant_model,
    cosine_model_24,
    cosine_model_damped,
    cosine_model_free_period,
)

MIN_VALID_POINTS = 4
PERIOD_LOWER_BOUND = 20.0
PERIOD_UPPER_BOUND = 28.0


def test_constant_model_returns_mesor() -> None:
    x = np.linspace(0.0, 1.0, 5, dtype=np.float64)
    mesor = 1.23
    result = constant_model(x, mesor)
    np.testing.assert_allclose(result, np.full_like(x, mesor))


def test_cosine_model_implementations_match_formulas() -> None:
    x = np.linspace(0.0, 24.0, 5, dtype=np.float64)
    amplitude = 2.5
    acrophase = 3.0
    mesor = 1.1
    period = 26.0
    damp = 0.02

    direct_24 = amplitude * np.cos(2 * np.pi * (x - acrophase) / 24.0) + mesor
    direct_free = amplitude * np.cos(2 * np.pi * (x - acrophase) / period) + mesor
    direct_damped = amplitude * np.exp(-damp * x) * np.cos(2 * np.pi * (x - acrophase) / period) + mesor

    np.testing.assert_allclose(cosine_model_24(x, amplitude, acrophase, mesor), direct_24)
    np.testing.assert_allclose(cosine_model_free_period(x, amplitude, acrophase, period, mesor), direct_free)
    np.testing.assert_allclose(
        cosine_model_damped(x, amplitude, damp, acrophase, period, mesor),
        direct_damped,
    )


def test_sanitize_xy_enforces_minimum_points() -> None:
    x = np.array([0.0, 1.0, np.nan], dtype=np.float64)
    y = np.array([np.nan, 2.0, 3.0], dtype=np.float64)
    with pytest.raises(ValueError, match="Not enough valid points"):
        _sanitize_xy(x, y, min_points=3)


def _first_valid_trace(analysis: CosinorAnalysis) -> tuple[np.ndarray, np.ndarray]:
    for column in analysis.time_series.T:
        mask = np.isfinite(column)
        if np.count_nonzero(mask) >= MIN_VALID_POINTS:
            return analysis.time[mask], column[mask]
    msg = "No trace has sufficient finite observations for cosinor fitting."
    pytest.skip(msg)
    raise AssertionError


def test_fit_cosinor_24_returns_expected_structure(cosinor_analysis: CosinorAnalysis) -> None:
    x, y = _first_valid_trace(cosinor_analysis)
    results, t_test, y_test = cosinor_analysis.fit_cosinor_24(x, y)

    expected_keys = {"mesor", "amplitude", "acrophase", "p-val osc", "r2", "r2_adj"}
    assert expected_keys.issubset(results.keys())  # noqa: S101
    assert t_test.shape == y_test.shape  # noqa: S101
    assert t_test.ndim == 1  # noqa: S101
    assert np.all(np.isfinite(y_test))  # noqa: S101


def test_fit_cosinor_free_period_period_within_bounds(cosinor_analysis: CosinorAnalysis) -> None:
    x, y = _first_valid_trace(cosinor_analysis)
    results, _, _ = cosinor_analysis.fit_cosinor_free_period(x, y)
    period = results["period"]
    assert PERIOD_LOWER_BOUND <= period <= PERIOD_UPPER_BOUND  # noqa: S101


def test_fit_cosinor_damped_returns_non_negative_damp(cosinor_analysis: CosinorAnalysis) -> None:
    x, y = _first_valid_trace(cosinor_analysis)
    results, _, _ = cosinor_analysis.fit_cosinor_damped(x, y)
    damp = results["damp"]
    assert damp >= 0.0  # noqa: S101


def test_get_cosinor_fits_invalid_method_raises(cosinor_analysis: CosinorAnalysis) -> None:
    x, y = _first_valid_trace(cosinor_analysis)
    with pytest.raises(ValueError, match="Unknown cosine model"):
        cosinor_analysis.get_cosinor_fits(x, y, method="invalid")


def test_fit_cosinor_exports_results(cosinor_analysis: CosinorAnalysis) -> None:
    df_export, csv_path, fig, pdf_path = cosinor_analysis.fit_cosinor(
        group="group1",
        cosinor_model="cosinor_24",
    )

    try:
        assert not df_export.empty  # noqa: S101
        assert Path(csv_path).is_file()  # noqa: S101
        assert Path(pdf_path).is_file()  # noqa: S101
    finally:
        Path(csv_path).unlink(missing_ok=True)
        Path(pdf_path).unlink(missing_ok=True)
        plt.close(fig)
