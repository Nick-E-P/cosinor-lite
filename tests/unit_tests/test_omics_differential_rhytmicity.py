from __future__ import annotations

from collections.abc import Iterable
from typing import TypeVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from cosinor_lite.omics_dataset import OmicsDataset
from cosinor_lite.omics_differential_rhytmicity import (
    OmicsHeatmap,
    TimeSeriesExample,
    W,
    _weight_mapping,
    akaike_weights_from_bics,
    amp_from_ab,
    bic,
    build_design,
    build_design_cond1,
    build_design_cond2,
    phase_from_ab,
)

_T = TypeVar("_T")


def test_phase_from_ab_returns_expected_hours() -> None:
    result_zero = phase_from_ab(1.0, 0.0)
    result_quarter = phase_from_ab(0.0, 1.0)

    assert result_zero == pytest.approx(0.0)  # noqa: S101
    assert result_quarter == pytest.approx(6.0)  # noqa: S101


def test_amp_from_ab_matches_euclidean_norm() -> None:
    result = amp_from_ab(3.0, 4.0)

    assert result == pytest.approx(5.0)  # noqa: S101


def test_bic_prefers_lower_log_likelihood_penalized() -> None:
    base = bic(llf=-50.0, k=4, n=100)
    more_params = bic(llf=-49.0, k=10, n=100)

    assert base < more_params  # noqa: S101


def test_akaike_weights_are_normalized_and_ranked() -> None:
    weights = akaike_weights_from_bics(np.array([100.0, 102.0, 105.0]))

    assert weights.sum() == pytest.approx(1.0)  # noqa: S101
    assert weights[0] > weights[1] > weights[2]  # noqa: S101


def test_weight_mapping_populates_model_columns() -> None:
    mapping = _weight_mapping(model_ids=(2, 4), weights=np.array([0.7, 0.2]))

    assert mapping["w_model2"] == pytest.approx(0.7)  # noqa: S101
    assert mapping["w_model4"] == pytest.approx(0.2)  # noqa: S101
    assert np.isnan(mapping["w_model1"])  # noqa: S101


def test_build_design_includes_all_expected_columns(omics_dataset: OmicsDataset) -> None:
    row = omics_dataset.df.iloc[0]
    alpha_vals = row[omics_dataset.columns_cond1].to_numpy(float)
    beta_vals = row[omics_dataset.columns_cond2].to_numpy(float)

    design = build_design(alpha_vals, beta_vals, omics_dataset.t_cond1, omics_dataset.t_cond2)

    expected_columns = {
        "y",
        "time",
        "dataset",
        "constant",
        "cos_wt",
        "sin_wt",
        "is_alpha",
        "is_beta",
    }

    assert set(design.columns) == expected_columns  # noqa: S101
    assert set(design["dataset"]) == {"alpha", "beta"}  # noqa: S101

    alpha_mask = design["dataset"] == "alpha"
    beta_mask = design["dataset"] == "beta"

    assert np.allclose(  # noqa: S101
        np.cos(W * design.loc[alpha_mask, "time"].to_numpy(float)),
        design.loc[alpha_mask, "cos_wt"].to_numpy(float),
    )
    assert np.allclose(  # noqa: S101
        np.sin(W * design.loc[beta_mask, "time"].to_numpy(float)),
        design.loc[beta_mask, "sin_wt"].to_numpy(float),
    )


def test_build_design_cond1_only_contains_alpha_rows(omics_dataset: OmicsDataset) -> None:
    row = omics_dataset.df.iloc[0]
    alpha_vals = row[omics_dataset.columns_cond1].to_numpy(float)

    design = build_design_cond1(alpha_vals, omics_dataset.t_cond1)

    assert set(design["dataset"]) == {"alpha"}  # noqa: S101
    assert np.allclose(  # noqa: S101
        np.cos(W * design["time"].to_numpy(float)),
        design["cos_wt"].to_numpy(float),
    )


def test_build_design_cond2_only_contains_beta_rows(omics_dataset: OmicsDataset) -> None:
    row = omics_dataset.df.iloc[0]
    beta_vals = row[omics_dataset.columns_cond2].to_numpy(float)

    design = build_design_cond2(beta_vals, omics_dataset.t_cond2)

    assert set(design["dataset"]) == {"beta"}  # noqa: S101
    assert np.allclose(  # noqa: S101
        np.sin(W * design["time"].to_numpy(float)),
        design["sin_wt"].to_numpy(float),
    )


class _ProgressStub:
    def tqdm(
        self,
        iterable: Iterable[_T],
        total: int | None = None,
        desc: str | None = None,
    ) -> Iterable[_T]:
        del total, desc
        return iterable


@pytest.fixture
def synthetic_omics_dataset() -> OmicsDataset:
    times = np.array([0.0, 8.0, 16.0], dtype=float)
    dataset_frame = pd.DataFrame(
        {
            "Genes": ["gene_both", "gene_cond1", "gene_cond2"],
            "alpha_0": [1.0, 1.5, np.nan],
            "alpha_8": [1.2, 1.7, np.nan],
            "alpha_16": [1.1, 1.6, np.nan],
            "beta_0": [0.9, np.nan, 1.0],
            "beta_8": [1.1, np.nan, 1.2],
            "beta_16": [1.0, np.nan, 1.3],
        },
    )
    dataset = OmicsDataset(
        df=dataset_frame.copy(),
        columns_cond1=["alpha_0", "alpha_8", "alpha_16"],
        columns_cond2=["beta_0", "beta_8", "beta_16"],
        t_cond1=times,
        t_cond2=times,
        cond1_label="Alpha",
        cond2_label="Beta",
    )
    dataset.df["is_expressed_cond1"] = [True, True, False]
    dataset.df["is_expressed_cond2"] = [True, False, True]
    return dataset


@pytest.fixture
def heatmap_dataframe() -> pd.DataFrame:
    data = {
        "Genes": [
            "gene_m2b",
            "gene_m2c",
            "gene_m3a",
            "gene_m3c",
            "gene_m4",
            "gene_m5",
        ],
        "model": [2, 2, 3, 3, 4, 5],
        "subclass": ["b", "c", "a", "c", "c", "c"],
        "alpha_phase": [np.nan, 1.0, 2.0, 3.0, 4.0, 5.0],
        "beta_phase": [0.0, 1.5, np.nan, 3.0, 4.0, 5.0],
        "alpha_amp": [np.nan, 0.6, 1.2, 0.8, 1.1, 1.0],
        "beta_amp": [0.9, 1.1, np.nan, 0.7, 0.6, 0.5],
        "w_model1": [np.nan, 0.0, 0.0, 0.0, 0.0, 0.0],
        "w_model2": [0.7, 0.2, 0.1, 0.1, 0.1, 0.1],
        "w_model3": [np.nan, 0.1, 0.6, 0.3, 0.2, 0.2],
        "w_model4": [np.nan, 0.1, np.nan, 0.3, 0.5, 0.3],
        "w_model5": [np.nan, 0.6, np.nan, 0.2, 0.2, 0.4],
        "alpha_0": [np.nan, 1.2, 2.0, 1.1, 0.9, 1.0],
        "alpha_8": [np.nan, 1.0, 2.2, 1.3, 1.0, 1.2],
        "alpha_16": [np.nan, 1.1, 2.4, 1.4, 1.1, 1.3],
        "beta_0": [1.0, 1.0, np.nan, 0.9, 1.2, 1.3],
        "beta_8": [1.1, 0.9, np.nan, 1.2, 0.8, 1.5],
        "beta_16": [1.2, 0.8, np.nan, 1.1, 0.9, 1.4],
        "is_expressed_cond1": [False, True, True, True, True, True],
        "is_expressed_cond2": [True, True, False, True, True, True],
    }
    heatmap_frame = pd.DataFrame(data)
    heatmap_frame["mean_cond1"] = heatmap_frame[["alpha_0", "alpha_8", "alpha_16"]].mean(axis=1)
    heatmap_frame["mean_cond2"] = heatmap_frame[["beta_0", "beta_8", "beta_16"]].mean(axis=1)
    return heatmap_frame


@pytest.fixture
def omics_heatmap(heatmap_dataframe: pd.DataFrame) -> OmicsHeatmap:
    times = np.array([0.0, 8.0, 16.0], dtype=float)
    return OmicsHeatmap(
        df=heatmap_dataframe,
        columns_cond1=["alpha_0", "alpha_8", "alpha_16"],
        columns_cond2=["beta_0", "beta_8", "beta_16"],
        t_cond1=times,
        t_cond2=times,
        cond1_label="Alpha",
        cond2_label="Beta",
    )


def test_omics_heatmap_timepoint_means_shape(omics_heatmap: OmicsHeatmap) -> None:
    means_cond1 = omics_heatmap.timepoint_means(
        omics_heatmap.df,
        omics_heatmap.columns_cond1,
        omics_heatmap.t_cond1,
    )
    means_cond2 = omics_heatmap.timepoint_means(
        omics_heatmap.df,
        omics_heatmap.columns_cond2,
        omics_heatmap.t_cond2,
    )

    assert means_cond1.shape == (len(omics_heatmap.df), len(np.unique(omics_heatmap.t_cond1)))  # noqa: S101
    assert means_cond2.shape == (len(omics_heatmap.df), len(np.unique(omics_heatmap.t_cond2)))  # noqa: S101


def test_omics_heatmap_get_z_score_standardizes_rows(omics_heatmap: OmicsHeatmap) -> None:
    values = np.array([[1.0, 2.0, 3.0], [2.0, 2.0, 2.0]], dtype=float)
    z_scores = omics_heatmap.get_z_score(values)

    assert np.allclose(z_scores[0].mean(), 0.0)  # noqa: S101
    assert np.allclose(z_scores[0].std(ddof=0), 1.0)  # noqa: S101
    assert np.all(z_scores[1] == 0.0)  # noqa: S101


def test_omics_heatmap_plot_heatmap_returns_figure(omics_heatmap: OmicsHeatmap) -> None:
    fig = omics_heatmap.plot_heatmap()

    assert fig.axes  # noqa: S101
    plt.close(fig)


@pytest.fixture
def time_series_dataframe() -> pd.DataFrame:
    data = {
        "Genes": ["gene_both", "gene_cond1", "gene_cond2"],
        "alpha_0": [1.0, 1.5, np.nan],
        "alpha_8": [1.2, 1.7, np.nan],
        "alpha_16": [1.1, 1.6, np.nan],
        "beta_0": [1.0, np.nan, 1.1],
        "beta_8": [1.2, np.nan, 1.3],
        "beta_16": [1.1, np.nan, 1.2],
        "model": [1, 1, 1],
        "subclass": ["c", "a", "b"],
        "is_expressed_cond1": [True, True, False],
        "is_expressed_cond2": [True, False, True],
    }
    series_frame = pd.DataFrame(data)
    series_frame["mean_cond1"] = series_frame[["alpha_0", "alpha_8", "alpha_16"]].mean(axis=1)
    series_frame["mean_cond2"] = series_frame[["beta_0", "beta_8", "beta_16"]].mean(axis=1)
    return series_frame


@pytest.fixture
def time_series_example(time_series_dataframe: pd.DataFrame) -> TimeSeriesExample:
    times = np.array([0.0, 8.0, 16.0], dtype=float)
    return TimeSeriesExample(
        df=time_series_dataframe,
        columns_cond1=["alpha_0", "alpha_8", "alpha_16"],
        columns_cond2=["beta_0", "beta_8", "beta_16"],
        t_cond1=times,
        t_cond2=times,
        cond1_label="Alpha",
        cond2_label="Beta",
    )


def test_time_series_example_get_test_function_expressed_both(time_series_example: TimeSeriesExample) -> None:
    row = time_series_example.df[time_series_example.df["Genes"] == "gene_both"].iloc[0]
    alpha_vec = row[time_series_example.columns_cond1].to_numpy(float)
    beta_vec = row[time_series_example.columns_cond2].to_numpy(float)

    t_test, cond1_pred, cond2_pred = time_series_example.get_test_function_expressed_both(
        alpha_vec,
        beta_vec,
        time_series_example.t_cond1,
        time_series_example.t_cond2,
        model=1,
    )

    assert t_test.shape == (100,)  # noqa: S101
    assert cond1_pred.shape == (100,)  # noqa: S101
    assert cond2_pred.shape == (100,)  # noqa: S101
    assert np.isfinite(cond1_pred).all()  # noqa: S101
    assert np.isfinite(cond2_pred).all()  # noqa: S101


def test_time_series_example_conditional_predictions(time_series_example: TimeSeriesExample) -> None:
    row_cond1 = time_series_example.df[time_series_example.df["Genes"] == "gene_cond1"].iloc[0]
    alpha_vec = row_cond1[time_series_example.columns_cond1].to_numpy(float)

    _, cond1_only_pred, cond1_only_placeholder = time_series_example.get_test_function_expressed_cond1(
        alpha_vec,
        time_series_example.t_cond1,
        model=1,
    )

    row_cond2 = time_series_example.df[time_series_example.df["Genes"] == "gene_cond2"].iloc[0]
    beta_vec = row_cond2[time_series_example.columns_cond2].to_numpy(float)

    _, cond2_placeholder, cond2_only_pred = time_series_example.get_test_function_expressed_cond2(
        beta_vec,
        time_series_example.t_cond2,
        model=1,
    )

    assert np.isfinite(cond1_only_pred).all()  # noqa: S101
    assert np.isnan(cond1_only_placeholder).all()  # noqa: S101
    assert np.isnan(cond2_placeholder).all()  # noqa: S101
    assert np.isfinite(cond2_only_pred).all()  # noqa: S101


def test_time_series_example_plot_time_series_missing_gene(time_series_example: TimeSeriesExample) -> None:
    with pytest.raises(ValueError, match="not found"):
        time_series_example.plot_time_series("missing_gene")
