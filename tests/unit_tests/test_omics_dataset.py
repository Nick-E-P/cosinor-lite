from __future__ import annotations

import numpy as np
import pytest
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from scipy.stats import spearmanr

from cosinor_lite.omics_dataset import OmicsDataset


def test_detected_timepoint_counts_cond1_matches_manual(omics_dataset: OmicsDataset) -> None:
    dataset = omics_dataset
    manual = dataset.df[dataset.columns_cond1].notna().T.groupby(dataset.t_cond1).any().T.sum(axis=1).to_numpy()

    result = dataset.detected_timepoint_counts("cond1")

    assert np.array_equal(result, manual)  # noqa: S101


def test_detected_timepoint_counts_cond2_matches_manual(omics_dataset: OmicsDataset) -> None:
    dataset = omics_dataset
    manual = dataset.df[dataset.columns_cond2].notna().T.groupby(dataset.t_cond2).any().T.sum(axis=1).to_numpy()

    result = dataset.detected_timepoint_counts("cond2")

    assert np.array_equal(result, manual)  # noqa: S101


def test_detected_timepoint_counts_invalid_condition(omics_dataset: OmicsDataset) -> None:
    with pytest.raises(ValueError, match="Invalid condition: bad"):
        omics_dataset.detected_timepoint_counts("bad")


def test_mean_expression_columns_initialised(omics_dataset: OmicsDataset) -> None:
    dataset = omics_dataset
    mean_cond1 = dataset.df[dataset.columns_cond1].mean(axis=1)
    mean_cond2 = dataset.df[dataset.columns_cond2].mean(axis=1)

    assert np.allclose(dataset.df["mean_cond1"].to_numpy(), mean_cond1.to_numpy())  # noqa: S101
    assert np.allclose(dataset.df["mean_cond2"].to_numpy(), mean_cond2.to_numpy())  # noqa: S101


def test_number_detected_columns_initialised(omics_dataset: OmicsDataset) -> None:
    dataset = omics_dataset
    count_cond1 = dataset.df[dataset.columns_cond1].count(axis=1)
    count_cond2 = dataset.df[dataset.columns_cond2].count(axis=1)

    assert np.array_equal(  # noqa: S101
        dataset.df["num_detected_cond1"].to_numpy(),
        count_cond1.to_numpy(),
    )
    assert np.array_equal(  # noqa: S101
        dataset.df["num_detected_cond2"].to_numpy(),
        count_cond2.to_numpy(),
    )


def test_expression_histogram_returns_two_panels(omics_dataset: OmicsDataset) -> None:
    figure = omics_dataset.expression_histogram(bins=15)
    expected_axis_count = 2

    try:
        assert isinstance(figure, Figure)  # noqa: S101
        assert len(figure.axes) == expected_axis_count  # noqa: S101
        assert figure.axes[0].get_title() == "Mean expression (Alpha)"  # noqa: S101
        assert figure.axes[1].get_title() == "Mean expression (Beta)"  # noqa: S101
    finally:
        plt.close(figure)


def test_replicate_scatterplot_title_matches_correlations(omics_dataset: OmicsDataset) -> None:
    dataset = omics_dataset
    sample1, sample2 = dataset.columns_cond1[:2]
    figure = dataset.replicate_scatterplot(sample1, sample2)

    try:
        filtered = dataset.df[[sample1, sample2]].dropna()
        x = filtered[sample1].to_numpy().flatten()
        y = filtered[sample2].to_numpy().flatten()
        expected_title = f"Pearson R = {np.corrcoef(x, y)[0, 1]:.2f}, Spearman R = {spearmanr(x, y).statistic:.2f}"
        assert isinstance(figure, Figure)  # noqa: S101
        assert figure.axes[0].get_title() == expected_title  # noqa: S101
    finally:
        plt.close(figure)


def test_log2_transform_applies_to_measurements(omics_dataset: OmicsDataset) -> None:
    dataset = omics_dataset
    measurement_cols = dataset.columns_cond1 + dataset.columns_cond2
    transformed = OmicsDataset(
        df=dataset.df.copy(),
        columns_cond1=list(dataset.columns_cond1),
        columns_cond2=list(dataset.columns_cond2),
        t_cond1=dataset.t_cond1.copy(),
        t_cond2=dataset.t_cond2.copy(),
        cond1_label=dataset.cond1_label,
        cond2_label=dataset.cond2_label,
        log2_transform=True,
    )

    expected = np.log2(dataset.df[measurement_cols].astype(float) + 1.0)
    actual = transformed.df[measurement_cols]

    assert np.allclose(actual.to_numpy(), expected.to_numpy(), equal_nan=True)  # noqa: S101
