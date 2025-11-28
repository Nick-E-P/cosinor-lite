from __future__ import annotations

from typing import Literal, Self

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import ConfigDict, field_validator, model_validator
from pydantic.dataclasses import dataclass
from scipy.stats import spearmanr

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 8,
        "pdf.fonttype": 42,
    },
)
plt.style.use("seaborn-v0_8-ticks")


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class OmicsDataset:
    """
    Represent an omics expression table and derived metrics.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table containing gene expression measurements.
    columns_cond1 : list[str]
        Column names for condition 1 samples.
    columns_cond2 : list[str]
        Column names for condition 2 samples.
    t_cond1 : numpy.ndarray
        Time points associated with ``columns_cond1``.
    t_cond2 : numpy.ndarray
        Time points associated with ``columns_cond2``.
    cond1_label : str, optional
        Display label for condition 1, by default ``"cond1"``.
    cond2_label : str, optional
        Display label for condition 2, by default ``"cond2"``.
    deduplicate_on_init : bool, optional
        Whether to deduplicate genes immediately after initialization, by default ``False``.

    """

    df: pd.DataFrame

    columns_cond1: list[str]
    columns_cond2: list[str]
    t_cond1: np.ndarray
    t_cond2: np.ndarray
    cond1_label: str = "cond1"
    cond2_label: str = "cond2"

    deduplicate_on_init: bool = False

    @field_validator("t_cond1", "t_cond2", mode="before")
    @classmethod
    def _to_1d_f64(cls, v: object) -> np.ndarray:  # ← return type fixes ANN206
        """
        Coerce incoming arrays to one-dimensional ``float64`` np.ndarrays.

        Parameters
        ----------
        v : object
            Sequence-like data to convert.

        Returns
        -------
        numpy.ndarray
            Converted one-dimensional array.

        Raises
        ------
        ValueError
            If the incoming value cannot be reshaped to one dimension.


        """
        a = np.asarray(v, dtype=np.float64)
        if a.ndim != 1:
            text: str = "expected 1D array"
            raise ValueError(text)
        return a

    @model_validator(mode="after")
    def _check_columns(self) -> Self:
        """
        Ensure all requested columns are present in the DataFrame.

        Returns
        -------
        Self
            The validated dataset instance.

        Raises
        ------
        ValueError
            If any required columns are missing.


        """
        missing = [c for c in (self.columns_cond1 + self.columns_cond2) if c not in self.df.columns]
        if missing:
            text = f"Missing columns: {missing}"  # satisfies EM101 (no string literal in raise)
            raise ValueError(text)
        return self

    def __post_init__(self) -> None:
        """Populate derived columns and optionally deduplicate entries."""
        self.add_detected_timepoint_counts()
        self.add_mean_expression()
        self.add_number_detected()
        if self.deduplicate_on_init:
            self.deduplicate_genes()

    def detected_timepoint_counts(self, cond: str) -> list[int]:
        """
        Count detected time points per gene for the requested condition.

        Parameters
        ----------
        cond : str
            Either ``"cond1"`` or ``"cond2"`` specifying which condition to evaluate.

        Returns
        -------
        list[int]
            Number of distinct time points detected for each gene.

        Raises
        ------
        ValueError
            If ``cond`` is not a recognized condition label.


        """
        if cond == "cond1":
            y = self.df[self.columns_cond1]
            t = self.t_cond1
        elif cond == "cond2":
            y = self.df[self.columns_cond2]
            t = self.t_cond2
        else:
            text = f"Invalid condition: {cond}"  # satisfies EM101 (no string literal in raise)
            raise ValueError(text)

        mask = y.notna()

        detected_timepoints = mask.T.groupby(t).any().T.sum(axis=1).to_numpy()

        return detected_timepoints

    def add_detected_timepoint_counts(self) -> None:
        """Augment the DataFrame with detected time-point counts for each condition."""
        self.df["detected_timepoints_cond1"] = self.detected_timepoint_counts("cond1")
        self.df["detected_timepoints_cond2"] = self.detected_timepoint_counts("cond2")

    def add_mean_expression(self) -> None:
        """Compute mean expression for both conditions and store in the DataFrame."""
        self.df["mean_cond1"] = self.df[self.columns_cond1].mean(axis=1)
        self.df["mean_cond2"] = self.df[self.columns_cond2].mean(axis=1)

    def add_number_detected(self) -> None:
        """Store the count of non-null measurements for each condition."""
        self.df["num_detected_cond1"] = self.df[self.columns_cond1].count(axis=1)
        self.df["num_detected_cond2"] = self.df[self.columns_cond2].count(axis=1)

    def deduplicate_genes(self) -> None:
        """Remove duplicate gene entries, keeping the highest combined mean expression."""
        if not {"mean_cond1", "mean_cond2"}.issubset(self.df):
            self.add_mean_expression()

        self.df = (
            self.df.assign(total_mean=self.df["mean_cond1"] + self.df["mean_cond2"])
            .sort_values("total_mean", ascending=False)
            .drop_duplicates(subset="Genes", keep="first")
            .drop(columns="total_mean")
        )

    def add_is_expressed(
        self,
        *,
        detected_min: int | None = None,
        mean_min: float | None = None,
        num_detected_min: int | None = None,
    ) -> None:
        """
        Flag genes as expressed based on configurable thresholds.

        Parameters
        ----------
        detected_min : int | None, optional
            Minimum detected time points required, by default ``None``.
        mean_min : float | None, optional
            Minimum mean expression required, by default ``None``.
        num_detected_min : int | None, optional
            Minimum number of detected samples required, by default ``None``.


        """
        if not {"detected_timepoints_cond1", "detected_timepoints_cond2"}.issubset(self.df):
            self.add_detected_timepoint_counts()
        if not {"mean_cond1", "mean_cond2"}.issubset(self.df):
            self.add_mean_expression()
        if not {"num_detected_cond1", "num_detected_cond2"}.issubset(self.df):
            self.add_number_detected()

        def _mask(which: Literal["cond1", "cond2"]) -> pd.Series:
            m_detected = pd.Series(True, index=self.df.index)
            m_mean = pd.Series(True, index=self.df.index)
            m_num = pd.Series(True, index=self.df.index)

            if detected_min is not None:
                m_detected = self.df[f"detected_{which}"] >= detected_min
            if mean_min is not None:
                m_mean = self.df[f"mean_{which}"] >= mean_min
            if num_detected_min is not None:
                m_num = self.df[f"num_detected_{which}"] >= num_detected_min

            return m_detected & m_mean & m_num

        self.df["is_expressed_cond1"] = _mask("cond1")
        self.df["is_expressed_cond2"] = _mask("cond2")

    def expression_histogram(self, bins: int = 20) -> plt.Figure:
        """
        Plot histograms of mean expression for both conditions.

        Parameters
        ----------
        bins : int, optional
            Number of histogram bins, by default 20.

        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the histogram panels.


        """
        print(plt.rcParams["font.size"])
        fig = plt.figure(figsize=(6 / 2.54, 12 / 2.54))
        plt.subplot(2, 1, 1)
        plt.hist(self.df["mean_cond1"].to_numpy().flatten(), bins=bins)
        plt.xlabel("Mean Expression")
        plt.ylabel("Frequency")
        plt.title(f"Mean expression ({self.cond1_label})")
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=10))
        plt.subplot(2, 1, 2)
        plt.hist(self.df["mean_cond2"].to_numpy().flatten(), bins=bins)
        plt.xlabel("Mean Expression")
        plt.ylabel("Density")
        plt.title(f"Mean expression ({self.cond2_label})")
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=10))
        plt.tight_layout()

        return fig

    def replicate_scatterplot(self, sample1: str, sample2: str) -> plt.Figure:
        """
        Create a scatterplot comparing two samples with correlation annotations.

        Parameters
        ----------
        sample1 : str
            Column name of the first sample.
        sample2 : str
            Column name of the second sample.

        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the scatter plot.

        Raises
        ------
        ValueError
            If either sample column is missing from the DataFrame.


        """
        if sample1 not in self.df.columns or sample2 not in self.df.columns:
            text = f"Samples {sample1} and/or {sample2} not in DataFrame columns."
            raise ValueError(text)

        fig = plt.figure(figsize=(8 / 2.54, 8 / 2.54))
        xy = self.df[[sample1, sample2]].dropna()
        x: np.ndarray = xy[sample1].to_numpy().flatten()
        y: np.ndarray = xy[sample2].to_numpy().flatten()
        r_pearson: float = np.corrcoef(x, y)[0, 1]
        r_spearman: float = spearmanr(x, y).statistic
        plt.scatter(x, y, alpha=0.1, s=4)
        plt.xlabel(sample1)
        plt.ylabel(sample2)
        plt.title(f"Pearson R = {r_pearson:.2f}, Spearman R = {r_spearman:.2f}")
        plt.axis("equal")
        plt.plot(
            [x.min(), x.max()],
            [x.min(), x.max()],
            color="grey",
            linestyle="--",
            alpha=0.8,
        )
        plt.tight_layout()

        return fig
