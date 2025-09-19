from __future__ import annotations

from typing import Literal, Self

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import ConfigDict, field_validator, model_validator
from pydantic.dataclasses import dataclass
from scipy.stats import spearmanr


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class OmicsDataset:
    df: pd.DataFrame

    columns_cond1: list[str]
    columns_cond2: list[str]
    t_cond1: np.ndarray
    t_cond2: np.ndarray
    cond1_label: str = "cond1"
    cond2_label: str = "cond2"

    deduplicate_on_init: bool = False

    # --- validators ---
    @field_validator("t_cond1", "t_cond2", mode="before")
    @classmethod
    def _to_1d_f64(cls, v: object) -> np.ndarray:  # ← return type fixes ANN206
        a = np.asarray(v, dtype=np.float64)
        if a.ndim != 1:
            text: str = "expected 1D array"
            raise ValueError(text)
        return a

    @model_validator(mode="after")
    def _check_columns(self) -> Self:
        missing = [c for c in (self.columns_cond1 + self.columns_cond2) if c not in self.df.columns]
        if missing:
            text = f"Missing columns: {missing}"  # satisfies EM101 (no string literal in raise)
            raise ValueError(text)
        return self

    def __post_init__(self) -> None:
        self.add_detected_timepoint_counts()
        self.add_mean_expression()
        self.add_number_detected()
        if self.deduplicate_on_init:
            self.deduplicate_genes()

    def detected_timepoint_counts(self, cond: str) -> list[int]:
        """
        Count number of timepoints with detected values for each gene.

        Args:
            cond (str): "cond1" or "cond2"

        Returns:
            list[int]: List of counts for each gene.

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

        # y-like frame, boolean mask of non-NaN
        mask = y.notna()  # shape: (n_rows, n_cols)

        # t_beta must be length n_cols and aligned to those columns
        # Group columns by time, check if ANY non-NaN per group, then count groups per row
        detected_timepoints = (
            mask.T.groupby(t)
            .any()  # (n_rows, n_unique_times) booleans
            .T.sum(axis=1)  # per-row counts
            .to_numpy()
        )

        return detected_timepoints

    def add_detected_timepoint_counts(self) -> None:
        """Add two columns to self.df with counts for cond1 and cond2."""
        self.df["detected_cond1"] = self.detected_timepoint_counts("cond1")
        self.df["detected_cond2"] = self.detected_timepoint_counts("cond2")

    def add_mean_expression(self) -> None:
        """Add two columns to self.df with mean expression for cond1 and cond2."""
        self.df["mean_cond1"] = self.df[self.columns_cond1].mean(axis=1)
        self.df["mean_cond2"] = self.df[self.columns_cond2].mean(axis=1)

    def add_number_detected(self) -> None:
        """Add two columns to self.df with number of detected values for cond1 and cond2."""
        self.df["num_detected_cond1"] = self.df[self.columns_cond1].count(axis=1)
        self.df["num_detected_cond2"] = self.df[self.columns_cond2].count(axis=1)

    def deduplicate_genes(self) -> None:
        """Deduplicate self.df by 'Genes', keeping entry with highest total mean expression."""
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
        """Add is_expressed_cond1/cond2 based on thresholds."""
        # Ensure prerequisite columns exist
        if not {"detected_cond1", "detected_cond2"}.issubset(self.df):
            self.add_detected_timepoint_counts()
        if not {"mean_cond1", "mean_cond2"}.issubset(self.df):
            self.add_mean_expression()
        if not {"num_detected_cond1", "num_detected_cond2"}.issubset(self.df):
            self.add_number_detected()

        def _mask(which: Literal["cond1", "cond2"]) -> pd.Series:
            # start with all-True masks
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

    def expression_histogram(self, bins: int = 20) -> None:
        """Plot histogram of mean expression for cond1 and cond2."""
        print(plt.rcParams["font.size"])
        plt.figure(figsize=(6 / 2.54, 12 / 2.54))
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
        plt.show()

    def replicate_scatterplot(self, sample1: str, sample2: str) -> None:
        """Scatterplot of two replicates."""
        if sample1 not in self.df.columns or sample2 not in self.df.columns:
            text = f"Samples {sample1} and/or {sample2} not in DataFrame columns."
            raise ValueError(text)

        plt.figure(figsize=(8 / 2.54, 8 / 2.54))
        x: np.ndarray = self.df[sample1].to_numpy().flatten()
        y: np.ndarray = self.df[sample2].to_numpy().flatten()
        r_pearson: float = np.corrcoef(x, y)[0, 1]
        r_spearman: float = spearmanr(x, y).statistic
        plt.scatter(x, y, alpha=0.1, s=4)
        plt.xlabel(sample1)
        plt.ylabel(sample2)
        plt.title(f"{sample1} vs {sample2} (Pearson R = {r_pearson:.2f}, Spearman R = {r_spearman:.2f})")
        plt.axis("equal")
        plt.plot([x.min(), x.max()], [x.min(), x.max()], color="grey", linestyle="--", alpha=0.8)
        plt.tight_layout()
        plt.show()
