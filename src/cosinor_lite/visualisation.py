from __future__ import annotations

from typing import Self

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from pydantic import ConfigDict, field_validator, model_validator
from pydantic.dataclasses import dataclass


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class OmicsHeatmap:
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

    def timepoint_means(self, df: pd.DataFrame, columns: list[str], times: np.ndarray) -> np.ndarray:
        """
        Compute mean expression across columns at each unique timepoint.

        Args:
            df: DataFrame with expression values.
            columns: Subset of column names to use.
            times: 1D array of timepoints, length must equal len(columns).

        Returns:
            np.ndarray of shape (n_genes, n_unique_times).

        """
        if len(columns) != len(times):
            text: str = f"Length of columns ({len(columns)}) must match length of times ({len(times)})"
            raise ValueError(text)

        values: np.ndarray = df[columns].to_numpy()
        unique_times: np.ndarray = np.unique(times)

        result: np.ndarray = np.column_stack(
            [values[:, times == t].mean(axis=1) for t in unique_times],
        )

        return result

    def get_z_score(self, arr: np.ndarray) -> np.ndarray:
        """Compute z-score normalization for each row in a 2D array."""
        arr: np.ndarray = (arr - np.mean(arr, axis=1).reshape(-1, 1)) / np.where(
            np.std(arr, axis=1).reshape(-1, 1) == 0,
            1,
            np.std(arr, axis=1).reshape(-1, 1),
        )
        return arr

    def plot_heatmap(self, cmap: str = "bwr") -> None:
        df: pd.DataFrame = self.df
        df: pd.DataFrame = df.sort_values(by=["alpha_phase", "beta_phase"]).reset_index(drop=True)

        t_unique: np.ndarray = np.unique(self.t_cond1).astype(int)

        mean_cond1: np.ndarray = self.timepoint_means(df, self.columns_cond1, self.t_cond1)
        mean_cond2: np.ndarray = self.timepoint_means(df, self.columns_cond2, self.t_cond2)

        z_cond1: np.ndarray = self.get_z_score(mean_cond1)
        z_cond2: np.ndarray = self.get_z_score(mean_cond2)

        total_rows: int = (df["model"].isin([2, 3, 4, 5])).sum()

        m2: int = 2
        m3: int = 3
        m4: int = 4
        m5: int = 5

        n_m2a: int = ((df["model"] == m2) & (df["subclass"] == "b")).sum()
        n_m2b: int = ((df["model"] == m2) & (df["subclass"] == "c")).sum()
        n_m3a: int = ((df["model"] == m3) & (df["subclass"] == "a")).sum()
        n_m3b: int = ((df["model"] == m3) & (df["subclass"] == "c")).sum()
        n_m4: int = (df["model"] == m4).sum()
        n_m5: int = (df["model"] == m5).sum()

        _fig, axes = plt.subplots(
            nrows=6,
            ncols=2,
            gridspec_kw={
                "height_ratios": [
                    n_m2a / total_rows,
                    n_m2b / total_rows,
                    n_m3a / total_rows,
                    n_m3b / total_rows,
                    n_m4 / total_rows,
                    n_m5 / total_rows,
                ],
                "width_ratios": [1, 1],  # Equal column widths
            },
            figsize=(12 / 2.54, 18 / 2.54),  # Adjust figure size as needed
        )

        vmin_global = -2.5
        vmax_global = 2.5

        mask = (df["model"] == m2) & (df["subclass"] == "b")
        z_cond1_filtered = z_cond1[mask.to_numpy()]
        im1 = axes[0, 0].imshow(
            z_cond1_filtered,
            aspect="auto",
            cmap=cmap,
            vmin=vmin_global,
            vmax=vmax_global,
            rasterized=False,
            interpolation="none",
        )

        axes[0, 0].set_title("Alpha cells")
        axes[0, 0].set_xticks([])
        axes[0, 0].set_yticks([])

        # Add text to the left of the y-axis
        axes[0, 0].text(
            -0.2,
            0.5,
            "M2a",  # Coordinates: (x, y)
            transform=axes[0, 0].transAxes,  # Use axis-relative coordinates
            ha="center",
            va="center",  # Align text
            rotation=0,  # Rotate text vertically
            fontsize=8,  # Adjust font size as needed
        )

        mask = (df["model"] == m2) & (df["subclass"] == "c")
        z_cond1_filtered = z_cond1[mask.to_numpy()]
        im2 = axes[1, 0].imshow(
            z_cond1_filtered,
            aspect="auto",
            cmap=cmap,
            vmin=vmin_global,
            vmax=vmax_global,
            rasterized=False,
            interpolation="none",
        )
        axes[1, 0].set_xticks([])
        axes[1, 0].set_yticks([])

        # Add text to the left of the y-axis
        axes[1, 0].text(
            -0.2,
            0.5,
            "M2b",  # Coordinates: (x, y)
            transform=axes[1, 0].transAxes,  # Use axis-relative coordinates
            ha="center",
            va="center",  # Align text
            rotation=0,  # Rotate text vertically
            fontsize=8,  # Adjust font size as needed
        )
        mask = (df["model"] == m3) & (df["subclass"] == "a")
        z_cond1_filtered = z_cond1[mask.to_numpy()]
        im2 = axes[2, 0].imshow(
            z_cond1_filtered,
            aspect="auto",
            cmap=cmap,
            vmin=vmin_global,
            vmax=vmax_global,
            rasterized=False,
            interpolation="none",
        )
        axes[2, 0].set_xticks([])
        axes[2, 0].set_yticks([])

        # Add text to the left of the y-axis
        axes[2, 0].text(
            -0.2,
            0.5,
            "M3a",  # Coordinates: (x, y)
            transform=axes[2, 0].transAxes,  # Use axis-relative coordinates
            ha="center",
            va="center",  # Align text
            rotation=0,  # Rotate text vertically
            fontsize=8,  # Adjust font size as needed
        )
        mask = (df["model"] == m3) & (df["subclass"] == "c")
        z_cond1_filtered = z_cond1[mask.to_numpy()]
        im3 = axes[3, 0].imshow(
            z_cond1_filtered,
            aspect="auto",
            cmap=cmap,
            vmin=vmin_global,
            vmax=vmax_global,
            rasterized=False,
            interpolation="none",
        )
        axes[3, 0].set_xticks([])
        axes[3, 0].set_yticks([])
        # Add text to the left of the y-axis
        axes[3, 0].text(
            -0.2,
            0.5,
            "M3b",  # Coordinates: (x, y)
            transform=axes[3, 0].transAxes,  # Use axis-relative coordinates
            ha="center",
            va="center",  # Align text
            rotation=0,  # Rotate text vertically
            fontsize=8,  # Adjust font size as needed
        )

        mask = df["model"] == m4
        z_cond1_filtered = z_cond1[mask.to_numpy()]
        im4 = axes[4, 0].imshow(
            z_cond1_filtered,
            aspect="auto",
            cmap=cmap,
            vmin=vmin_global,
            vmax=vmax_global,
            rasterized=False,
            interpolation="none",
        )
        axes[4, 0].set_xticks([])
        axes[4, 0].set_yticks([])

        axes[4, 0].text(
            -0.2,
            0.5,
            "M4",  # Coordinates: (x, y)
            transform=axes[4, 0].transAxes,  # Use axis-relative coordinates
            ha="center",
            va="center",  # Align text
            rotation=0,  # Rotate text vertically
            fontsize=8,  # Adjust font size as needed
        )

        mask = df["model"] == m5
        z_cond1_filtered = z_cond1[mask.to_numpy()]
        im5 = axes[5, 0].imshow(
            z_cond1_filtered,
            aspect="auto",
            cmap=cmap,
            vmin=vmin_global,
            vmax=vmax_global,
            rasterized=False,
            interpolation="none",
        )

        axes[5, 0].set_xticks(range(len(t_unique)), t_unique)
        axes[5, 0].set_yticks([])
        axes[5, 0].set_xlabel("Time(h)")

        axes[5, 0].text(
            -0.2,
            0.5,
            "M5",  # Coordinates: (x, y)
            transform=axes[5, 0].transAxes,  # Use axis-relative coordinates
            ha="center",
            va="center",  # Align text
            rotation=0,  # Rotate text vertically
            fontsize=8,  # Adjust font size as needed
        )

        mask = (df["model"] == m2) & (df["subclass"] == "b")
        z_cond2_filtered = z_cond2[mask.to_numpy()]
        im6 = axes[0, 1].imshow(
            z_cond2_filtered,
            aspect="auto",
            cmap=cmap,
            vmin=vmin_global,
            vmax=vmax_global,
            rasterized=False,
            interpolation="none",
        )

        axes[0, 1].set_title("Beta cells")
        axes[0, 1].set_xticks([])
        axes[0, 1].set_yticks([])

        mask = (df["model"] == m2) & (df["subclass"] == "c")
        z_cond2_filtered = z_cond2[mask.to_numpy()]
        im7 = axes[1, 1].imshow(
            z_cond2_filtered,
            aspect="auto",
            cmap=cmap,
            vmin=vmin_global,
            vmax=vmax_global,
            rasterized=False,
            interpolation="none",
        )

        axes[1, 1].set_xticks([])
        axes[1, 1].set_yticks([])

        mask = (df["model"] == m3) & (df["subclass"] == "a")
        z_cond2_filtered = z_cond2[mask.to_numpy()]
        im8 = axes[2, 1].imshow(
            z_cond2_filtered,
            aspect="auto",
            cmap=cmap,
            vmin=vmin_global,
            vmax=vmax_global,
            rasterized=False,
            interpolation="none",
        )

        axes[2, 1].set_xticks([])
        axes[2, 1].set_yticks([])

        mask = (df["model"] == m3) & (df["subclass"] == "c")
        z_cond2_filtered = z_cond2[mask.to_numpy()]
        im9 = axes[3, 1].imshow(
            z_cond2_filtered,
            aspect="auto",
            cmap=cmap,
            vmin=vmin_global,
            vmax=vmax_global,
            rasterized=False,
            interpolation="none",
        )

        axes[3, 1].set_xticks([])
        axes[3, 1].set_yticks([])

        mask = df["model"] == m4
        z_cond2_filtered = z_cond2[mask.to_numpy()]
        im10 = axes[4, 1].imshow(
            z_cond2_filtered,
            aspect="auto",
            cmap=cmap,
            vmin=vmin_global,
            vmax=vmax_global,
            rasterized=False,
            interpolation="none",
        )

        axes[4, 1].set_xticks([])
        axes[4, 1].set_yticks([])

        mask = df["model"] == m5
        z_cond2_filtered = z_cond2[mask.to_numpy()]
        im11 = axes[5, 1].imshow(
            z_cond2_filtered,
            aspect="auto",
            cmap=cmap,
            vmin=vmin_global,
            vmax=vmax_global,
            rasterized=False,
            interpolation="none",
        )

        axes[5, 1].set_xticks(range(len(t_unique)), t_unique)
        axes[5, 1].set_yticks([])
        axes[5, 1].set_xlabel("Time(h)")


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class TimeSeriesExample:
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

    def plot_time_series(self, gene: str, xticks: np.ndarray | None = None) -> None:
        if xticks is None:
            xticks = np.unique(np.concatenate((self.t_cond1, self.t_cond2))).astype(int)
        df: pd.DataFrame = self.df
        df_curr = df[df["Genes"] == gene]
        if df_curr.empty:
            msg = f"Gene '{gene}' not found in the DataFrame."
            raise ValueError(msg)
        if len(df_curr) > 1:
            df_curr = df_curr.loc[[df_curr["mean_cond2"].idxmax()]]

        alpha_vec: np.ndarray = df_curr[self.columns_cond1].to_numpy(float).flatten()
        beta_vec: np.ndarray = df_curr[self.columns_cond2].to_numpy(float).flatten()

        is_expressed_cond1: bool = df_curr["is_expressed_cond1"].to_numpy()[0]
        is_expressed_cond2: bool = df_curr["is_expressed_cond2"].to_numpy()[0]

        model: int = df_curr["model"].to_numpy()[0]

        if is_expressed_cond1 and is_expressed_cond2:
            t_test, y_test_cond1, y_test_cond2 = self.get_test_function_expressed_both(
                alpha_vec,
                beta_vec,
                self.t_cond1,
                self.t_cond2,
                model,
            )
        elif is_expressed_cond1 and not is_expressed_cond2:
            t_test, y_test_cond1, y_test_cond2 = self.get_test_function_expressed_cond1(
                alpha_vec,
                self.t_cond1,
                model,
            )
        elif not is_expressed_cond1 and is_expressed_cond2:
            t_test, y_test_cond1, y_test_cond2 = self.get_test_function_expressed_cond2(
                beta_vec,
                self.t_cond2,
                model,
            )
        plt.figure(figsize=(12 / 2.54, 6 / 2.54))
        plt.subplot(1, 2, 1)
        plt.scatter(self.t_cond1, alpha_vec, s=4)
        plt.plot(t_test, y_test_cond1)
        plt.ylabel("Expression Level")
        plt.xticks(xticks)
        # plt.xlim(xticks[0], xticks[-1])
        plt.title(f"Time Series for {gene}")
        plt.xlabel("Time (h)")
        plt.subplot(1, 2, 2)
        plt.scatter(self.t_cond2, beta_vec, s=4, color="r")
        plt.plot(t_test, y_test_cond2, color="r")
        plt.title(f"Time Series for {gene}")
        plt.xlabel("Time (h)")

        plt.xticks(xticks)
        plt.xlim(xticks[0] - 0.05 * (xticks[-1] - xticks[0]), xticks[-1] + 0.05 * (xticks[-1] - xticks[0]))
        plt.tight_layout()
        plt.show()

    def get_test_function_expressed_both(
        self,
        alpha_vec: np.ndarray,
        beta_vec: np.ndarray,
        t_cond1: np.ndarray,
        t_cond2: np.ndarray,
        model: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        design: pd.DataFrame = build_design(alpha_vec, beta_vec, t_cond1, t_cond2)
        t_test: np.ndarray = np.linspace(0, 24, 100)
        if model == 0:
            print("No model selected")
            y_test_cond1 = np.full_like(t_test, np.nan)
            y_test_cond2 = np.full_like(t_test, np.nan)
        else:
            model_formula: str = MODELS[model - 1].formula
            res = smf.ols(model_formula, data=design).fit()
            df_cond1: pd.DataFrame = pd.DataFrame({"time": t_test, "dataset": "alpha"}).dropna()
            df_cond2: pd.DataFrame = pd.DataFrame({"time": t_test, "dataset": "beta"}).dropna()
            df_test: pd.DataFrame = pd.concat([df_cond1, df_cond2], ignore_index=True)
            df_test["constant"] = 1.0
            df_test["cos_wt"] = np.cos(W * df_test["time"].to_numpy().astype(float))
            df_test["sin_wt"] = np.sin(W * df_test["time"].to_numpy().astype(float))
            df_test["is_alpha"] = (df_test["dataset"] == "alpha").astype(int)
            df_test["is_beta"] = (df_test["dataset"] == "beta").astype(int)
            y_test = res.predict(exog=df_test)
            print(y_test)
            y_test_cond1: np.ndarray = y_test[df_test["dataset"] == "alpha"].to_numpy()
            y_test_cond2: np.ndarray = y_test[df_test["dataset"] == "beta"].to_numpy()
        return t_test, y_test_cond1, y_test_cond2

    def get_test_function_expressed_cond1(
        self,
        alpha_vec: np.ndarray,
        t_cond1: np.ndarray,
        model: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        design: pd.DataFrame = build_design_cond1(alpha_vec, t_cond1)
        t_test: np.ndarray = np.linspace(0, 24, 100)
        if model == 0:
            print("No model selected")
            y_test_cond1 = np.full_like(t_test, np.nan)
            y_test_cond2 = np.full_like(t_test, np.nan)
        else:
            print(f"Fitting model {model}")
            print(MODELS_ONE_CONDITION)
            if model == 1:
                model_formula: str = MODELS_ONE_CONDITION[0].formula
            else:
                model_formula: str = MODELS_ONE_CONDITION[1].formula
            res = smf.ols(model_formula, data=design).fit()
            df_test: pd.DataFrame = pd.DataFrame({"time": t_test, "dataset": "alpha"}).dropna()
            df_test["constant"] = 1.0
            df_test["cos_wt"] = np.cos(W * df_test["time"].to_numpy().astype(float))
            df_test["sin_wt"] = np.sin(W * df_test["time"].to_numpy().astype(float))
            y_test = res.predict(exog=df_test)
            print(y_test)
            y_test_cond1: np.ndarray = y_test.to_numpy()
            y_test_cond2: np.ndarray = np.full_like(t_test, np.nan)
        return t_test, y_test_cond1, y_test_cond2

    def get_test_function_expressed_cond2(
        self,
        beta_vec: np.ndarray,
        t_cond2: np.ndarray,
        model: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        design: pd.DataFrame = build_design_cond2(beta_vec, t_cond2)
        t_test: np.ndarray = np.linspace(0, 24, 100)
        if model == 0:
            print("No model selected")
            y_test_cond1 = np.full_like(t_test, np.nan)
            y_test_cond2 = np.full_like(t_test, np.nan)
        else:
            print(f"Fitting model {model}")
            print(MODELS_ONE_CONDITION)
            if model == 1:
                model_formula: str = MODELS_ONE_CONDITION[0].formula
            else:
                model_formula: str = MODELS_ONE_CONDITION[1].formula
            res = smf.ols(model_formula, data=design).fit()
            df_test: pd.DataFrame = pd.DataFrame({"time": t_test, "dataset": "alpha"}).dropna()
            df_test["constant"] = 1.0
            df_test["cos_wt"] = np.cos(W * df_test["time"].to_numpy().astype(float))
            df_test["sin_wt"] = np.sin(W * df_test["time"].to_numpy().astype(float))
            y_test = res.predict(exog=df_test)
            print(y_test)
            y_test_cond1: np.ndarray = np.full_like(t_test, np.nan)
            y_test_cond2: np.ndarray = y_test.to_numpy()

        return t_test, y_test_cond1, y_test_cond2
