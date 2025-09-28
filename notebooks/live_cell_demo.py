# %%
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pydantic import ConfigDict, field_validator, model_validator
from pydantic.dataclasses import dataclass
from scipy.optimize import curve_fit
from scipy.stats import f

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
class LiveCellDataset:
    ids: list[str]
    group: list[str]
    replicate: list[int]
    time_series: np.ndarray
    time: np.ndarray

    group1_label: str = "group1"
    group2_label: str = "group2"

    color_group1: str = "tab:blue"
    color_group2: str = "tab:orange"

    # --- validators ---
    @field_validator("time_series", mode="before")
    @classmethod
    def _to_2d_f64(cls, v: object) -> np.ndarray:  # ← return type fixes ANN206
        a = np.asarray(v, dtype=np.float64)
        if a.ndim != 2:
            text: str = "expected 2D array"
            raise ValueError(text)
        return a

    @model_validator(mode="after")
    def _check_columns(self) -> LiveCellDataset:
        if len(self.ids) != self.time_series.shape[1]:
            msg = "Length of ids must match number of columns in time_series"
            raise ValueError(msg)
        if len(self.group) != self.time_series.shape[1]:
            msg = "Length of group must match number of columns in time_series"
            raise ValueError(msg)
        if len(self.replicate) != self.time_series.shape[1]:
            msg = "Length of replicate must match number of columns in time_series"
            raise ValueError(msg)
        unique_groups = set(self.group)
        if len(unique_groups) != 2:
            msg = "There must be exactly two unique groups in 'group' column"
            raise ValueError(msg)
        return self

    def get_group1_ids_replicates_data(self) -> tuple[list[str], list[int], np.ndarray]:
        mask = np.array(self.group) == self.group1_label
        ids = list(np.array(self.ids)[mask])
        replicates = list(np.array(self.replicate)[mask])
        data = self.time_series[:, mask]
        return ids, replicates, data

    def get_group2_ids_replicates_data(self) -> tuple[list[str], list[int], np.ndarray]:
        mask = np.array(self.group) == self.group2_label
        ids = list(np.array(self.ids)[mask])
        replicates = list(np.array(self.replicate)[mask])
        data = self.time_series[:, mask]
        return ids, replicates, data

    def linear_trend(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        model = sm.OLS(y, sm.add_constant(x)).fit()
        linear_fit = model.predict(sm.add_constant(x))
        return x, linear_fit

    def poly2_trend(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        coeffs = np.polyfit(x, y, 2)
        poly_fit = np.polyval(coeffs, x)
        return x, poly_fit

    def moving_avergae_trend(self, x: np.ndarray, y: np.ndarray, window: int = 5) -> tuple[np.ndarray, np.ndarray]:
        if window < 1:
            msg = "Window size must be at least 1."
            raise ValueError(msg)
        y_series = pd.Series(y)
        ma_fit = y_series.rolling(window=window, center=True).mean().to_numpy()
        return x, ma_fit

    def get_trend(self, y: np.ndarray, method: str = "linear", window: int = 5) -> tuple[np.ndarray, np.ndarray]:
        if method == "linear":
            x_processed, trend = self.linear_trend(self.time, y)
        elif method == "poly2":
            x_processed, trend = self.poly2_trend(self.time, y)
        elif method == "moving_average":
            x_processed, trend = self.moving_avergae_trend(self.time, y, window=window)
        else:
            msg = f"Unknown detrending method: {method}"
            raise ValueError(msg)

        return x_processed, trend

    def plot_group_data(self, group: str, m: int = 5) -> None:
        if group == "group1":
            ids, replicates, data = self.get_group1_ids_replicates_data()
            color = self.color_group1
            n_group = len(ids)
        elif group == "group2":
            ids, replicates, data = self.get_group2_ids_replicates_data()
            color = self.color_group2
            n_group = len(ids)
        else:
            msg = "group must be 'group1' or 'group2'"
            raise ValueError(msg)

        n = np.ceil(n_group / m).astype(int)

        study_list = np.unique(ids).tolist()

        fig = plt.figure(figsize=(5 * n / 2.54, 5 * m / 2.54))

        for i, id_curr in enumerate(study_list):
            mask = np.array(ids) == id_curr
            n_reps = np.sum(mask)

            ax = fig.add_subplot(n, m, i + 1)

            for j in range(n_reps):
                x = self.time
                y = data[:, mask][:, j]
                ax.plot(x, y, color=color)
                # x_processed, y_processed = self.linear_trend(x, y)
                # ax.plot(x_processed, y_processed, color="black", linestyle="--", linewidth=0.8)
                # x_processed, y_processed = self.poly2_trend(x, y)
                # ax.plot(x_processed, y_processed, color="black", linestyle="--", linewidth=0.8)
                # x_processed, y_processed = self.moving_avergae_trend(x, y, window=144)
                x_processed, y_processed = self.get_trend(y, method="moving_average", window=144)
                ax.plot(x_processed, y_processed, color="black", linestyle="--", linewidth=0.8)
            ax.set_title(f"{ids[i]} (n={n_reps})")
            ax.set_xlabel("Time")
            ax.set_ylabel("Expression")

        plt.tight_layout()
        plt.show()


# %%


def constant_model(x, mesor):
    # mesor = params
    return mesor


def cosine_model_24(x, amplitude, acrophase, mesor):
    # amplitude, acrophase, mesor = params
    period = 24.0
    return amplitude * np.cos(2 * np.pi * (x - acrophase) / period) + mesor


class CosinorAnalysis(LiveCellDataset):
    def __init__(
        self,
        *args,
        period: float = 24.0,
        method: str = "ols",
        t_lower=0.0,
        t_upper=720.0,
        **kwargs,
    ):
        # initialize the parent dataset
        super().__init__(*args, **kwargs)

        # store cosinor-specific parameters
        self.t_lower = t_lower
        self.t_upper = t_upper
        self.period = period
        self.method = method

    def fit_cosinor_24(self, x: np.ndarray, y: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
        # initial parameter guesses: amplitude, acrophase, mesor, sigma

        initial_params = [np.mean(y)]
        params_opt, _ = curve_fit(constant_model, x, y, p0=initial_params)
        model_predictions_constant = constant_model(x, params_opt)

        initial_params = [np.std(y), 0.0, np.mean(y)]
        params_opt, _ = curve_fit(cosine_model_24, x, y, p0=initial_params)
        amplitude_fit, acrophase_fit, mesor_fit = params_opt
        model_predictions_cosine = cosine_model_24(x, amplitude_fit, acrophase_fit, mesor_fit)

        rss_cosinor = np.sum((model_predictions_cosine - y) ** 2)
        rss_constant = np.sum((model_predictions_constant - y) ** 2)

        n = len(y)
        p1 = 1
        p2 = 3

        f_test = ((rss_constant - rss_cosinor) / rss_cosinor) * ((n - p2) / (p2 - p1))

        df_model = p2 - p1
        df_residuals = n - p2

        f_statistic_p_value = f.sf(f_test, df_model, df_residuals)

        t_test_acro = np.linspace(0, 24, 1440)
        y_test_acro = cosine_model_24(t_test_acro, amplitude_fit, acrophase_fit, mesor_fit)

        mesor = mesor_fit
        amplitude = abs(amplitude_fit)
        acrophase = t_test_acro[np.argmax(y_test_acro)]

        t_test = np.linspace(x[0], x[-1], 1440)
        y_test = cosine_model_24(t_test, amplitude_fit, acrophase_fit, mesor_fit)

        results = {
            "mesor": mesor,
            "amplitude": amplitude,
            "acrophase": acrophase,
            "p-val osc": f_statistic_p_value,
        }
        return results, t_test, y_test

    def fit_cosinor(self, group: str, m: int = 5) -> None:
        ids1, replicates1, data1 = self.get_group1_ids_replicates_data()
        ids2, replicates2, data2 = self.get_group2_ids_replicates_data()

        N_group1 = len(ids1)
        N_group2 = len(ids2)

        n = np.ceil(N_group1 / 5).astype(int)

        study_list = np.unique(ids1).tolist()

        fig = plt.figure(figsize=(5 * n / 2.54, 5 * m / 2.54))

        for i, id_curr in enumerate(study_list):
            mask = np.array(ids1) == id_curr
            n_reps = np.sum(mask)

            ax = fig.add_subplot(n, m, i + 1)

            for j in range(n_reps):
                x = self.time
                y = data1[:, mask][:, j]
                # ax.plot(x, y, color=self.color_group1)
                x_processed, y_processed = self.get_trend(y, method="moving_average", window=144)
                # ax.plot(x_processed, y_processed, color="black", linestyle="--", linewidth=0.8)
                y_detrended = y - y_processed
                ax.plot(x_processed, y_detrended, color=self.color_group1)
                #  select only valid points (non-NaN) for fitting
                valid_mask = ~np.isnan(y_detrended)
                x_valid = x_processed[valid_mask]
                y_valid = y_detrended[valid_mask]
                #  use only with t_lower and t_upper parameters
                range_mask = (x_valid >= self.t_lower) & (x_valid <= self.t_upper)
                x_fit = x_valid[range_mask]
                y_fit = y_valid[range_mask]

                results, t_test, model_predictions_cosine = self.fit_cosinor_24(x_fit, y_fit)
                ax.plot(t_test, model_predictions_cosine, color="k", linestyle="--")

            ax.set_title(f"{ids1[i]} (n={n_reps})")
            ax.set_xlabel("Time")
            ax.set_ylabel("Expression")

        plt.tight_layout()
        plt.show()


# %%
file = (
    "/Users/phillips/Documents/Agata circadian/Bioluminescence data Aug 2024/Bioluminescence_all_raw_detrended_v2.xlsx"
)
df_data = pd.read_excel(file, sheet_name="Detrended", index_col=0, skiprows=2, header=None)

participant_id = df_data.iloc[0, :].astype(str)
replicate = df_data.iloc[1, :].astype(int)
group = df_data.iloc[2, :].astype(str)
time = df_data.index[3:].to_numpy(dtype=float)

time_rows = df_data.iloc[3:].apply(pd.to_numeric)
time_series = time_rows.iloc[:, 0].to_numpy(dtype=float)

dataset = LiveCellDataset(
    ids=participant_id.tolist(),
    group=group.tolist(),
    replicate=replicate.tolist(),
    time_series=time_rows.to_numpy(dtype=float),
    time=time,
    group1_label="ND",
    group2_label="T2D",
)

dataset.plot_group_data("group2", m=5)

# %%

# cosinor_analysis = CosinorAnalysis(dataset=dataset)
# cosinor_analysis.plot_group_data(m=5)

cosinor_analysis = CosinorAnalysis(
    ids=participant_id.tolist(),
    group=group.tolist(),
    replicate=replicate.tolist(),
    time_series=time_rows.to_numpy(dtype=float),
    time=time,
    group1_label="ND",
    group2_label="T2D",
)

cosinor_analysis.fit_cosinor(m=5)

# %%
