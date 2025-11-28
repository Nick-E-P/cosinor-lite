from __future__ import annotations

import re
import tempfile
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cosinor_lite.livecell_cosinor_analysis import CosinorAnalysis
from cosinor_lite.livecell_dataset import LiveCellDataset
from cosinor_lite.omics_dataset import OmicsDataset
from cosinor_lite.omics_differential_rhytmicity import DifferentialRhythmicity, OmicsHeatmap

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

# --- (your rcParams / styles as-is) ---

APP_DIR = Path(__file__).parent
bioluminescence_file = str(APP_DIR / "data" / "bioluminescence_example.csv")
cytokine_file = str(APP_DIR / "data" / "cytokine_example.csv")
omics_example = str(APP_DIR / "data" / "GSE95156_Alpha_Beta.txt")
method_img = str(APP_DIR / "images" / "live_cell_fitting-01.png")
model_selection_img = str(APP_DIR / "images" / "model_selection.png")


# def create_cosinor_inputs(
#     ids,
#     group,
#     replicate,
#     time_series,
#     time,
#     group1_label,
#     group2_label,
# ):
#     return f"""
#     CosinorAnalysis(
#     ids={ids},
#     group={group},
#     replicate={replicate},
#     time_series={time_series},
#     time={time},
#     group1_label="{group1_label}",
#     group2_label="{group2_label}"
#     )"""


# @dataclass(config=ConfigDict(arbitrary_types_allowed=True))
# class LiveCellDataset:
#     ids: list[str]
#     group: list[str]
#     replicate: list[int]
#     time_series: np.ndarray  # shape: [T, N]
#     time: np.ndarray  # shape: [T]

#     group1_label: str = "group1"
#     group2_label: str = "group2"

#     color_group1: str = "tab:blue"
#     color_group2: str = "tab:orange"

#     # --- validators ---
#     @field_validator("time_series", mode="before")
#     @classmethod
#     def _to_2d_f64(cls, v: object) -> np.ndarray:
#         a = np.asarray(v, dtype=np.float64)
#         if a.ndim != 2:
#             raise ValueError("expected 2D array")
#         return a

#     @model_validator(mode="after")
#     def _check_columns(self) -> LiveCellDataset:  # <-- quoted or use __future__ above
#         if len(self.ids) != self.time_series.shape[1]:
#             raise ValueError(
#                 "Length of ids must match number of columns in time_series",
#             )
#         if len(self.group) != self.time_series.shape[1]:
#             raise ValueError(
#                 "Length of group must match number of columns in time_series",
#             )
#         if len(self.replicate) != self.time_series.shape[1]:
#             raise ValueError(
#                 "Length of replicate must match number of columns in time_series",
#             )
#         # unique_groups = set(self.group)
#         # if len(unique_groups) != 2:
#         #     raise ValueError(
#         #         "There must be exactly two unique groups in 'group' column"
#         #     )
#         return self

#     def get_group1_ids_replicates_data(self):
#         mask = np.array(self.group) == self.group1_label
#         ids = list(np.array(self.ids)[mask])
#         replicates = list(np.array(self.replicate)[mask])
#         data = self.time_series[:, mask]
#         return ids, replicates, data

#     def get_group2_ids_replicates_data(self):
#         mask = np.array(self.group) == self.group2_label
#         ids = list(np.array(self.ids)[mask])
#         replicates = list(np.array(self.replicate)[mask])
#         data = self.time_series[:, mask]
#         return ids, replicates, data

#     def linear_trend(self, x, y):
#         model = sm.OLS(y, sm.add_constant(x)).fit()
#         linear_fit = model.predict(sm.add_constant(x))
#         return x, linear_fit

#     def poly2_trend(self, x, y):
#         coeffs = np.polyfit(x, y, 2)
#         poly_fit = np.polyval(coeffs, x)
#         return x, poly_fit

#     def moving_average_trend(self, x, y, window: int = 5):
#         if window < 1:
#             raise ValueError("Window size must be at least 1.")
#         y_series = pd.Series(y)
#         ma_fit = y_series.rolling(window=window, center=True).mean().to_numpy()

#         # return x, ma_fit
#         good = np.isfinite(x) & np.isfinite(ma_fit)
#         return x[good], ma_fit[good]

#     def get_trend(self, x, y, method: str = "linear", window: int = 5):
#         if method == "none":
#             return np.asarray(x, float), np.zeros_like(np.asarray(y, float))
#         if method == "linear":
#             return self.linear_trend(x, y)
#         if method == "poly2":
#             return self.poly2_trend(x, y)
#         if method == "moving_average":
#             return self.moving_average_trend(x, y, window=window)
#         raise ValueError(f"Unknown detrending method: {method}")

#     def plot_group_data(
#         self,
#         group: str,
#         method: str = "linear",
#         window: int = 5,
#         m: int = 5,
#         plot_style: str = "scatter",
#     ):
#         if group == "group1":
#             ids, replicates, data = self.get_group1_ids_replicates_data()
#             color = self.color_group1
#             group_label = self.group1_label
#         elif group == "group2":
#             ids, replicates, data = self.get_group2_ids_replicates_data()
#             color = self.color_group2
#             group_label = self.group2_label
#         else:
#             raise ValueError("group must be 'group1' or 'group2'")

#         n_group = len(np.unique(ids))
#         n_cols = m
#         n_rows = int(np.ceil(n_group / n_cols))

#         study_list = np.unique(ids).tolist()
#         fig = plt.figure(figsize=(5 * n_cols / 2.54, 5 * n_rows / 2.54))

#         for i, id_curr in enumerate(study_list):
#             mask = np.array(ids) == id_curr
#             n_reps = int(np.sum(mask))
#             ax = fig.add_subplot(n_rows, n_cols, i + 1)

#             for j in range(n_reps):
#                 x = self.time
#                 y = data[:, mask][:, j]

#                 if plot_style == "scatter":
#                     ax.scatter(x, y, s=4, alpha=0.8, color=color)
#                 else:
#                     ax.plot(x, y, color=color)

#                 valid_mask = ~np.isnan(y)
#                 x_fit = x[valid_mask]
#                 y_fit = y[valid_mask]

#                 x_processed, y_processed = self.get_trend(
#                     x_fit,
#                     y_fit,
#                     method=method,
#                     window=window,
#                 )
#                 if method != "none":
#                     ax.plot(
#                         x_processed,
#                         y_processed,
#                         color="black",
#                         linestyle="--",
#                         linewidth=0.8,
#                     )

#             ax.set_title(f"ID: {id_curr} (n={n_reps}) - {group_label}")
#             ax.set_xlabel("Time (h)")
#             if i % n_cols == 0:
#                 ax.set_ylabel("Expression")

#         plt.tight_layout()

#         # Save once here and return both fig + path so Gradio can preview & download
#         tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
#         fig.savefig(tmpfile.name)
#         # do NOT plt.show() in Gradio; it will try to pop a window
#         return fig, tmpfile.name


# def constant_model(x, mesor):
#     # mesor = params
#     return mesor


# def cosine_model_24(x, amplitude, acrophase, mesor):
#     period = 24.0
#     return amplitude * np.cos(2 * np.pi * (x - acrophase) / period) + mesor


# def cosine_model_free_period(x, amplitude, acrophase, period, mesor):
#     return amplitude * np.cos(2 * np.pi * (x - acrophase) / period) + mesor


# def cosine_model_damped(x, amplitude, damp, acrophase, period, mesor):
#     return amplitude * np.exp(-damp * x) * np.cos(2 * np.pi * (x - acrophase) / period) + mesor


# def _metrics(y_true, y_pred, p: int):
#     y_true = np.asarray(y_true, float)
#     y_pred = np.asarray(y_pred, float)
#     n = y_true.size
#     rss = np.sum((y_true - y_pred) ** 2)
#     sst = np.sum((y_true - np.mean(y_true)) ** 2)
#     r2 = np.nan
#     r2_adj = np.nan
#     if sst > 0:
#         r2 = 1.0 - (rss / sst)
#         if n > p and n > 1:
#             r2_adj = 1.0 - (rss / (n - p)) / (sst / (n - 1))
#     return rss, r2, r2_adj


# def _sanitize_xy(x, y, min_points: int = 4):
#     x = np.asarray(x, float)
#     y = np.asarray(y, float)
#     ok = np.isfinite(x) & np.isfinite(y)
#     x, y = x[ok], y[ok]
#     order = np.argsort(x)
#     x, y = x[order], y[order]
#     if y.size < min_points:
#         raise ValueError(
#             f"Not enough valid points after cleaning (need ≥{min_points}, got {y.size}).",
#         )
#     return x, y


# class CosinorAnalysis(LiveCellDataset):
#     def __init__(
#         self,
#         *args,
#         period: float = 24.0,
#         method: str = "ols",
#         t_lower=0.0,
#         t_upper=720.0,
#         **kwargs,
#     ):
#         # initialize the parent dataset
#         super().__init__(*args, **kwargs)

#         # store cosinor-specific parameters
#         self.t_lower = t_lower
#         self.t_upper = t_upper
#         self.period = period
#         self.method = method

#     def fit_cosinor_24(
#         self,
#         x: np.ndarray,
#         y: np.ndarray,
#     ) -> tuple[dict, np.ndarray, np.ndarray]:
#         x, y = _sanitize_xy(x, y)

#         # 1) Constant (null) model
#         p0_const = [float(np.mean(y))]
#         params_const, _ = curve_fit(constant_model, x, y, p0=p0_const)
#         yhat_const = constant_model(x, *params_const)  # unpack for safety

#         # 2) 24h cosinor
#         p0_cos = [
#             float(np.std(y)),
#             0.0,
#             float(np.mean(y)),
#         ]  # [amplitude, acrophase, mesor]
#         params_cos, _ = curve_fit(cosine_model_24, x, y, p0=p0_cos)
#         amp_fit, acro_fit, mesor_fit = params_cos
#         yhat_cos = cosine_model_24(x, amp_fit, acro_fit, mesor_fit)

#         # Metrics
#         rss_cos, r2, r2_adj = _metrics(y, yhat_cos, p=3)
#         rss_const, _, _ = _metrics(y, yhat_const, p=1)

#         # F-test (nested)
#         n = len(y)
#         p1, p2 = 1, 3
#         num = max(rss_const - rss_cos, 0.0)
#         den = max(rss_cos, 1e-12)
#         f_stat = (num / (p2 - p1)) / (den / max(n - p2, 1))
#         p_val = np.nan
#         if n > p2:
#             p_val = f_dist.sf(f_stat, p2 - p1, n - p2)

#         # Acrophase as time-of-peak over [0, 24]
#         t_test_acro = np.linspace(0.0, 24.0, 1440)
#         y_test_acro = cosine_model_24(t_test_acro, amp_fit, acro_fit, mesor_fit)
#         amplitude = abs(amp_fit)
#         acrophase = t_test_acro[int(np.argmax(y_test_acro))]
#         mesor = mesor_fit

#         # Smooth curve over observed range
#         t_test = np.linspace(float(x[0]), float(x[-1]), 1440)
#         y_test = cosine_model_24(t_test, amp_fit, acro_fit, mesor_fit)

#         results = {
#             "mesor": mesor,
#             "amplitude": amplitude,
#             "acrophase": acrophase,
#             "p-val osc": p_val,
#             "r2": r2,
#             "r2_adj": r2_adj,
#         }
#         return results, t_test, y_test

#     def fit_cosinor_free_period(
#         self,
#         x: np.ndarray,
#         y: np.ndarray,
#     ) -> tuple[dict, np.ndarray, np.ndarray]:
#         x, y = _sanitize_xy(x, y)

#         # Constant (null)
#         p0_const = [float(np.mean(y))]
#         params_const, _ = curve_fit(constant_model, x, y, p0=p0_const)
#         yhat_const = constant_model(x, *params_const)

#         # Warm start with 24h model
#         p0_24 = [float(np.std(y)), 0.0, float(np.mean(y))]
#         params_24, _ = curve_fit(cosine_model_24, x, y, p0=p0_24)
#         amp0, acro0, mesor0 = params_24
#         p0_free = [amp0, acro0, 24.0, mesor0]  # [A, phase, period, mesor]

#         bounds = ([-np.inf, -np.inf, 20.0, -np.inf], [np.inf, np.inf, 28.0, np.inf])
#         params_free, _ = curve_fit(
#             cosine_model_free_period,
#             x,
#             y,
#             p0=p0_free,
#             bounds=bounds,
#         )
#         amp_fit, acro_fit, period_fit, mesor_fit = params_free
#         yhat_free = cosine_model_free_period(
#             x,
#             amp_fit,
#             acro_fit,
#             period_fit,
#             mesor_fit,
#         )

#         # Metrics
#         rss_free, r2, r2_adj = _metrics(y, yhat_free, p=4)
#         rss_const, _, _ = _metrics(y, yhat_const, p=1)

#         # F-test
#         n = len(y)
#         p1, p2 = 1, 4
#         num = max(rss_const - rss_free, 0.0)
#         den = max(rss_free, 1e-12)
#         f_stat = (num / (p2 - p1)) / (den / max(n - p2, 1))
#         p_val = np.nan
#         if n > p2:
#             p_val = f_dist.sf(f_stat, p2 - p1, n - p2)

#         # Acrophase over one fitted period (not fixed 24 h)
#         t_test_acro = np.linspace(0.0, float(period_fit), 2000)
#         y_test_acro = cosine_model_free_period(
#             t_test_acro,
#             amp_fit,
#             acro_fit,
#             period_fit,
#             mesor_fit,
#         )
#         amplitude = abs(amp_fit)
#         acrophase = t_test_acro[int(np.argmax(y_test_acro))]
#         mesor = mesor_fit
#         period = float(period_fit)

#         # Smooth curve over observed range
#         t_test = np.linspace(float(x[0]), float(x[-1]), 1440)
#         y_test = cosine_model_free_period(
#             t_test,
#             amp_fit,
#             acro_fit,
#             period_fit,
#             mesor_fit,
#         )

#         results = {
#             "mesor": mesor,
#             "amplitude": amplitude,
#             "acrophase": acrophase,
#             "period": period,
#             "p-val osc": p_val,
#             "r2": r2,
#             "r2_adj": r2_adj,
#         }
#         return results, t_test, y_test

#     def fit_cosinor_damped(
#         self,
#         x: np.ndarray,
#         y: np.ndarray,
#     ) -> tuple[dict, np.ndarray, np.ndarray]:
#         x, y = _sanitize_xy(x, y)

#         # Constant (null)
#         p0_const = [float(np.mean(y))]
#         params_const, _ = curve_fit(constant_model, x, y, p0=p0_const)
#         yhat_const = constant_model(x, *params_const)

#         # Warm start with 24h (non-damped)
#         p0_24 = [float(np.std(y)), 0.0, float(np.mean(y))]
#         params_24, _ = curve_fit(cosine_model_24, x, y, p0=p0_24)
#         amp0, acro0, mesor0 = params_24

#         # Damped: [A, damp, phase, period, mesor]
#         p0_damped = [amp0, 0.01, acro0, 24.0, mesor0]
#         bounds = (
#             [-np.inf, 0.0, -np.inf, 20.0, -np.inf],
#             [np.inf, np.inf, np.inf, 28.0, np.inf],
#         )
#         params_damped, _ = curve_fit(
#             cosine_model_damped,
#             x,
#             y,
#             p0=p0_damped,
#             bounds=bounds,
#         )
#         amp_fit, damp_fit, acro_fit, period_fit, mesor_fit = params_damped
#         yhat_damped = cosine_model_damped(
#             x,
#             amp_fit,
#             damp_fit,
#             acro_fit,
#             period_fit,
#             mesor_fit,
#         )

#         # Metrics
#         rss_damped, r2, r2_adj = _metrics(y, yhat_damped, p=5)
#         rss_const, _, _ = _metrics(y, yhat_const, p=1)

#         # F-test
#         n = len(y)
#         p1, p2 = 1, 5
#         num = max(rss_const - rss_damped, 0.0)
#         den = max(rss_damped, 1e-12)
#         f_stat = (num / (p2 - p1)) / (den / max(n - p2, 1))
#         p_val = np.nan
#         if n > p2:
#             p_val = f_dist.sf(f_stat, p2 - p1, n - p2)

#         # Acrophase over one fitted period (using damped model)
#         t_test_acro = np.linspace(0.0, float(period_fit), 1440)
#         y_test_acro = cosine_model_damped(
#             t_test_acro,
#             amp_fit,
#             damp_fit,
#             acro_fit,
#             period_fit,
#             mesor_fit,
#         )
#         amplitude = abs(amp_fit)
#         acrophase = t_test_acro[int(np.argmax(y_test_acro))]
#         mesor = mesor_fit
#         period = float(period_fit)
#         damp = float(damp_fit)

#         # Smooth curve over observed range
#         t_test = np.linspace(float(x[0]), float(x[-1]), 1440)
#         y_test = cosine_model_damped(
#             t_test,
#             amp_fit,
#             damp_fit,
#             acro_fit,
#             period_fit,
#             mesor_fit,
#         )

#         results = {
#             "mesor": mesor,
#             "amplitude": amplitude,
#             "acrophase": acrophase,
#             "period": period,
#             "damp": damp,
#             "p-val osc": p_val,
#             "r2": r2,
#             "r2_adj": r2_adj,
#         }
#         return results, t_test, y_test

#     def get_cosinor_fits(
#         self,
#         x: np.ndarray,
#         y: np.ndarray,
#         method: str = "cosinor_24",
#     ) -> tuple[dict, np.ndarray, np.ndarray]:
#         if method == "cosinor_24":
#             results, t_test, model_predictions_cosine = self.fit_cosinor_24(x, y)
#         elif method == "cosinor_free_period":
#             results, t_test, model_predictions_cosine = self.fit_cosinor_free_period(
#                 x,
#                 y,
#             )
#         elif method == "cosinor_damped":
#             results, t_test, model_predictions_cosine = self.fit_cosinor_damped(x, y)
#         else:
#             msg = f"Unknown cosine model: {method}"
#             raise ValueError(msg)

#         return results, t_test, model_predictions_cosine

#     def fit_cosinor(
#         self,
#         group: str,
#         method: str = "linear",
#         window: int = 5,
#         cosinor_model: str = "cosinor_24",
#         m: int = 5,
#         plot_style: str = "scatter",
#     ) -> pd.DataFrame:
#         if group == "group1":
#             ids, replicates, data = self.get_group1_ids_replicates_data()
#             color = self.color_group1
#             n_group = len(np.unique(ids))
#             group_label = self.group1_label
#         elif group == "group2":
#             ids, replicates, data = self.get_group2_ids_replicates_data()
#             color = self.color_group2
#             n_group = len(np.unique(ids))
#             group_label = self.group2_label
#         else:
#             msg = "group must be 'group1' or 'group2'"
#             raise ValueError(msg)

#         n = np.ceil(n_group / m).astype(int)

#         study_list = np.unique(ids).tolist()

#         fig = plt.figure(figsize=(5 * m / 2.54, 5 * n / 2.54))

#         to_export_list = []

#         for i, id_curr in enumerate(study_list):
#             mask = np.array(ids) == id_curr
#             n_reps = np.sum(mask)

#             ax = fig.add_subplot(n, m, i + 1)

#             for j in range(n_reps):
#                 exp_info = {"id": id_curr, "replicate": j + 1, "group": group_label}

#                 x = self.time
#                 y = data[:, mask][:, j]

#                 #  select only valid points (non-NaN) for fitting
#                 valid_mask = ~np.isnan(y)
#                 x_valid = x[valid_mask]
#                 y_valid = y[valid_mask]
#                 #  use only with t_lower and t_upper parameters
#                 range_mask = (x_valid >= self.t_lower) & (x_valid <= self.t_upper)
#                 x_fit = x_valid[range_mask]
#                 y_fit = y_valid[range_mask]

#                 x_processed, y_processed = self.get_trend(
#                     x_fit,
#                     y_fit,
#                     method=method,
#                     window=window,
#                 )
#                 y_detrended = y_fit - y_processed + np.mean(y_fit)
#                 # ax.plot(x_processed, y_detrended, color=self.color_group1)

#                 if plot_style == "scatter":
#                     ax.scatter(x_processed, y_detrended, s=4, alpha=0.8, color=color)
#                 else:
#                     ax.plot(x_processed, y_detrended, color=color)
#                 # #  select only valid points (non-NaN) for fitting
#                 # valid_mask = ~np.isnan(y_detrended)
#                 # x_valid = x_processed[valid_mask]
#                 # y_valid = y_detrended[valid_mask]
#                 # #  use only with t_lower and t_upper parameters
#                 # range_mask = (x_valid >= self.t_lower) & (x_valid <= self.t_upper)
#                 # x_fit = x_valid[range_mask]
#                 # y_fit = y_valid[range_mask]

#                 # results, t_test, model_predictions_cosine = self.fit_cosinor_24(x_fit, y_fit)
#                 results, t_test, model_predictions_cosine = self.get_cosinor_fits(
#                     x_processed,
#                     y_detrended,
#                     method=cosinor_model,
#                 )
#                 to_export_list.append({**exp_info, **results})
#                 ax.plot(t_test, model_predictions_cosine, color="k", linestyle="--")

#             ax.set_title(f"ID: {id_curr} (n={n_reps}) - {group_label}")
#             ax.set_xlabel("Time (h)")
#             if i % m == 0:
#                 ax.set_ylabel("Expression")

#         plt.tight_layout()

#         df_export = pd.DataFrame(to_export_list)
#         tmp1 = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
#         df_export.to_csv(tmp1.name, index=False)

#         tmp2 = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
#         fig.savefig(tmp2.name)

#         return df_export, tmp1.name, fig, tmp2.name


# @dataclass(config=ConfigDict(arbitrary_types_allowed=True))
# class OmicsDataset:
#     df: pd.DataFrame

#     columns_cond1: list[str]
#     columns_cond2: list[str]
#     t_cond1: np.ndarray
#     t_cond2: np.ndarray
#     cond1_label: str = "cond1"
#     cond2_label: str = "cond2"

#     deduplicate_on_init: bool = False

#     # --- validators ---
#     @field_validator("t_cond1", "t_cond2", mode="before")
#     @classmethod
#     def _to_1d_f64(cls, v: object) -> np.ndarray:  # ← return type fixes ANN206
#         a = np.asarray(v, dtype=np.float64)
#         if a.ndim != 1:
#             text: str = "expected 1D array"
#             raise ValueError(text)
#         return a

#     @model_validator(mode="after")
#     def _check_columns(self) -> Self:
#         missing = [c for c in (self.columns_cond1 + self.columns_cond2) if c not in self.df.columns]
#         if missing:
#             text = f"Missing columns: {missing}"  # satisfies EM101 (no string literal in raise)
#             raise ValueError(text)
#         return self

#     def __post_init__(self) -> None:
#         self.add_detected_timepoint_counts()
#         self.add_mean_expression()
#         self.add_number_detected()
#         if self.deduplicate_on_init:
#             self.deduplicate_genes()

#     def detected_timepoint_counts(self, cond: str) -> list[int]:
#         """
#         Count number of timepoints with detected values for each gene.

#         Args:
#             cond (str): "cond1" or "cond2"

#         Returns:
#             list[int]: List of counts for each gene.

#         """
#         if cond == "cond1":
#             y = self.df[self.columns_cond1]
#             t = self.t_cond1
#         elif cond == "cond2":
#             y = self.df[self.columns_cond2]
#             t = self.t_cond2
#         else:
#             text = f"Invalid condition: {cond}"  # satisfies EM101 (no string literal in raise)
#             raise ValueError(text)

#         # y-like frame, boolean mask of non-NaN
#         mask = y.notna()  # shape: (n_rows, n_cols)

#         # t_beta must be length n_cols and aligned to those columns
#         # Group columns by time, check if ANY non-NaN per group, then count groups per row
#         detected_timepoints = (
#             mask.T.groupby(t)
#             .any()  # (n_rows, n_unique_times) booleans
#             .T.sum(axis=1)  # per-row counts
#             .to_numpy()
#         )

#         return detected_timepoints

#     def add_detected_timepoint_counts(self) -> None:
#         """Add two columns to self.df with counts for cond1 and cond2."""
#         self.df["detected_cond1"] = self.detected_timepoint_counts("cond1")
#         self.df["detected_cond2"] = self.detected_timepoint_counts("cond2")

#     def add_mean_expression(self) -> None:
#         """Add two columns to self.df with mean expression for cond1 and cond2."""
#         self.df["mean_cond1"] = self.df[self.columns_cond1].mean(axis=1)
#         self.df["mean_cond2"] = self.df[self.columns_cond2].mean(axis=1)

#     def add_number_detected(self) -> None:
#         """Add two columns to self.df with number of detected values for cond1 and cond2."""
#         self.df["num_detected_cond1"] = self.df[self.columns_cond1].count(axis=1)
#         self.df["num_detected_cond2"] = self.df[self.columns_cond2].count(axis=1)

#     def deduplicate_genes(self) -> None:
#         """Deduplicate self.df by 'Genes', keeping entry with highest total mean expression."""
#         if not {"mean_cond1", "mean_cond2"}.issubset(self.df):
#             self.add_mean_expression()

#         self.df = (
#             self.df.assign(total_mean=self.df["mean_cond1"] + self.df["mean_cond2"])
#             .sort_values("total_mean", ascending=False)
#             .drop_duplicates(subset="Genes", keep="first")
#             .drop(columns="total_mean")
#         )

#     def add_is_expressed(
#         self,
#         *,
#         detected_min: int | None = None,
#         mean_min: float | None = None,
#         num_detected_min: int | None = None,
#     ) -> None:
#         """Add is_expressed_cond1/cond2 based on thresholds."""
#         # Ensure prerequisite columns exist
#         if not {"detected_cond1", "detected_cond2"}.issubset(self.df):
#             self.add_detected_timepoint_counts()
#         if not {"mean_cond1", "mean_cond2"}.issubset(self.df):
#             self.add_mean_expression()
#         if not {"num_detected_cond1", "num_detected_cond2"}.issubset(self.df):
#             self.add_number_detected()

#         def _mask(which: Literal[cond1, cond2]) -> pd.Series:
#             # start with all-True masks
#             m_detected = pd.Series(True, index=self.df.index)
#             m_mean = pd.Series(True, index=self.df.index)
#             m_num = pd.Series(True, index=self.df.index)

#             if detected_min is not None:
#                 m_detected = self.df[f"detected_{which}"] >= detected_min
#             if mean_min is not None:
#                 m_mean = self.df[f"mean_{which}"] >= mean_min
#             if num_detected_min is not None:
#                 m_num = self.df[f"num_detected_{which}"] >= num_detected_min

#             return m_detected & m_mean & m_num

#         self.df["is_expressed_cond1"] = _mask("cond1")
#         self.df["is_expressed_cond2"] = _mask("cond2")

#     def expression_histogram(self, bins: int = 20) -> None:
#         """Plot histogram of mean expression for cond1 and cond2."""
#         print(plt.rcParams["font.size"])
#         fig = plt.figure(figsize=(6 / 2.54, 12 / 2.54))
#         plt.subplot(2, 1, 1)
#         plt.hist(self.df["mean_cond1"].to_numpy().flatten(), bins=bins)
#         plt.xlabel("Mean Expression")
#         plt.ylabel("Frequency")
#         plt.title(f"Mean expression ({self.cond1_label})")
#         plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=10))
#         plt.subplot(2, 1, 2)
#         plt.hist(self.df["mean_cond2"].to_numpy().flatten(), bins=bins)
#         plt.xlabel("Mean Expression")
#         plt.ylabel("Density")
#         plt.title(f"Mean expression ({self.cond2_label})")
#         plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=10))
#         plt.tight_layout()

#         return fig

#     def replicate_scatterplot(self, sample1: str, sample2: str) -> None:
#         """Scatterplot of two replicates."""
#         if sample1 not in self.df.columns or sample2 not in self.df.columns:
#             text = f"Samples {sample1} and/or {sample2} not in DataFrame columns."
#             raise ValueError(text)

#         fig = plt.figure(figsize=(8 / 2.54, 8 / 2.54))
#         x: np.ndarray = self.df[sample1].to_numpy().flatten()
#         y: np.ndarray = self.df[sample2].to_numpy().flatten()
#         r_pearson: float = np.corrcoef(x, y)[0, 1]
#         r_spearman: float = spearmanr(x, y).statistic
#         plt.scatter(x, y, alpha=0.1, s=4)
#         plt.xlabel(sample1)
#         plt.ylabel(sample2)
#         plt.title(f"Pearson R = {r_pearson:.2f}, Spearman R = {r_spearman:.2f}")
#         plt.axis("equal")
#         plt.plot(
#             [x.min(), x.max()],
#             [x.min(), x.max()],
#             color="grey",
#             linestyle="--",
#             alpha=0.8,
#         )
#         plt.tight_layout()

#         return fig


# W: float = 2 * np.pi / 24.0
# RAD2H: float = 24.0 / (2.0 * np.pi)


# def phase_from_ab(a: float, b: float) -> float:
#     return (np.arctan2(b, a) * RAD2H) % 24.0


# def amp_from_ab(a: float, b: float) -> float:
#     return float(np.hypot(a, b))


# def bic(llf: float, k: int, n: int) -> float:
#     return k * np.log(n) - 2.0 * llf


# @dataclass
# class ModelResult:
#     name: int
#     llf: float
#     bic: float
#     alpha_phase: float
#     alpha_amp: float
#     beta_phase: float
#     beta_amp: float


# @dataclass
# class ModelResultOneCondition:
#     name: int
#     llf: float
#     bic: float
#     phase: float
#     amp: float


# class BaseModel(ABC):
#     name: str
#     k: int
#     formula: str

#     def fit(self, df: pd.DataFrame) -> ModelResult:
#         # df must already contain: y, constant, cos_wt, sin_wt, is_alpha, is_beta
#         model = smf.ols(self.formula, data=df).fit()
#         n = len(df)
#         alpha_phase, alpha_amp, beta_phase, beta_amp = self.extract(model.params)
#         return ModelResult(
#             name=self.name,
#             llf=model.llf,
#             bic=bic(model.llf, self.k, n),
#             alpha_phase=alpha_phase,
#             alpha_amp=alpha_amp,
#             beta_phase=beta_phase,
#             beta_amp=beta_amp,
#         )

#     @abstractmethod
#     def extract(self, params: pd.Series) -> tuple[float, float, float, float]: ...


# class BaseModelOneCondition(ABC):
#     name: str
#     k: int
#     formula: str

#     def fit(self, df: pd.DataFrame) -> ModelResultOneCondition:
#         # df must already contain: y, constant, cos_wt, sin_wt, is_alpha, is_beta
#         model = smf.ols(self.formula, data=df).fit()
#         n = len(df)
#         phase, amp = self.extract(model.params)
#         return ModelResultOneCondition(
#             name=self.name,
#             llf=model.llf,
#             bic=bic(model.llf, self.k, n),
#             phase=phase,
#             amp=amp,
#         )

#     @abstractmethod
#     def extract(self, params: pd.Series) -> tuple[float, float]: ...


# # ----- Five concrete models -----


# @dataclass
# class M0:
#     name: int = 0
#     alpha_phase: float = np.nan
#     alpha_amp: float = np.nan
#     beta_phase: float = np.nan
#     beta_amp: float = np.nan
#     amp: float = np.nan
#     phase: float = np.nan
#     bic: float = np.nan


# class M1(BaseModel):
#     name: int = 1
#     k: int = 3
#     formula: str = "y ~ is_alpha:constant + is_beta:constant -1"

#     def extract(self, params: pd.Series) -> tuple[float, float, float, float]:
#         # No rhythmic terms → phases/amps are NaN
#         return (np.nan, np.nan, np.nan, np.nan)


# class M1OneCondition(BaseModelOneCondition):
#     name: int = 1
#     k: int = 1
#     formula: str = "y ~ 1"

#     def extract(self, params: pd.Series) -> tuple[float, float]:
#         # No rhythmic terms → phases/amps are NaN
#         return (np.nan, np.nan)


# class M2(BaseModel):
#     name: int = 2
#     k: int = 5
#     formula: str = "y ~ is_alpha:constant + is_beta:constant + is_beta:cos_wt + is_beta:sin_wt -1"

#     def extract(self, params: pd.Series) -> tuple[float, float, float, float]:
#         # beta has rhythmic terms, alpha does not
#         a: float = params["is_beta:cos_wt"]
#         b: float = params["is_beta:sin_wt"]
#         beta_phase: float = phase_from_ab(a, b)
#         beta_amp: float = amp_from_ab(a, b)
#         return (np.nan, np.nan, beta_phase, beta_amp)


# class M3(BaseModel):
#     name: int = 3
#     k: int = 5
#     formula: str = "y ~ is_alpha:constant + is_beta:constant + is_alpha:cos_wt + is_alpha:sin_wt -1"

#     def extract(self, params: pd.Series) -> tuple[float, float, float, float]:
#         # alpha has rhythmic terms, beta does not
#         a: float = params["is_alpha:cos_wt"]
#         b: float = params["is_alpha:sin_wt"]
#         alpha_phase: float = phase_from_ab(a, b)
#         alpha_amp: float = amp_from_ab(a, b)
#         return (alpha_phase, alpha_amp, np.nan, np.nan)


# class MOscOneCondition(BaseModelOneCondition):
#     name: int = 3
#     k: int = 3
#     formula: str = "y ~ 1 + cos_wt + sin_wt"

#     def extract(self, params: pd.Series) -> tuple[float, float]:
#         # alpha has rhythmic terms, beta does not
#         a: float = params["cos_wt"]
#         b: float = params["sin_wt"]
#         phase: float = phase_from_ab(a, b)
#         amp: float = amp_from_ab(a, b)
#         return (phase, amp)


# class M4(BaseModel):
#     name: int = 4
#     k: int = 5
#     formula: str = "y ~ is_alpha:constant + is_beta:constant + cos_wt + sin_wt -1"

#     def extract(self, params: pd.Series) -> tuple[float, float, float, float]:
#         # shared rhythmic terms for both alpha and beta
#         a: float = params["cos_wt"]
#         b: float = params["sin_wt"]
#         ph: float = phase_from_ab(a, b)
#         am: float = amp_from_ab(a, b)
#         return (ph, am, ph, am)


# class M5(BaseModel):
#     name: int = 5
#     k: int = 7
#     formula: str = (
#         "y ~ is_alpha:constant + is_beta:constant + "
#         "is_alpha:cos_wt + is_alpha:sin_wt + is_beta:cos_wt + is_beta:sin_wt -1"
#     )

#     def extract(self, params: pd.Series) -> tuple[float, float, float, float]:
#         a_cond1: float = params["is_alpha:cos_wt"]
#         b_cond1: float = params["is_alpha:sin_wt"]
#         a_cond2: float = params["is_beta:cos_wt"]
#         b_cond2: float = params["is_beta:sin_wt"]

#         alpha_phase: float = phase_from_ab(a_cond1, b_cond1)
#         alpha_amp: float = amp_from_ab(a_cond1, b_cond1)
#         beta_phase: float = phase_from_ab(a_cond2, b_cond2)
#         beta_amp: float = amp_from_ab(a_cond2, b_cond2)
#         return (alpha_phase, alpha_amp, beta_phase, beta_amp)


# # ----- Runner utilities -----

# MODELS: tuple[BaseModel, ...] = (M1(), M2(), M3(), M4(), M5())
# MODELS_ONE_CONDITION: tuple[BaseModelOneCondition, ...] = (
#     M1OneCondition(),
#     MOscOneCondition(),
# )


# def akaike_weights_from_bics(bics: np.ndarray) -> np.ndarray:
#     d: np.ndarray = bics - np.nanmin(bics)
#     w: np.ndarray = np.exp(-0.5 * d)
#     return w / np.nansum(w)


# def build_design(
#     alpha_vals: np.ndarray,
#     beta_vals: np.ndarray,
#     t_cond1: np.ndarray,
#     t_cond2: np.ndarray,
# ) -> pd.DataFrame:
#     df_cond1: pd.DataFrame = pd.DataFrame(
#         {"y": alpha_vals, "time": t_cond1, "dataset": "alpha"},
#     ).dropna()
#     df_cond2: pd.DataFrame = pd.DataFrame(
#         {"y": beta_vals, "time": t_cond2, "dataset": "beta"},
#     ).dropna()
#     df: pd.DataFrame = pd.concat([df_cond1, df_cond2], ignore_index=True)
#     df["constant"] = 1.0
#     df["cos_wt"] = np.cos(W * df["time"].to_numpy().astype(float))
#     df["sin_wt"] = np.sin(W * df["time"].to_numpy().astype(float))
#     df["is_alpha"] = (df["dataset"] == "alpha").astype(int)
#     df["is_beta"] = (df["dataset"] == "beta").astype(int)
#     return df


# def build_design_cond1(
#     alpha_vals: np.ndarray,
#     t_cond1: np.ndarray,
# ) -> pd.DataFrame:
#     df_cond1: pd.DataFrame = pd.DataFrame(
#         {"y": alpha_vals, "time": t_cond1, "dataset": "alpha"},
#     ).dropna()
#     df: pd.DataFrame = pd.concat([df_cond1], ignore_index=True)
#     df["constant"] = 1.0
#     df["cos_wt"] = np.cos(W * df["time"].to_numpy().astype(float))
#     df["sin_wt"] = np.sin(W * df["time"].to_numpy().astype(float))
#     return df


# def build_design_cond2(
#     beta_vals: np.ndarray,
#     t_cond2: np.ndarray,
# ) -> pd.DataFrame:
#     df_cond2: pd.DataFrame = pd.DataFrame(
#         {"y": beta_vals, "time": t_cond2, "dataset": "beta"},
#     ).dropna()
#     df: pd.DataFrame = pd.concat([df_cond2], ignore_index=True)
#     df["constant"] = 1.0
#     df["cos_wt"] = np.cos(W * df["time"].to_numpy().astype(float))
#     df["sin_wt"] = np.sin(W * df["time"].to_numpy().astype(float))
#     return df


# @dataclass(config=ConfigDict(arbitrary_types_allowed=True))
# class DifferentialRhythmicity:
#     dataset: OmicsDataset
#     BIC_cutoff: float = 0.5

#     @property
#     def df(self) -> pd.DataFrame:
#         return self.dataset.df

#     @property
#     def columns_cond1(self) -> list[str]:
#         return self.dataset.columns_cond1

#     @property
#     def columns_cond2(self) -> list[str]:
#         return self.dataset.columns_cond2

#     @property
#     def t_cond1(self) -> np.ndarray:
#         return self.dataset.t_cond1

#     @property
#     def t_cond2(self) -> np.ndarray:
#         return self.dataset.t_cond2

#     @property
#     def cond1_label(self) -> str:
#         return self.dataset.cond1_label

#     @property
#     def cond2_label(self) -> str:
#         return self.dataset.cond2_label

#     def rhythmic_genes_expressed_both(self, progress=gr.Progress()) -> pd.DataFrame:
#         df: pd.DataFrame = self.df
#         mask: pd.Series = (df["is_expressed_cond1"]) & (df["is_expressed_cond2"])
#         df_to_analyse: pd.DataFrame = df[mask].reset_index(drop=True)

#         rows: list[dict] = []
#         for gene, row in progress.tqdm(
#             df_to_analyse.set_index("Genes").iterrows(),
#             total=len(df_to_analyse),
#             desc="Fitting models to genes expressed in both conditions",
#         ):
#             alpha_vec: np.ndarray = row[self.columns_cond1].to_numpy(float)
#             beta_vec: np.ndarray = row[self.columns_cond2].to_numpy(float)
#             design: pd.DataFrame = build_design(
#                 alpha_vec,
#                 beta_vec,
#                 self.t_cond1,
#                 self.t_cond2,
#             )
#             results: list[BaseModel] = [m.fit(design) for m in MODELS]
#             bics: np.ndarray = np.array([r.bic for r in results])
#             weights: np.ndarray = akaike_weights_from_bics(bics)
#             if np.max(weights) < self.BIC_cutoff:
#                 best: M0 = M0()
#                 chosen_model_biw: float = np.nan
#                 model: int = 0
#             else:
#                 pick: int = int(np.argmax(weights))
#                 best: BaseModel = results[pick]
#                 chosen_model_biw: np.ndarray = weights[pick]
#                 model: int = best.name
#             rows.append(
#                 {
#                     "gene": gene,
#                     "model": model,
#                     "chosen_model_bicw": chosen_model_biw,
#                     # "weight": weights[pick],
#                     **{f"w_model{i}": weights[i - 1] for i in range(1, 6)},
#                     "alpha_phase": best.alpha_phase,
#                     "alpha_amp": best.alpha_amp,
#                     "beta_phase": best.beta_phase,
#                     "beta_amp": best.beta_amp,
#                 },
#             )

#         df_results: pd.DataFrame = pd.DataFrame(rows)
#         df_results["subclass"] = "c"

#         return df_results

#     def rhythmic_genes_expressed_cond1(self) -> pd.DataFrame:
#         df: pd.DataFrame = self.df
#         mask: pd.Series = (df["is_expressed_cond1"]) & ~(df["is_expressed_cond2"])
#         df_to_analyse: pd.DataFrame = df[mask].reset_index(drop=True)

#         rows: list[dict] = []
#         for gene, row in tqdm(
#             df_to_analyse.set_index("Genes").iterrows(),
#             total=len(df_to_analyse),
#             desc="Fitting models to genes expressed in cond1 only",
#         ):
#             alpha_vec: np.ndarray = row[self.columns_cond1].to_numpy(float)
#             design: pd.DataFrame = build_design_cond1(alpha_vec, self.t_cond1)
#             results: list[BaseModelOneCondition] = [m.fit(design) for m in MODELS_ONE_CONDITION]
#             bics: np.ndarray = np.array([r.bic for r in results])
#             weights: np.ndarray = akaike_weights_from_bics(bics)
#             if np.max(weights) < self.BIC_cutoff:
#                 best: M0 = M0()
#                 chosen_model_biw: float = np.nan
#                 model: int = 0
#             else:
#                 pick: int = int(np.argmax(weights))
#                 best: BaseModel = results[pick]
#                 chosen_model_biw: np.ndarray = weights[pick]
#                 model: int = [1, 3][pick]
#             rows.append(
#                 {
#                     "gene": gene,
#                     "model": model,
#                     "chosen_model_bicw": chosen_model_biw,
#                     # "weight": weights[pick],
#                     **{f"w_model{model}": weights[i - 1] for i, model in enumerate([1, 3])},
#                     "alpha_phase": best.phase,
#                     "alpha_amp": best.amp,
#                     "beta_phase": np.nan,
#                     "beta_amp": np.nan,
#                 },
#             )

#         df_results: pd.DataFrame = pd.DataFrame(rows)
#         df_results["subclass"] = "a"

#         return df_results

#     def rhythmic_genes_expressed_cond2(self) -> pd.DataFrame:
#         df: pd.DataFrame = self.df
#         mask: pd.Series = ~(df["is_expressed_cond1"]) & (df["is_expressed_cond2"])
#         df_to_analyse: pd.DataFrame = df[mask].reset_index(drop=True)

#         rows: list[dict] = []
#         for gene, row in tqdm(
#             df_to_analyse.set_index("Genes").iterrows(),
#             total=len(df_to_analyse),
#             desc="Fitting models to genes expressed in cond2 only",
#         ):
#             beta_vec: np.ndarray = row[self.columns_cond2].to_numpy(float)
#             design: pd.DataFrame = build_design_cond2(beta_vec, self.t_cond2)
#             results: list[BaseModelOneCondition] = [m.fit(design) for m in MODELS_ONE_CONDITION]
#             bics: np.ndarray = np.array([r.bic for r in results])
#             weights: np.ndarray = akaike_weights_from_bics(bics)
#             if np.max(weights) < self.BIC_cutoff:
#                 best: M0 = M0()
#                 chosen_model_biw: float = np.nan
#                 model: int = 0
#             else:
#                 pick: int = int(np.argmax(weights))
#                 best: BaseModel = results[pick]
#                 chosen_model_biw: np.ndarray = weights[pick]
#                 model: int = [1, 2][pick]
#             rows.append(
#                 {
#                     "gene": gene,
#                     "model": model,
#                     "chosen_model_bicw": chosen_model_biw,
#                     # "weight": weights[pick],
#                     **{f"w_model{model}": weights[i - 1] for i, model in enumerate([1, 2])},
#                     "alpha_phase": np.nan,
#                     "alpha_amp": np.nan,
#                     "beta_phase": best.phase,
#                     "beta_amp": best.amp,
#                 },
#             )

#         df_results: pd.DataFrame = pd.DataFrame(rows)
#         df_results["subclass"] = "b"

#         return df_results

#     def extract_all_circadian_params(self) -> pd.DataFrame:
#         rhythmic_analysis_expressed_both = self.rhythmic_genes_expressed_both()
#         rhythmic_analysis_cond1 = self.rhythmic_genes_expressed_cond1()

#         rhythmic_analysis_cond2 = self.rhythmic_genes_expressed_cond2()

#         results_total = pd.concat(
#             [
#                 rhythmic_analysis_expressed_both,
#                 rhythmic_analysis_cond1,
#                 rhythmic_analysis_cond2,
#             ],
#         )

#         df_pre_export = self.df.merge(
#             results_total,
#             left_on="Genes",
#             right_on="gene",
#             how="right",
#         )
#         column_list: list = [
#             "Genes",
#             *self.columns_cond1,
#             *self.columns_cond2,
#             "w_model1",
#             "w_model2",
#             "w_model3",
#             "w_model4",
#             "w_model5",
#             "model",
#             "subclass",
#             "mean_cond1",
#             "alpha_amp",
#             "alpha_phase",
#             "is_expressed_cond1",
#             "mean_cond2",
#             "beta_amp",
#             "beta_phase",
#             "is_expressed_cond2",
#         ]

#         df_export = df_pre_export[column_list]
#         return df_export


# @dataclass(config=ConfigDict(arbitrary_types_allowed=True))
# class OmicsHeatmap:
#     df: pd.DataFrame

#     columns_cond1: list[str]
#     columns_cond2: list[str]
#     t_cond1: np.ndarray
#     t_cond2: np.ndarray
#     cond1_label: str = "cond1"
#     cond2_label: str = "cond2"

#     show_unexpressed: bool = True

#     # --- validators ---
#     @field_validator("t_cond1", "t_cond2", mode="before")
#     @classmethod
#     def _to_1d_f64(cls, v: object) -> np.ndarray:  # ← return type fixes ANN206
#         a = np.asarray(v, dtype=np.float64)
#         if a.ndim != 1:
#             text: str = "expected 1D array"
#             raise ValueError(text)
#         return a

#     @model_validator(mode="after")
#     def _check_columns(self) -> Self:
#         missing = [c for c in (self.columns_cond1 + self.columns_cond2) if c not in self.df.columns]
#         if missing:
#             text = f"Missing columns: {missing}"  # satisfies EM101 (no string literal in raise)
#             raise ValueError(text)
#         return self

#     def timepoint_means(
#         self,
#         df: pd.DataFrame,
#         columns: list[str],
#         times: np.ndarray,
#     ) -> np.ndarray:
#         """
#         Compute mean expression across columns at each unique timepoint.

#         Args:
#             df: DataFrame with expression values.
#             columns: Subset of column names to use.
#             times: 1D array of timepoints, length must equal len(columns).

#         Returns:
#             np.ndarray of shape (n_genes, n_unique_times).

#         """
#         if len(columns) != len(times):
#             text: str = f"Length of columns ({len(columns)}) must match length of times ({len(times)})"
#             raise ValueError(text)

#         values: np.ndarray = df[columns].to_numpy()
#         unique_times: np.ndarray = np.unique(times)

#         result: np.ndarray = np.column_stack(
#             [values[:, times == t].mean(axis=1) for t in unique_times],
#         )

#         return result

#     def get_z_score(self, arr: np.ndarray) -> np.ndarray:
#         """Compute z-score normalization for each row in a 2D array."""
#         arr: np.ndarray = (arr - np.mean(arr, axis=1).reshape(-1, 1)) / np.where(
#             np.std(arr, axis=1).reshape(-1, 1) == 0,
#             1,
#             np.std(arr, axis=1).reshape(-1, 1),
#         )
#         return arr

#     def plot_heatmap(self, cmap: str = "bwr") -> None:
#         df: pd.DataFrame = self.df
#         df: pd.DataFrame = df.sort_values(by=["alpha_phase", "beta_phase"]).reset_index(
#             drop=True,
#         )

#         t_unique: np.ndarray = np.unique(self.t_cond1).astype(int)

#         mean_cond1: np.ndarray = self.timepoint_means(
#             df,
#             self.columns_cond1,
#             self.t_cond1,
#         )
#         mean_cond2: np.ndarray = self.timepoint_means(
#             df,
#             self.columns_cond2,
#             self.t_cond2,
#         )

#         z_cond1: np.ndarray = self.get_z_score(mean_cond1)
#         z_cond2: np.ndarray = self.get_z_score(mean_cond2)

#         total_rows: int = (df["model"].isin([2, 3, 4, 5])).sum()

#         m2: int = 2
#         m3: int = 3
#         m4: int = 4
#         m5: int = 5

#         n_m2a: int = ((df["model"] == m2) & (df["subclass"] == "b")).sum()
#         n_m2b: int = ((df["model"] == m2) & (df["subclass"] == "c")).sum()
#         n_m3a: int = ((df["model"] == m3) & (df["subclass"] == "a")).sum()
#         n_m3b: int = ((df["model"] == m3) & (df["subclass"] == "c")).sum()
#         n_m4: int = (df["model"] == m4).sum()
#         n_m5: int = (df["model"] == m5).sum()

#         fig, axes = plt.subplots(
#             nrows=6,
#             ncols=2,
#             gridspec_kw={
#                 "height_ratios": [
#                     n_m2a / total_rows,
#                     n_m2b / total_rows,
#                     n_m3a / total_rows,
#                     n_m3b / total_rows,
#                     n_m4 / total_rows,
#                     n_m5 / total_rows,
#                 ],
#                 "width_ratios": [1, 1],  # Equal column widths
#             },
#             figsize=(12 / 2.54, 18 / 2.54),  # Adjust figure size as needed
#         )

#         vmin_global = -2.5
#         vmax_global = 2.5

#         mask = (df["model"] == m2) & (df["subclass"] == "b")
#         z_cond1_filtered = z_cond1[mask.to_numpy()].copy()
#         if not self.show_unexpressed:
#             z_cond1_filtered[:] = 0
#         im1 = axes[0, 0].imshow(
#             z_cond1_filtered,
#             aspect="auto",
#             cmap=cmap,
#             vmin=vmin_global,
#             vmax=vmax_global,
#             rasterized=False,
#             interpolation="none",
#         )

#         axes[0, 0].set_title("Alpha cells")
#         axes[0, 0].set_xticks([])
#         axes[0, 0].set_yticks([])

#         # Add text to the left of the y-axis
#         axes[0, 0].text(
#             -0.2,
#             0.5,
#             "M2a",  # Coordinates: (x, y)
#             transform=axes[0, 0].transAxes,  # Use axis-relative coordinates
#             ha="center",
#             va="center",  # Align text
#             rotation=0,  # Rotate text vertically
#             fontsize=8,  # Adjust font size as needed
#         )

#         mask = (df["model"] == m2) & (df["subclass"] == "c")
#         z_cond1_filtered = z_cond1[mask.to_numpy()]
#         im2 = axes[1, 0].imshow(
#             z_cond1_filtered,
#             aspect="auto",
#             cmap=cmap,
#             vmin=vmin_global,
#             vmax=vmax_global,
#             rasterized=False,
#             interpolation="none",
#         )
#         axes[1, 0].set_xticks([])
#         axes[1, 0].set_yticks([])

#         # Add text to the left of the y-axis
#         axes[1, 0].text(
#             -0.2,
#             0.5,
#             "M2b",  # Coordinates: (x, y)
#             transform=axes[1, 0].transAxes,  # Use axis-relative coordinates
#             ha="center",
#             va="center",  # Align text
#             rotation=0,  # Rotate text vertically
#             fontsize=8,  # Adjust font size as needed
#         )
#         mask = (df["model"] == m3) & (df["subclass"] == "a")
#         z_cond1_filtered = z_cond1[mask.to_numpy()]
#         im2 = axes[2, 0].imshow(
#             z_cond1_filtered,
#             aspect="auto",
#             cmap=cmap,
#             vmin=vmin_global,
#             vmax=vmax_global,
#             rasterized=False,
#             interpolation="none",
#         )
#         axes[2, 0].set_xticks([])
#         axes[2, 0].set_yticks([])

#         # Add text to the left of the y-axis
#         axes[2, 0].text(
#             -0.2,
#             0.5,
#             "M3a",  # Coordinates: (x, y)
#             transform=axes[2, 0].transAxes,  # Use axis-relative coordinates
#             ha="center",
#             va="center",  # Align text
#             rotation=0,  # Rotate text vertically
#             fontsize=8,  # Adjust font size as needed
#         )
#         mask = (df["model"] == m3) & (df["subclass"] == "c")
#         z_cond1_filtered = z_cond1[mask.to_numpy()]
#         im3 = axes[3, 0].imshow(
#             z_cond1_filtered,
#             aspect="auto",
#             cmap=cmap,
#             vmin=vmin_global,
#             vmax=vmax_global,
#             rasterized=False,
#             interpolation="none",
#         )
#         axes[3, 0].set_xticks([])
#         axes[3, 0].set_yticks([])
#         # Add text to the left of the y-axis
#         axes[3, 0].text(
#             -0.2,
#             0.5,
#             "M3b",  # Coordinates: (x, y)
#             transform=axes[3, 0].transAxes,  # Use axis-relative coordinates
#             ha="center",
#             va="center",  # Align text
#             rotation=0,  # Rotate text vertically
#             fontsize=8,  # Adjust font size as needed
#         )

#         mask = df["model"] == m4
#         z_cond1_filtered = z_cond1[mask.to_numpy()]
#         im4 = axes[4, 0].imshow(
#             z_cond1_filtered,
#             aspect="auto",
#             cmap=cmap,
#             vmin=vmin_global,
#             vmax=vmax_global,
#             rasterized=False,
#             interpolation="none",
#         )
#         axes[4, 0].set_xticks([])
#         axes[4, 0].set_yticks([])

#         axes[4, 0].text(
#             -0.2,
#             0.5,
#             "M4",  # Coordinates: (x, y)
#             transform=axes[4, 0].transAxes,  # Use axis-relative coordinates
#             ha="center",
#             va="center",  # Align text
#             rotation=0,  # Rotate text vertically
#             fontsize=8,  # Adjust font size as needed
#         )

#         mask = df["model"] == m5
#         z_cond1_filtered = z_cond1[mask.to_numpy()]
#         im5 = axes[5, 0].imshow(
#             z_cond1_filtered,
#             aspect="auto",
#             cmap=cmap,
#             vmin=vmin_global,
#             vmax=vmax_global,
#             rasterized=False,
#             interpolation="none",
#         )

#         axes[5, 0].set_xticks(range(len(t_unique)), t_unique)
#         axes[5, 0].set_yticks([])
#         axes[5, 0].set_xlabel("Time(h)")

#         axes[5, 0].text(
#             -0.2,
#             0.5,
#             "M5",  # Coordinates: (x, y)
#             transform=axes[5, 0].transAxes,  # Use axis-relative coordinates
#             ha="center",
#             va="center",  # Align text
#             rotation=0,  # Rotate text vertically
#             fontsize=8,  # Adjust font size as needed
#         )

#         mask = (df["model"] == m2) & (df["subclass"] == "b")
#         z_cond2_filtered = z_cond2[mask.to_numpy()]
#         im6 = axes[0, 1].imshow(
#             z_cond2_filtered,
#             aspect="auto",
#             cmap=cmap,
#             vmin=vmin_global,
#             vmax=vmax_global,
#             rasterized=False,
#             interpolation="none",
#         )

#         axes[0, 1].set_title("Beta cells")
#         axes[0, 1].set_xticks([])
#         axes[0, 1].set_yticks([])

#         mask = (df["model"] == m2) & (df["subclass"] == "c")
#         z_cond2_filtered = z_cond2[mask.to_numpy()]
#         im7 = axes[1, 1].imshow(
#             z_cond2_filtered,
#             aspect="auto",
#             cmap=cmap,
#             vmin=vmin_global,
#             vmax=vmax_global,
#             rasterized=False,
#             interpolation="none",
#         )

#         axes[1, 1].set_xticks([])
#         axes[1, 1].set_yticks([])

#         mask = (df["model"] == m3) & (df["subclass"] == "a")
#         z_cond2_filtered = z_cond2[mask.to_numpy()].copy()
#         if not self.show_unexpressed:
#             z_cond2_filtered[:] = 0
#         im8 = axes[2, 1].imshow(
#             z_cond2_filtered,
#             aspect="auto",
#             cmap=cmap,
#             vmin=vmin_global,
#             vmax=vmax_global,
#             rasterized=False,
#             interpolation="none",
#         )

#         axes[2, 1].set_xticks([])
#         axes[2, 1].set_yticks([])

#         mask = (df["model"] == m3) & (df["subclass"] == "c")
#         z_cond2_filtered = z_cond2[mask.to_numpy()]
#         im9 = axes[3, 1].imshow(
#             z_cond2_filtered,
#             aspect="auto",
#             cmap=cmap,
#             vmin=vmin_global,
#             vmax=vmax_global,
#             rasterized=False,
#             interpolation="none",
#         )

#         axes[3, 1].set_xticks([])
#         axes[3, 1].set_yticks([])

#         mask = df["model"] == m4
#         z_cond2_filtered = z_cond2[mask.to_numpy()]
#         im10 = axes[4, 1].imshow(
#             z_cond2_filtered,
#             aspect="auto",
#             cmap=cmap,
#             vmin=vmin_global,
#             vmax=vmax_global,
#             rasterized=False,
#             interpolation="none",
#         )

#         axes[4, 1].set_xticks([])
#         axes[4, 1].set_yticks([])

#         mask = df["model"] == m5
#         z_cond2_filtered = z_cond2[mask.to_numpy()]
#         im11 = axes[5, 1].imshow(
#             z_cond2_filtered,
#             aspect="auto",
#             cmap=cmap,
#             vmin=vmin_global,
#             vmax=vmax_global,
#             rasterized=False,
#             interpolation="none",
#         )

#         axes[5, 1].set_xticks(range(len(t_unique)), t_unique)
#         axes[5, 1].set_yticks([])
#         axes[5, 1].set_xlabel("Time(h)")

#         return fig


# @dataclass(config=ConfigDict(arbitrary_types_allowed=True))
# class TimeSeriesExample:
#     df: pd.DataFrame

#     columns_cond1: list[str]
#     columns_cond2: list[str]
#     t_cond1: np.ndarray
#     t_cond2: np.ndarray
#     cond1_label: str = "cond1"
#     cond2_label: str = "cond2"

#     deduplicate_on_init: bool = False

#     # --- validators ---
#     @field_validator("t_cond1", "t_cond2", mode="before")
#     @classmethod
#     def _to_1d_f64(cls, v: object) -> np.ndarray:  # ← return type fixes ANN206
#         a = np.asarray(v, dtype=np.float64)
#         if a.ndim != 1:
#             text: str = "expected 1D array"
#             raise ValueError(text)
#         return a

#     @model_validator(mode="after")
#     def _check_columns(self) -> Self:
#         missing = [c for c in (self.columns_cond1 + self.columns_cond2) if c not in self.df.columns]
#         if missing:
#             text = f"Missing columns: {missing}"  # satisfies EM101 (no string literal in raise)
#             raise ValueError(text)
#         return self

#     def plot_time_series(self, gene: str, xticks: np.ndarray | None = None) -> None:
#         if xticks is None:
#             xticks = np.unique(np.concatenate((self.t_cond1, self.t_cond2))).astype(int)
#         df: pd.DataFrame = self.df
#         df_curr = df[df["Genes"] == gene]
#         if df_curr.empty:
#             msg = f"Gene '{gene}' not found in the DataFrame."
#             raise ValueError(msg)
#         if len(df_curr) > 1:
#             df_curr = df_curr.loc[[df_curr["mean_cond2"].idxmax()]]

#         alpha_vec: np.ndarray = df_curr[self.columns_cond1].to_numpy(float).flatten()
#         beta_vec: np.ndarray = df_curr[self.columns_cond2].to_numpy(float).flatten()

#         is_expressed_cond1: bool = df_curr["is_expressed_cond1"].to_numpy()[0]
#         is_expressed_cond2: bool = df_curr["is_expressed_cond2"].to_numpy()[0]

#         model: int = df_curr["model"].to_numpy()[0]

#         if is_expressed_cond1 and is_expressed_cond2:
#             t_test, y_test_cond1, y_test_cond2 = self.get_test_function_expressed_both(
#                 alpha_vec,
#                 beta_vec,
#                 self.t_cond1,
#                 self.t_cond2,
#                 model,
#             )
#         elif is_expressed_cond1 and not is_expressed_cond2:
#             t_test, y_test_cond1, y_test_cond2 = self.get_test_function_expressed_cond1(
#                 alpha_vec,
#                 self.t_cond1,
#                 model,
#             )
#         elif not is_expressed_cond1 and is_expressed_cond2:
#             t_test, y_test_cond1, y_test_cond2 = self.get_test_function_expressed_cond2(
#                 beta_vec,
#                 self.t_cond2,
#                 model,
#             )
#         plt.figure(figsize=(12 / 2.54, 6 / 2.54))
#         plt.subplot(1, 2, 1)
#         plt.scatter(self.t_cond1, alpha_vec, s=4)
#         plt.plot(t_test, y_test_cond1)
#         plt.ylabel("Expression Level")
#         plt.xticks(xticks)
#         # plt.xlim(xticks[0], xticks[-1])
#         plt.title(f"Time Series for {gene}")
#         plt.xlabel("Time (h)")
#         plt.subplot(1, 2, 2)
#         plt.scatter(self.t_cond2, beta_vec, s=4, color="r")
#         plt.plot(t_test, y_test_cond2, color="r")
#         plt.title(f"Time Series for {gene}")
#         plt.xlabel("Time (h)")

#         plt.xticks(xticks)
#         plt.xlim(
#             xticks[0] - 0.05 * (xticks[-1] - xticks[0]),
#             xticks[-1] + 0.05 * (xticks[-1] - xticks[0]),
#         )
#         plt.tight_layout()
#         plt.show()

#     def get_test_function_expressed_both(
#         self,
#         alpha_vec: np.ndarray,
#         beta_vec: np.ndarray,
#         t_cond1: np.ndarray,
#         t_cond2: np.ndarray,
#         model: int,
#     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
#         design: pd.DataFrame = build_design(alpha_vec, beta_vec, t_cond1, t_cond2)
#         t_test: np.ndarray = np.linspace(0, 24, 100)
#         if model == 0:
#             print("No model selected")
#             y_test_cond1 = np.full_like(t_test, np.nan)
#             y_test_cond2 = np.full_like(t_test, np.nan)
#         else:
#             model_formula: str = MODELS[model - 1].formula
#             res = smf.ols(model_formula, data=design).fit()
#             df_cond1: pd.DataFrame = pd.DataFrame(
#                 {"time": t_test, "dataset": "alpha"},
#             ).dropna()
#             df_cond2: pd.DataFrame = pd.DataFrame(
#                 {"time": t_test, "dataset": "beta"},
#             ).dropna()
#             df_test: pd.DataFrame = pd.concat([df_cond1, df_cond2], ignore_index=True)
#             df_test["constant"] = 1.0
#             df_test["cos_wt"] = np.cos(W * df_test["time"].to_numpy().astype(float))
#             df_test["sin_wt"] = np.sin(W * df_test["time"].to_numpy().astype(float))
#             df_test["is_alpha"] = (df_test["dataset"] == "alpha").astype(int)
#             df_test["is_beta"] = (df_test["dataset"] == "beta").astype(int)
#             y_test = res.predict(exog=df_test)
#             y_test_cond1: np.ndarray = y_test[df_test["dataset"] == "alpha"].to_numpy()
#             y_test_cond2: np.ndarray = y_test[df_test["dataset"] == "beta"].to_numpy()
#         return t_test, y_test_cond1, y_test_cond2

#     def get_test_function_expressed_cond1(
#         self,
#         alpha_vec: np.ndarray,
#         t_cond1: np.ndarray,
#         model: int,
#     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
#         design: pd.DataFrame = build_design_cond1(alpha_vec, t_cond1)
#         t_test: np.ndarray = np.linspace(0, 24, 100)
#         if model == 0:
#             print("No model selected")
#             y_test_cond1 = np.full_like(t_test, np.nan)
#             y_test_cond2 = np.full_like(t_test, np.nan)
#         else:
#             print(f"Fitting model {model}")
#             print(MODELS_ONE_CONDITION)
#             if model == 1:
#                 model_formula: str = MODELS_ONE_CONDITION[0].formula
#             else:
#                 model_formula: str = MODELS_ONE_CONDITION[1].formula
#             res = smf.ols(model_formula, data=design).fit()
#             df_test: pd.DataFrame = pd.DataFrame(
#                 {"time": t_test, "dataset": "alpha"},
#             ).dropna()
#             df_test["constant"] = 1.0
#             df_test["cos_wt"] = np.cos(W * df_test["time"].to_numpy().astype(float))
#             df_test["sin_wt"] = np.sin(W * df_test["time"].to_numpy().astype(float))
#             y_test = res.predict(exog=df_test)
#             y_test_cond1: np.ndarray = y_test.to_numpy()
#             y_test_cond2: np.ndarray = np.full_like(t_test, np.nan)
#         return t_test, y_test_cond1, y_test_cond2

#     def get_test_function_expressed_cond2(
#         self,
#         beta_vec: np.ndarray,
#         t_cond2: np.ndarray,
#         model: int,
#     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
#         design: pd.DataFrame = build_design_cond2(beta_vec, t_cond2)
#         t_test: np.ndarray = np.linspace(0, 24, 100)
#         if model == 0:
#             print("No model selected")
#             y_test_cond1 = np.full_like(t_test, np.nan)
#             y_test_cond2 = np.full_like(t_test, np.nan)
#         else:
#             print(f"Fitting model {model}")
#             print(MODELS_ONE_CONDITION)
#             if model == 1:
#                 model_formula: str = MODELS_ONE_CONDITION[0].formula
#             else:
#                 model_formula: str = MODELS_ONE_CONDITION[1].formula
#             res = smf.ols(model_formula, data=design).fit()
#             df_test: pd.DataFrame = pd.DataFrame(
#                 {"time": t_test, "dataset": "alpha"},
#             ).dropna()
#             df_test["constant"] = 1.0
#             df_test["cos_wt"] = np.cos(W * df_test["time"].to_numpy().astype(float))
#             df_test["sin_wt"] = np.sin(W * df_test["time"].to_numpy().astype(float))
#             y_test = res.predict(exog=df_test)
#             y_test_cond1: np.ndarray = np.full_like(t_test, np.nan)
#             y_test_cond2: np.ndarray = y_test.to_numpy()

#         return t_test, y_test_cond1, y_test_cond2


with gr.Blocks(title="Cosinor Analysis — Live Cell & Omics") as demo:
    gr.Markdown("""
    # Cosinor Analysis App
                
    A simple app for circadian cosinor analysis & biostatistics.

    Choose between: 
    - inferring rhythmic properties of live cell data using three different cosinor models
    - differential rhythmicity analysis of omics datasets                       

    """)

    with gr.Tabs() as tabs:
        with gr.Tab("Live cell", id=0):
            gr.Image(
                value=method_img,
                label="Choosing a cosinor model and fitting parameters",
                interactive=False,
                show_label=True,
                height=600,  # adjust as needed; or remove to use natural size
            )

            gr.Markdown("""
            # Fitting live cell data
            
            This section allows the use to infer parameters describing circadian oscillations in live cell data.
            Once inferred, the extracted parameters can be compared between groups in downstream analyses. There are three
                        types of cosinor model to choose from:
            - 24h period cosinor
            - Free period (constrained within 20-28h) cosinor
            - Damped cosinor (equivalent to Chronostar analysis), with an additional dampening coefficient            
                        
            There are many valid ways to organise the underlying live cell data file. Here we assume a specific format to facilitate data processing

            - Row 1: contains a unique identifier for the participant, mouse etc.
            - Row 2: replicate number. If there's only one replicate per unique ID, this can just be a row of 1's
            - Row 3: the group to which each measurement belongs
            - Left column; the left column contains the time (going down)            

            """)

            file = gr.File(label="Upload CSV", file_types=[".csv"], type="filepath")

            gr.Examples(
                examples=[bioluminescence_file, cytokine_file],
                inputs=file,
                label="Example input",
            )

            status = gr.Textbox(label="CSV status", interactive=False)

            with gr.Row():
                group1_label = gr.Textbox(label="Group 1 label", value="Group 1")
                group2_label = gr.Textbox(label="Group 2 label", value="Group 2")

            # State (values will be pandas / numpy objects, not components)
            st_participant_id = gr.State()
            st_replicate = gr.State()
            st_group = gr.State()
            st_time = gr.State()
            st_time_rows = gr.State()

            def process_csv(fpath):
                # If needed, normalize FileData dict -> str path here
                df_data = pd.read_csv(fpath, index_col=0, header=None)

                participant_id = df_data.iloc[0, :].astype(str)
                replicate = df_data.iloc[1, :].astype(int)
                group = df_data.iloc[2, :].astype(str)

                time_index = df_data.index[3:]
                try:
                    time = time_index.astype(float).to_numpy()
                except Exception:
                    raise ValueError(
                        f"Time index not numeric from row 4 onward: {list(time_index)}",
                    )

                time_rows = df_data.iloc[3:].apply(pd.to_numeric, errors="raise")

                shape_info = (
                    f"Loaded shape: {df_data.shape} | participants: {len(participant_id)} | time points: {len(time)}"
                )
                return shape_info, participant_id, replicate, group, time, time_rows

            # Trigger on both upload and change (Examples sets the value via change)
            for evt in (file.upload, file.change):
                evt(
                    process_csv,
                    inputs=file,
                    outputs=[
                        status,
                        st_participant_id,
                        st_replicate,
                        st_group,
                        st_time,
                        st_time_rows,
                    ],
                )

            group_choice = gr.Radio(
                choices=[("Group 1", "group1"), ("Group 2", "group2")],
                value="group1",
                label="Select group to analyse",
            )
            m_slider = gr.Slider(
                1,
                10,
                value=5,
                step=1,
                label="Number of columns for plot",
            )

            def make_plot(
                ids_series,
                group_series,
                repl_series,
                time_array,
                time_rows_df,
                g1,
                g2,
                which_group,
                method,
                window,
                m,
                plot_style,
            ):
                if ids_series is None:
                    return None, None
                ds = LiveCellDataset(
                    ids=list(ids_series),
                    group=list(group_series),
                    replicate=[int(x) for x in list(repl_series)],
                    time_series=time_rows_df.to_numpy(dtype=float),
                    time=np.asarray(time_array, dtype=float),
                    group1_label=g1,
                    group2_label=g2,
                )
                if method == "moving_average":
                    fig, pdf_path = ds.plot_group_data(
                        which_group,
                        method=method,
                        m=int(m),
                        window=int(window),
                        plot_style=plot_style,
                    )
                else:
                    fig, pdf_path = ds.plot_group_data(
                        which_group,
                        method=method,
                        m=int(m),
                        plot_style=plot_style,
                    )
                return fig, pdf_path

            method_choice = gr.Radio(
                choices=[
                    ("None (no detrending)", "none"),
                    ("Linear", "linear"),
                    ("Linear + quadratic", "poly2"),
                    ("Moving average", "moving_average"),
                ],
                value="none",
                label="Detrending method",
            )

            window_slider = gr.Slider(
                1,
                1000,
                value=144,
                step=1,
                label="Window size for moving average (if used)",
            )
            gr.Markdown(
                "⚠️ Note: Window size is in **data points** (measurement intervals), not hours.",
            )

            plot_style_choice = gr.Radio(
                choices=[("Line", "line"), ("Scatter", "scatter")],
                value="line",
                label="Plot style for raw data",
            )

            build_btn = gr.Button("Plot raw data and trend", variant="primary")
            plot = gr.Plot(label="Preview")
            download = gr.File(label="Download plot")

            build_btn.click(
                make_plot,
                inputs=[
                    st_participant_id,
                    st_group,
                    st_replicate,
                    st_time,
                    st_time_rows,
                    group1_label,
                    group2_label,
                    group_choice,
                    method_choice,
                    window_slider,
                    m_slider,
                    plot_style_choice,
                ],
                outputs=[plot, download],
            )

            cosinor_model = gr.Radio(
                choices=[
                    ("24h period cosinor", "cosinor_24"),
                    ("Free period (20-28h) cosinor", "cosinor_free_period"),
                    ("Damped cosinor (Chronostar)", "cosinor_damped"),
                ],
                value="cosinor_24",  # must match one of the VALUES above
                label="Cosinor model",
            )

            t_lower_slider = gr.Slider(
                0,
                1000,
                value=0,
                step=1,
                label="Remove data below this time limit (hours)",
            )
            t_upper_slider = gr.Slider(
                0,
                1000,
                value=1000,
                step=1,
                label="Remove data above this time limit (hours)",
            )

            build_btn_cosinor = gr.Button("Build cosinor fits", variant="primary")
            plot_cosinor = gr.Plot(label="Model fit preview")
            pdf_export = gr.File(label="Download figure")
            table_export = gr.Dataframe()
            download_export = gr.File(label="Download fitted parameters")

            def make_cosinor_fits(
                ids_series,
                group_series,
                repl_series,
                time_array,
                time_rows_df,
                g1,
                g2,
                which_group,
                method,
                window,
                cosinor_model,
                t_lower,
                t_upper,
                m,
                plot_style,
            ):
                if ids_series is None:
                    return None, None, None, None
                ds = CosinorAnalysis(
                    ids=list(ids_series),
                    group=list(group_series),
                    replicate=[int(x) for x in list(repl_series)],
                    time_series=time_rows_df.to_numpy(dtype=float),
                    time=np.asarray(time_array, dtype=float),
                    group1_label=g1,
                    group2_label=g2,
                    t_lower=t_lower,
                    t_upper=t_upper,
                )
                if method == "moving_average":
                    df_export, csv_path, fig, pdf_path = ds.fit_cosinor(
                        which_group,
                        method=method,
                        window=int(window),
                        cosinor_model=cosinor_model,
                        m=int(m),
                        plot_style=plot_style,
                    )
                else:
                    df_export, csv_path, fig, pdf_path = ds.fit_cosinor(
                        which_group,
                        method=method,
                        cosinor_model=cosinor_model,
                        m=int(m),
                        plot_style=plot_style,
                    )
                return fig, pdf_path, df_export, csv_path

            build_btn_cosinor.click(
                make_cosinor_fits,
                inputs=[
                    st_participant_id,
                    st_group,
                    st_replicate,
                    st_time,
                    st_time_rows,
                    group1_label,
                    group2_label,
                    group_choice,
                    method_choice,
                    window_slider,
                    cosinor_model,
                    t_lower_slider,
                    t_upper_slider,
                    m_slider,
                    plot_style_choice,  # <-- add
                ],
                outputs=[plot_cosinor, pdf_export, table_export, download_export],
            )
        with gr.Tab("Omics", id=1):
            gr.Markdown("""
            # Differential rhytmicity analysis of omics datasets
                        
            Here we perform differential rhythmicity analysis on omics data using a model selection approach. 
                        The example data includes published RNA-seq data, but in theory any types of omics data (RNA-seq, proteomics, metabolomics, lipidomics) could be used. The dataset is from the following publication:

            Petrenko V, Saini C, Giovannoni L, Gobet C, Sage D, Unser M, Heddad Masson M, Gu G, Bosco D, Gachon F, Philippe J, Dibner C. 2017. Pancreatic α- and β-cellular clocks have distinct molecular properties and impact on islet hormone secretion and gene expression. Genes Dev 31:383–398. doi:10.1101/gad.290379.116.
            """)

            gr.Image(
                value=model_selection_img,
                label="Differential rhytmicity analysis with model selection",
                interactive=False,
                show_label=True,
                height=600,  # adjust as needed; or remove to use natural size
            )

            gr.Markdown("""
            ## How does the method actually work?
            
            The details of the method are nicely explained in the article: 

            Pelikan A, Herzel H, Kramer A, Ananthasubramaniam B. 2022. Venn diagram analysis overestimates the extent of circadian rhythm reprogramming. The FEBS Journal 289:6605–6621. doi:10.1111/febs.16095

            See the above adaptation of their figure explaining the methdology.

            For condition 1 (i.e. alpha cells) and condition 2 (i.e. beta cells), we fit 5 different models:

            - Model 1) Arrhythmic in alpha and beta cells

            - Model 2) Rhythmic in beta cells only

            - Model 3) Rhythmic in alpha cells only

            - Model 4) Rhythmic in alpha and beta cells with the same rhythmic parameters (i.e. phase and amplitude)

            - Model 5) Rhythmic in both but with differential rhythmicity in alpha vs beta cells

            A degree of confidence is calculated for each model (called model weight, which sums to 1 across all models), and a model is chosen if the model weight exceeds a threshold (for this tutorial we will use 0.5). If no model exceeds this threshold, then the model is unclassified, which we define as Model 0

            - Model 0) unclassified        

            """)

            # File input + example
            omics_file = gr.File(
                label="Upload Omics TXT/TSV",
                file_types=[".txt", ".tsv", ".csv"],
                type="filepath",
            )
            gr.Examples(
                examples=[omics_example],
                inputs=omics_file,
                label="Example input",
            )

            omics_status = gr.Textbox(label="File status", interactive=False)

            # Multiselects for choosing columns by header names
            columns_cond1_dd = gr.Dropdown(
                choices=[],
                multiselect=True,
                label="Condition A columns (e.g., ZT_*_a_*)",
            )
            columns_cond2_dd = gr.Dropdown(
                choices=[],
                multiselect=True,
                label="Condition B columns (e.g., ZT_*_b_*)",
            )

            # Optional manual time vectors
            override_time = gr.Checkbox(
                label="Override time vectors manually?",
                value=False,
            )
            t_cond1_tb = gr.Textbox(label="t_cond1 (comma-separated)", visible=False)
            t_cond2_tb = gr.Textbox(label="t_cond2 (comma-separated)", visible=False)

            # Hidden state to stash the DataFrame
            st_df_rna = gr.State()

            # Preview planned inputs for your class
            omics_preview = gr.Code(label="Planned class inputs", language="python")
            build_omics_btn = gr.Button("Build Omics inputs", variant="primary")

            # Sample dropdowns for replicate scatterplot
            sample1_dd = gr.Dropdown(choices=[], label="Sample 1 (x-axis)")
            sample2_dd = gr.Dropdown(choices=[], label="Sample 2 (y-axis)")
            scatter_btn = gr.Button(
                "Generate replicate scatterplot",
                variant="secondary",
            )
            scatter_plot = gr.Plot(label="Replicate scatterplot")
            scatter_download = gr.File(label="Download scatterplot")

            # Histogram outputs
            hist_btn = gr.Button("Generate histogram", variant="primary")
            omics_plot = gr.Plot(label="Expression histogram")
            omics_download = gr.File(label="Download histogram")

            # ---------- helpers ----------
            def _guess_cols(cols):
                cols = list(cols)
                a_guess = [c for c in cols if re.search(r"_a_", str(c))]
                b_guess = [c for c in cols if re.search(r"_b_", str(c))]

                def zt_key(c):
                    m = re.search(r"ZT_(\d+)", str(c))
                    return int(m.group(1)) if m else 0

                a_guess = sorted(a_guess, key=zt_key)
                b_guess = sorted(b_guess, key=zt_key)
                return a_guess, b_guess

            def _pick_default_samples(cols):
                # Try ZT_0 ... _1 and ZT_0 ... _2 as a sensible default pair
                s1 = next((c for c in cols if re.search(r"ZT_0_.*_1$", str(c))), None)
                s2 = next((c for c in cols if re.search(r"ZT_0_.*_2$", str(c))), None)
                if s1 and s2 and s1 != s2:
                    return s1, s2
                cols = [str(c) for c in cols]
                return (cols[0] if cols else None, cols[1] if len(cols) > 1 else None)

            def _build_time_vec(n_cols, manual_text):
                if manual_text:
                    try:
                        return [float(x.strip()) for x in manual_text.split(",") if x.strip()]
                    except Exception:
                        pass
                base = [0, 4, 8, 12, 16, 20]
                reps = max(1, (n_cols + len(base) - 1) // len(base))
                return (base * reps)[:n_cols]

            # ---------- loaders & toggles ----------
            def load_omics(fpath):
                # Read TSV (tab-separated). Pandas also handles CSV if present.
                df = pd.read_csv(fpath, sep="\t")
                # Drop first column by index (your pipeline)
                if df.shape[1] > 0:
                    df = df.drop(df.columns[0], axis=1)
                # Create Genes column from gene_name if present
                if "gene_name" in df.columns:
                    df["Genes"] = df["gene_name"].astype(str).str.split("|").str[1]

                status = f"Loaded {df.shape[0]} rows × {df.shape[1]} columns."
                choices = df.columns.tolist()
                a_guess, b_guess = _guess_cols(choices)
                s1, s2 = _pick_default_samples(choices)
                default_cycle = "0,4,8,12,16,20"

                return (
                    status,
                    gr.update(choices=choices, value=a_guess),
                    gr.update(choices=choices, value=b_guess),
                    df,
                    gr.update(value=default_cycle),
                    gr.update(value=default_cycle),
                    gr.update(choices=choices, value=s1),
                    gr.update(choices=choices, value=s2),
                )

            for evt in (omics_file.upload, omics_file.change):
                evt(
                    load_omics,
                    inputs=omics_file,
                    outputs=[
                        omics_status,
                        columns_cond1_dd,
                        columns_cond2_dd,
                        st_df_rna,
                        t_cond1_tb,
                        t_cond2_tb,
                        sample1_dd,
                        sample2_dd,
                    ],
                )

            def toggle_time_fields(checked):
                return gr.update(visible=checked), gr.update(visible=checked)

            override_time.change(
                toggle_time_fields,
                inputs=override_time,
                outputs=[t_cond1_tb, t_cond2_tb],
            )

            # ---------- preview inputs for your future class ----------
            def build_omics_inputs(
                df,
                cols_a,
                cols_b,
                use_manual_time,
                t_a_text,
                t_b_text,
            ):
                if df is None:
                    return "# Upload or select an example file first."

                t_a = _build_time_vec(
                    len(cols_a),
                    t_a_text if use_manual_time else None,
                )
                t_b = _build_time_vec(
                    len(cols_b),
                    t_b_text if use_manual_time else None,
                )

                snippet = f"""# Planned inputs for your Omics class
        columns_cond1 = {cols_a}
        columns_cond2 = {cols_b}
        t_cond1 = {t_a}
        t_cond2 = {t_b}

        # Example construction (later):
        # rna_data = OmicsDataset(
        #     df=df_rna,
        #     columns_cond1=columns_cond1,
        #     columns_cond2=columns_cond2,
        #     t_cond1=t_cond1,
        #     t_cond2=t_cond2,
        #     deduplicate_on_init=True,
        # )
        """
                return snippet

            build_omics_btn.click(
                build_omics_inputs,
                inputs=[
                    st_df_rna,
                    columns_cond1_dd,
                    columns_cond2_dd,
                    override_time,
                    t_cond1_tb,
                    t_cond2_tb,
                ],
                outputs=omics_preview,
            )

            # ---------- histogram ----------
            def run_histogram(df, cols_a, cols_b, use_manual_time, t_a_text, t_b_text):
                if df is None:
                    return None, None

                t_a = _build_time_vec(
                    len(cols_a),
                    t_a_text if use_manual_time else None,
                )
                t_b = _build_time_vec(
                    len(cols_b),
                    t_b_text if use_manual_time else None,
                )

                rna_data = OmicsDataset(
                    df=df,
                    columns_cond1=cols_a,
                    columns_cond2=cols_b,
                    t_cond1=t_a,
                    t_cond2=t_b,
                    deduplicate_on_init=True,
                )

                fig = rna_data.expression_histogram()

                tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                fig.savefig(tmpfile.name)
                plt.close(fig)
                return fig, tmpfile.name

            hist_btn.click(
                run_histogram,
                inputs=[
                    st_df_rna,
                    columns_cond1_dd,
                    columns_cond2_dd,
                    override_time,
                    t_cond1_tb,
                    t_cond2_tb,
                ],
                outputs=[omics_plot, omics_download],
            )

            # ---------- replicate scatterplot ----------
            def run_replicate_scatter(  # noqa: PLR0913
                df,
                cols_a,
                cols_b,
                use_manual_time,
                t_a_text,
                t_b_text,
                sample1,
                sample2,
            ):
                if df is None or not sample1 or not sample2:
                    return None, None

                t_a = _build_time_vec(
                    len(cols_a),
                    t_a_text if use_manual_time else None,
                )
                t_b = _build_time_vec(
                    len(cols_b),
                    t_b_text if use_manual_time else None,
                )

                rna_data = OmicsDataset(
                    df=df,
                    columns_cond1=cols_a,
                    columns_cond2=cols_b,
                    t_cond1=t_a,
                    t_cond2=t_b,
                    deduplicate_on_init=True,
                )

                fig = rna_data.replicate_scatterplot(sample1=sample1, sample2=sample2)

                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                fig.savefig(tmp.name)
                plt.close(fig)
                return fig, tmp.name

            scatter_btn.click(
                run_replicate_scatter,
                inputs=[
                    st_df_rna,
                    columns_cond1_dd,
                    columns_cond2_dd,
                    override_time,
                    t_cond1_tb,
                    t_cond2_tb,
                    sample1_dd,
                    sample2_dd,
                ],
                outputs=[scatter_plot, scatter_download],
            )

            # ------------------------------------------------------------
            # Differential rhythmicity + heatmap
            # (Paste this inside your existing `with gr.Tab("Omics", id=1):` block)
            # Re-uses helpers: _build_time_vec, and states/components you already created.
            # ------------------------------------------------------------

            # Labels for the heatmap and expressed-threshold
            with gr.Row():
                cond1_label_tb = gr.Textbox(
                    label="Condition 1 label",
                    value="Alpha cells",
                )
                cond2_label_tb = gr.Textbox(
                    label="Condition 2 label",
                    value="Beta cells",
                )
                mean_min_num = gr.Number(
                    label="mean_min (for is_expressed)",
                    value=0,
                    precision=0,
                )

            compute_dr_btn = gr.Button(
                "Compute differential rhythmicity & heatmap",
                variant="primary",
            )

            # Outputs
            heatmap_plot = gr.Plot(label="Heatmap preview")
            heatmap_download = gr.File(label="Download heatmap (PDF)")
            params_preview = gr.Dataframe(
                label="Rhythmic parameters (preview)",
                interactive=False,
            )
            params_download = gr.File(label="Download rhythmic parameters (CSV)")

            def run_dr_and_heatmap(  # noqa: PLR0913
                df,
                cols_a,
                cols_b,
                use_manual_time,
                t_a_text,
                t_b_text,
                cond1_label,
                cond2_label,
                mean_min,
            ):
                if df is None or not cols_a or not cols_b:
                    # Nothing to do yet
                    return None, None, None, None

                # Build time vectors (reuse your helper)
                t_a = _build_time_vec(
                    len(cols_a),
                    t_a_text if use_manual_time else None,
                )
                t_b = _build_time_vec(
                    len(cols_b),
                    t_b_text if use_manual_time else None,
                )

                # Construct dataset
                rna_data = OmicsDataset(
                    df=df,
                    columns_cond1=cols_a,
                    columns_cond2=cols_b,
                    t_cond1=t_a,
                    t_cond2=t_b,
                    deduplicate_on_init=True,
                )

                # Mark expressed genes
                try:
                    rna_data.add_is_expressed(mean_min=float(mean_min))
                except Exception:
                    # Fall back if mean_min is None/NaN
                    rna_data.add_is_expressed(mean_min=0.0)

                # Differential rhythmicity
                dr = DifferentialRhythmicity(dataset=rna_data)
                rhythmic_all = dr.extract_all_circadian_params()  # pandas DataFrame

                # Build heatmap
                heatmap = OmicsHeatmap(
                    df=rhythmic_all,
                    columns_cond1=cols_a,
                    columns_cond2=cols_b,
                    t_cond1=t_a,
                    t_cond2=t_b,
                    cond1_label=cond1_label or "Condition 1",
                    cond2_label=cond2_label or "Condition 2",
                    show_unexpressed=False,
                )
                fig = heatmap.plot_heatmap()  # should return a Matplotlib Figure

                # Save outputs for download
                import tempfile

                # Heatmap PDF
                tmp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                fig.savefig(tmp_pdf.name)

                # Rhythmic params CSV
                tmp_csv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
                rhythmic_all.to_csv(tmp_csv.name, index=False)

                # For preview table, show up to 200 rows to keep UI snappy
                preview_df = rhythmic_all.head(20)

                return fig, tmp_pdf.name, preview_df, tmp_csv.name

            compute_dr_btn.click(
                run_dr_and_heatmap,
                inputs=[
                    st_df_rna,
                    columns_cond1_dd,
                    columns_cond2_dd,
                    override_time,
                    t_cond1_tb,
                    t_cond2_tb,
                    cond1_label_tb,
                    cond2_label_tb,
                    mean_min_num,
                ],
                outputs=[
                    heatmap_plot,
                    heatmap_download,
                    params_preview,
                    params_download,
                ],
            )


if __name__ == "__main__":
    demo.launch()
