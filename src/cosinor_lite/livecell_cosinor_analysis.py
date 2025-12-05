from __future__ import annotations

import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import curve_fit
from scipy.stats import f as f_dist

from cosinor_lite.livecell_dataset import LiveCellDataset

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


def constant_model(x: NDArray[np.float64], mesor: float) -> NDArray[np.float64]:
    """
    Evaluate a constant model at the supplied mesor.

    Parameters
    ----------
    x : NDArray[np.float64]
        Time values at which the constant model is evaluated.
    mesor : float
        The mesor (mean level) used as the constant prediction.

    Returns
    -------
    NDArray[np.float64]
        The mesor value supplied by the caller broadcast over ``x``.

    """
    return np.full_like(x, mesor)


def cosine_model_24(x: NDArray[np.float64], amplitude: float, acrophase: float, mesor: float) -> NDArray[np.float64]:
    """
    Evaluate a 24 h cosine model for the provided time points.

    Parameters
    ----------
    x : NDArray[np.float64]
        Time values at which to evaluate the curve.
    amplitude : float
        Cosine amplitude.
    acrophase : float
        Phase shift (in hours) applied to the cosine.
    mesor : float
        Baseline offset of the curve.

    Returns
    -------
    NDArray[np.float64]
        The cosine values with a fixed 24 h period.

    """
    period = 24.0
    return amplitude * np.cos(2 * np.pi * (x - acrophase) / period) + mesor


def cosine_model_free_period(
    x: NDArray[np.float64],
    amplitude: float,
    acrophase: float,
    period: float,
    mesor: float,
) -> NDArray[np.float64]:
    """
    Evaluate a cosine model allowing the period to vary.

    Parameters
    ----------
    x : NDArray[np.float64]
        Time values at which to evaluate the cosine.
    amplitude : float
        Cosine amplitude.
    acrophase : float
        Phase shift (in hours) applied to the cosine.
    period : float
        Oscillation period in hours.
    mesor : float
        Baseline offset for the model.

    Returns
    -------
    NDArray[np.float64]
        The cosine values with the requested period.

    """
    return amplitude * np.cos(2 * np.pi * (x - acrophase) / period) + mesor


def cosine_model_damped(  # noqa: PLR0913
    x: NDArray[np.float64],
    amplitude: float,
    damp: float,
    acrophase: float,
    period: float,
    mesor: float,
) -> NDArray[np.float64]:
    """
    Evaluate a damped cosine model that decays exponentially over time.

    Parameters
    ----------
    x : NDArray[np.float64]
        Time values at which to evaluate the cosine.
    amplitude : float
        Initial amplitude of the oscillation.
    damp : float
        Exponential damping coefficient (positive values decay).
    acrophase : float
        Phase shift (in hours) applied to the cosine.
    period : float
        Oscillation period in hours.
    mesor : float
        Baseline offset for the model.

    Returns
    -------
    NDArray[np.float64]
        The damped cosine evaluated at ``x``.

    """
    return amplitude * np.exp(-damp * x) * np.cos(2 * np.pi * (x - acrophase) / period) + mesor


def _metrics(y_true: NDArray[np.float64], y_pred: NDArray[np.float64], p: int) -> tuple[float, float, float]:
    """
    Compute residual sum of squares and R-squared metrics.

    Parameters
    ----------
    y_true : NDArray[np.float64]
        Observed values.
    y_pred : NDArray[np.float64]
        Predicted values from a model.
    p : int
        Number of model parameters used in the fit.

    Returns
    -------
    tuple[float, float, float]
        Residual sum of squares, R-squared, and adjusted R-squared.

    """
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    n: int = y_true.size
    rss: float = np.sum((y_true - y_pred) ** 2)
    sst: float = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = np.nan
    r2_adj = np.nan
    if sst > 0:
        r2 = 1.0 - (rss / sst)
        if n > p and n > 1:
            r2_adj = 1.0 - (rss / (n - p)) / (sst / (n - 1))
    return rss, r2, r2_adj


def _sanitize_xy(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    min_points: int = 4,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Remove non-finite values, sort by time, and validate point count.

    Parameters
    ----------
    x : NDArray[np.float64]
        Candidate time points.
    y : NDArray[np.float64]
        Candidate measurements aligned with ``x``.
    min_points : int, optional
        Minimum required number of finite points after cleaning, by default 4.

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64]]
        Cleaned and sorted ``x`` and ``y`` arrays.

    Raises
    ------
    ValueError
        If fewer than ``min_points`` observations remain after cleaning.

    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    order: NDArray[np.float64] = np.argsort(x)
    x, y = x[order], y[order]
    if y.size < min_points:
        msg = f"Not enough valid points after cleaning (need ≥{min_points}, got {y.size})."
        raise ValueError(
            msg,
        )
    return x, y


class CosinorAnalysis(LiveCellDataset):
    """
    Extend ``LiveCellDataset`` with cosinor model fitting utilities.

    Parameters
    ----------
    *args
        Positional arguments forwarded to ``LiveCellDataset``.
    period : float, optional
        Default oscillation period for fits, by default 24.0.
    method : str, optional
        Default detrending method, by default ``"ols"``.
    t_lower : float, optional
        Lower bound on time window (hours) used when fitting, by default 0.0.
    t_upper : float, optional
        Upper bound on time window (hours) used when fitting, by default 720.0.
    **kwargs
        Keyword arguments forwarded to ``LiveCellDataset``.

    """

    def __init__(
        self,
        *args,  # noqa: ANN002
        period: float = 24.0,
        method: str = "ols",
        t_lower: float = 0.0,
        t_upper: float = 720.0,
        **kwargs,  # noqa: ANN003
    ) -> None:
        """
        Initialize the cosinor analysis object with dataset metadata.

        Parameters
        ----------
        *args
            Positional arguments passed to ``LiveCellDataset``.
        period : float, optional
            Default oscillation period (hours), by default 24.0.
        method : str, optional
            Default detrending method, by default ``"ols"``.
        t_lower : float, optional
            Lower time bound (hours), by default 0.0.
        t_upper : float, optional
            Upper time bound (hours), by default 720.0.
        **kwargs
            Keyword arguments forwarded to ``LiveCellDataset``.

        """
        super().__init__(*args, **kwargs)

        self.t_lower = t_lower
        self.t_upper = t_upper
        self.period = period
        self.method = method

    def fit_cosinor_24(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> tuple[dict, NDArray[np.float64], NDArray[np.float64]]:
        """
        Fit a fixed 24 h cosinor model to the supplied data.

        Parameters
        ----------
        x : NDArray[np.float64]
            Time values (hours).
        y : NDArray[np.float64]
            Measurements aligned with ``x``.

        Returns
        -------
        tuple[dict, NDArray[np.float64], NDArray[np.float64]]
            Fit summary metrics, high-resolution prediction times, and predictions.

        """
        x, y = _sanitize_xy(x, y)

        p0_const: list[float] = [float(np.mean(y))]
        params_const, _ = curve_fit(constant_model, x, y, p0=p0_const)
        yhat_const: float = constant_model(x, *params_const)

        p0_cos: list[float] = [
            float(np.std(y)),
            0.0,
            float(np.mean(y)),
        ]
        params_cos, _ = curve_fit(cosine_model_24, x, y, p0=p0_cos)
        amp_fit, acro_fit, mesor_fit = params_cos
        yhat_cos: NDArray[np.float64] = cosine_model_24(x, amp_fit, acro_fit, mesor_fit)

        rss_cos, r2, r2_adj = _metrics(y, yhat_cos, p=3)
        rss_const, _, _ = _metrics(y, np.full_like(x, yhat_const), p=1)

        n = len(y)
        p1, p2 = 1, 3
        num = max(rss_const - rss_cos, 0.0)
        den = max(rss_cos, 1e-12)
        f_stat = (num / (p2 - p1)) / (den / max(n - p2, 1))
        p_val = np.nan
        if n > p2:
            p_val = f_dist.sf(f_stat, p2 - p1, n - p2)

        t_test_acro = np.linspace(0.0, 24.0, 1440)
        y_test_acro = cosine_model_24(t_test_acro, amp_fit, acro_fit, mesor_fit)
        amplitude = abs(amp_fit)
        acrophase = t_test_acro[int(np.argmax(y_test_acro))]
        mesor = mesor_fit

        t_test: NDArray[np.float64] = np.linspace(float(x[0]), float(x[-1]), 1440)
        y_test = cosine_model_24(t_test, amp_fit, acro_fit, mesor_fit)

        results = {
            "mesor": mesor,
            "amplitude": amplitude,
            "acrophase": acrophase,
            "p-val osc": p_val,
            "r2": r2,
            "r2_adj": r2_adj,
        }
        return results, t_test, y_test

    def fit_cosinor_free_period(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> tuple[dict, NDArray[np.float64], NDArray[np.float64]]:
        """
        Fit a cosinor model with a free (bounded) period parameter.

        Parameters
        ----------
        x : NDArray[np.float64]
            Time values (hours).
        y : NDArray[np.float64]
            Measurements aligned with ``x``.

        Returns
        -------
        tuple[dict, NDArray[np.float64], NDArray[np.float64]]
            Fit summary metrics, high-resolution prediction times, and predictions.

        """
        x, y = _sanitize_xy(x, y)

        p0_const = [float(np.mean(y))]
        params_const, _ = curve_fit(constant_model, x, y, p0=p0_const)
        yhat_const = constant_model(x, *params_const)

        p0_24 = [float(np.std(y)), 0.0, float(np.mean(y))]
        params_24, _ = curve_fit(cosine_model_24, x, y, p0=p0_24)
        amp0, acro0, mesor0 = params_24
        p0_free = [amp0, acro0, 24.0, mesor0]  # [A, phase, period, mesor]

        bounds = ([-np.inf, -np.inf, 20.0, -np.inf], [np.inf, np.inf, 28.0, np.inf])
        params_free, _ = curve_fit(
            cosine_model_free_period,
            x,
            y,
            p0=p0_free,
            bounds=bounds,
        )
        amp_fit, acro_fit, period_fit, mesor_fit = params_free
        yhat_free = cosine_model_free_period(
            x,
            amp_fit,
            acro_fit,
            period_fit,
            mesor_fit,
        )

        rss_free, r2, r2_adj = _metrics(y, yhat_free, p=4)
        rss_const, _, _ = _metrics(y, yhat_const, p=1)

        n = len(y)
        p1, p2 = 1, 4
        num = max(rss_const - rss_free, 0.0)
        den = max(rss_free, 1e-12)
        f_stat = (num / (p2 - p1)) / (den / max(n - p2, 1))
        p_val = np.nan
        if n > p2:
            p_val = f_dist.sf(f_stat, p2 - p1, n - p2)

        t_test_acro = np.linspace(0.0, float(period_fit), 2000)
        y_test_acro = cosine_model_free_period(
            t_test_acro,
            amp_fit,
            acro_fit,
            period_fit,
            mesor_fit,
        )
        amplitude = abs(amp_fit)
        acrophase = t_test_acro[int(np.argmax(y_test_acro))]
        mesor = mesor_fit
        period = float(period_fit)

        t_test = np.linspace(float(x[0]), float(x[-1]), 1440)
        y_test = cosine_model_free_period(
            t_test,
            amp_fit,
            acro_fit,
            period_fit,
            mesor_fit,
        )

        results = {
            "mesor": mesor,
            "amplitude": amplitude,
            "acrophase": acrophase,
            "period": period,
            "p-val osc": p_val,
            "r2": r2,
            "r2_adj": r2_adj,
        }
        return results, t_test, y_test

    def fit_cosinor_damped(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> tuple[dict, NDArray[np.float64], NDArray[np.float64]]:
        """
        Fit a damped cosinor model with exponential decay.

        Parameters
        ----------
        x : NDArray[np.float64]
            Time values (hours).
        y : NDArray[np.float64]
            Measurements aligned with ``x``.

        Returns
        -------
        tuple[dict, NDArray[np.float64], NDArray[np.float64]]
            Fit summary metrics, high-resolution prediction times, and predictions.

        """
        x, y = _sanitize_xy(x, y)

        p0_const = [float(np.mean(y))]
        params_const, _ = curve_fit(constant_model, x, y, p0=p0_const)
        yhat_const = constant_model(x, *params_const)

        p0_24 = [float(np.std(y)), 0.0, float(np.mean(y))]
        params_24, _ = curve_fit(cosine_model_24, x, y, p0=p0_24)
        amp0, acro0, mesor0 = params_24

        p0_damped = [amp0, 0.01, acro0, 24.0, mesor0]
        bounds = (
            [-np.inf, 0.0, -np.inf, 20.0, -np.inf],
            [np.inf, np.inf, np.inf, 28.0, np.inf],
        )
        params_damped, _ = curve_fit(
            cosine_model_damped,
            x,
            y,
            p0=p0_damped,
            bounds=bounds,
        )
        amp_fit, damp_fit, acro_fit, period_fit, mesor_fit = params_damped
        yhat_damped = cosine_model_damped(
            x,
            amp_fit,
            damp_fit,
            acro_fit,
            period_fit,
            mesor_fit,
        )

        rss_damped, r2, r2_adj = _metrics(y, yhat_damped, p=5)
        rss_const, _, _ = _metrics(y, yhat_const, p=1)

        n = len(y)
        p1, p2 = 1, 5
        num = max(rss_const - rss_damped, 0.0)
        den = max(rss_damped, 1e-12)
        f_stat = (num / (p2 - p1)) / (den / max(n - p2, 1))
        p_val = np.nan
        if n > p2:
            p_val = f_dist.sf(f_stat, p2 - p1, n - p2)

        t_test_acro = np.linspace(0.0, float(period_fit), 1440)
        y_test_acro = cosine_model_damped(
            t_test_acro,
            amp_fit,
            damp_fit,
            acro_fit,
            period_fit,
            mesor_fit,
        )
        amplitude = abs(amp_fit)
        acrophase = t_test_acro[int(np.argmax(y_test_acro))]
        mesor = mesor_fit
        period = float(period_fit)
        damp = float(damp_fit)

        t_test = np.linspace(float(x[0]), float(x[-1]), 1440)
        y_test = cosine_model_damped(
            t_test,
            amp_fit,
            damp_fit,
            acro_fit,
            period_fit,
            mesor_fit,
        )

        results = {
            "mesor": mesor,
            "amplitude": amplitude,
            "acrophase": acrophase,
            "period": period,
            "damp": damp,
            "p-val osc": p_val,
            "r2": r2,
            "r2_adj": r2_adj,
        }
        return results, t_test, y_test

    def get_cosinor_fits(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        method: str = "cosinor_24",
    ) -> tuple[dict, NDArray[np.float64], NDArray[np.float64]]:
        """
        Dispatch to the requested cosinor model and return its fit.

        Parameters
        ----------
        x : NDArray[np.float64]
            Time values (hours).
        y : NDArray[np.float64]
            Measurements aligned with ``x``.
        method : str, optional
            Cosinor model name (``"cosinor_24"``, ``"cosinor_free_period"``, ``"cosinor_damped"``).

        Returns
        -------
        tuple[dict, NDArray[np.float64], NDArray[np.float64]]
            Fit summary metrics, prediction times, and predictions.

        Raises
        ------
        ValueError
            If an unknown model name is supplied.

        """
        if method == "cosinor_24":
            results, t_test, model_predictions_cosine = self.fit_cosinor_24(x, y)
        elif method == "cosinor_free_period":
            results, t_test, model_predictions_cosine = self.fit_cosinor_free_period(
                x,
                y,
            )
        elif method == "cosinor_damped":
            results, t_test, model_predictions_cosine = self.fit_cosinor_damped(x, y)
        else:
            msg = f"Unknown cosine model: {method}"
            raise ValueError(msg)

        return results, t_test, model_predictions_cosine

    def group_selector(
        self,
        group: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, int, str]:
        """
        Retrieve group-specific data and plotting metadata.

        Parameters
        ----------
        group : str
            Group name (``"group1"`` or ``"group2"``).

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, str, int, str]
            Tuple containing IDs, replicates, data matrix, color, number of unique IDs,
            and group label.

        Raises
        ------
        ValueError
            If ``group`` is not ``"group1"`` or ``"group2"``.

        """
        if group == "group1":
            ids, replicates, data = self.get_group1_ids_replicates_data()
            color = self.color_group1
            group_label = self.group1_label
        elif group == "group2":
            ids, replicates, data = self.get_group2_ids_replicates_data()
            color = self.color_group2
            group_label = self.group2_label
        else:
            msg = "group must be 'group1' or 'group2'"
            raise ValueError(msg)

        ids = np.asarray(ids)
        replicates = np.asarray(replicates)
        data = np.asarray(data)

        n_group = len(np.unique(ids))
        return ids, replicates, data, color, n_group, group_label

    def fit_cosinor(  # noqa: PLR0913
        self,
        group: str,
        method: str = "linear",
        window: int = 5,
        cosinor_model: str = "cosinor_24",
        m: int = 5,
        plot_style: str = "scatter",
    ) -> tuple[pd.DataFrame, str, plt.Figure, str]:
        """
        Detrend each replicate in a group and fit the selected cosinor model.

        Parameters
        ----------
        group : str
            Group name (``"group1"`` or ``"group2"``).
        method : str, optional
            Detrending method provided to ``get_trend``, by default ``"linear"``.
        window : int, optional
            Window size for moving-average detrending, by default 5.
        cosinor_model : str, optional
            Name of cosinor model to fit, by default ``"cosinor_24"``.
        m : int, optional
            Number of subplot columns, by default 5.
        plot_style : str, optional
            Plot style for data (``"scatter"`` or ``"line"``), by default ``"scatter"``.

        Returns
        -------
        tuple[pd.DataFrame, str, plt.Figure, str]
            DataFrame of fit metrics, CSV path, generated figure, and PDF path.

        Raises
        ------
        ValueError
            If ``group`` is not ``"group1"`` or ``"group2"``.

        """
        ids, _replicates, data, color, n_group, group_label = self.group_selector(group)
        n = np.ceil(n_group / m).astype(int)

        study_list = np.unique(ids).tolist()

        fig = plt.figure(figsize=(5 * m / 2.54, 5 * n / 2.54))

        to_export_list = []

        for i, id_curr in enumerate(study_list):
            mask = np.array(ids) == id_curr
            n_reps = np.sum(mask)

            ax = fig.add_subplot(n, m, i + 1)

            for j in range(n_reps):
                exp_info = {"id": id_curr, "replicate": j + 1, "group": group_label}

                x = self.time
                y = data[:, mask][:, j]

                valid_mask = ~np.isnan(y)
                x_valid = x[valid_mask]
                y_valid = y[valid_mask]

                range_mask = (x_valid >= self.t_lower) & (x_valid <= self.t_upper)
                x_fit = x_valid[range_mask]
                y_fit = y_valid[range_mask]

                x_processed, _y_trend, y_detrended = self.get_trend(
                    x_fit,
                    y_fit,
                    method=method,
                    window=window,
                )

                if plot_style == "scatter":
                    ax.scatter(x_processed, y_detrended, s=4, alpha=0.8, color=color)
                else:
                    ax.plot(x_processed, y_detrended, color=color)

                results, t_test, model_predictions_cosine = self.get_cosinor_fits(
                    x_processed,
                    y_detrended,
                    method=cosinor_model,
                )
                to_export_list.append({**exp_info, **results})
                ax.plot(t_test, model_predictions_cosine, color="k", linestyle="--")

            ax.set_title(f"ID: {id_curr} (n={n_reps}) - {group_label}")
            ax.set_xlabel("Time (h)")
            if i % m == 0:
                ax.set_ylabel("Expression")

        plt.tight_layout()

        df_export = pd.DataFrame(to_export_list)
        fd, tmp1_path = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        df_export.to_csv(tmp1_path, index=False)

        fd, tmp2_path = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
        fig.savefig(tmp2_path)

        return df_export, tmp1_path, fig, tmp2_path
