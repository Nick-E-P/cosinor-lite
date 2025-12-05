from __future__ import annotations

import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pydantic import ConfigDict, field_validator, model_validator
from pydantic.dataclasses import dataclass

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
    """
    Handles live cell dataset operations: validation, trend analysis, and plotting.

    Attributes:
        ids (list[str]): Unique identifiers for each sample.
        group (list[str]): Group labels for each sample.
        replicate (list[int]): Replicate numbers for each sample.
        time_series (np.ndarray): 2D array of time series data (shape: [timepoints, samples]).
        time (np.ndarray): 1D array of timepoints.
        group1_label (str): Label for group 1 (default: "group1").
        group2_label (str): Label for group 2 (default: "group2").
        color_group1 (str): Color for plotting group 1 (default: "tab:blue").
        color_group2 (str): Color for plotting group 2 (default: "tab:orange").

    Methods:
        _to_2d_f64(v): Converts input to a 2D numpy array of float64 type.
        _check_columns(): Validates that ids, group, and replicate lengths match time_series columns.
        get_group1_ids_replicates_data(): Returns ids, replicates, and data for group 1.
        get_group2_ids_replicates_data(): Returns ids, replicates, and data for group 2.
        linear_trend(x, y): Fits a linear trend to the data.
        poly2_trend(x, y): Fits a second-order polynomial trend to the data.
        moving_average_trend(x, y, window): Computes a moving average trend.
        get_trend(x, y, method, window): Applies the specified detrending method.
        plot_group_data(group, method, window, m, plot_style): Plots time series data for the specified group.

    """

    ids: list[str]
    group: list[str]
    replicate: list[int]
    time_series: np.ndarray
    time: np.ndarray

    group1_label: str = "group1"
    group2_label: str = "group2"

    color_group1: str = "tab:blue"
    color_group2: str = "tab:orange"

    @field_validator("time_series", mode="before")
    @classmethod
    def _to_2d_f64(cls, v: object) -> np.ndarray:
        """
        Converts the input object to a 2D NumPy array of type float64.

        Parameters
        ----------
        v : object
            The input object to be converted. Should be convertible to a NumPy array.

        Returns
        -------
        np.ndarray
            A 2D NumPy array of dtype float64.

        Raises
        ------
        ValueError
            If the input cannot be converted to a 2D array.

        """
        a = np.asarray(v, dtype=np.float64)
        two_d_array_dims = 2
        if a.ndim != two_d_array_dims:
            msg = "expected 2D array"
            raise ValueError(msg)
        return a

    @model_validator(mode="after")
    def _check_columns(self) -> LiveCellDataset:
        """
        Validates that the lengths of `ids`, `group`, and `replicate` match the number of columns in `time_series`.

        Parameters
        ----------
        self : LiveCellDataset
            The instance of the LiveCellDataset being validated.

        Returns
        -------
        LiveCellDataset
            The validated instance.

        Raises
        ------
        ValueError
            If the length of `ids`, `group`, or `replicate` does not match the number of columns in `time_series`.

        """
        if len(self.ids) != self.time_series.shape[1]:
            msg = "Length of ids must match number of columns in time_series"
            raise ValueError(
                msg,
            )
        if len(self.group) != self.time_series.shape[1]:
            msg = "Length of group must match number of columns in time_series"
            raise ValueError(
                msg,
            )
        if len(self.replicate) != self.time_series.shape[1]:
            msg = "Length of replicate must match number of columns in time_series"
            raise ValueError(
                msg,
            )
        return self

    def get_group1_ids_replicates_data(self) -> tuple[list[str], list[int], np.ndarray]:
        """
        Retrieves the IDs, replicate numbers, and time series data for samples belonging to group1.

        Returns:
            tuple[list[str], list[int], np.ndarray]:
                - ids: List of sample IDs in group1.
                - replicates: List of replicate numbers corresponding to group1 samples.
                - data: 2D NumPy array of time series data for group1 samples (columns correspond to samples).

        """
        mask = np.array(self.group) == self.group1_label
        ids = list(np.array(self.ids)[mask])
        replicates = list(np.array(self.replicate)[mask])
        data = self.time_series[:, mask]
        return ids, replicates, data

    def get_group2_ids_replicates_data(self) -> tuple[list[str], list[int], np.ndarray]:
        """
        Retrieves the IDs, replicate numbers, and time series data for samples belonging to group2.

        Returns:
            tuple[list[str], list[int], np.ndarray]:
                - ids: List of sample IDs in group2.
                - replicates: List of replicate numbers corresponding to group2 samples.
                - data: 2D NumPy array of time series data for group2 samples (columns correspond to samples).

        """
        mask = np.array(self.group) == self.group2_label
        ids = list(np.array(self.ids)[mask])
        replicates = list(np.array(self.replicate)[mask])
        data = self.time_series[:, mask]
        return ids, replicates, data

    def linear_trend(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fits a linear regression model to the provided data and returns the input x values along with the predicted linear trend.

        Parameters
        ----------
        x : np.ndarray
            The independent variable values.
        y : np.ndarray
            The dependent variable values.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            A tuple containing the original x values, the predicted linear fit values, and the detrended y values.

        """
        model = sm.OLS(y, sm.add_constant(x)).fit()
        linear_fit = model.predict(sm.add_constant(x))
        y_detrended = y - linear_fit + np.mean(y)
        return x, linear_fit, y_detrended

    def poly2_trend(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fits a second-degree polynomial (quadratic) trend to the given data.

        Args:
            x (np.ndarray): 1D array of independent variable values.
            y (np.ndarray): 1D array of dependent variable values.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]:
                - x: The input array of independent variable values.
                - poly_fit: The fitted quadratic values corresponding to x.
                - y_detrended: The detrended y values after removing the polynomial fit.

        """
        coeffs = np.polyfit(x, y, 2)
        poly_fit = np.polyval(coeffs, x)
        y_detrended = y - poly_fit + np.mean(y)
        return x, poly_fit, y_detrended

    def moving_average_trend(
        self,
        x: np.ndarray,
        y: np.ndarray,
        window: int = 5,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes the moving average trend of the input data using a specified window size.

        Parameters
        ----------
        x : np.ndarray
            The array of x-values (e.g., time points).
        y : np.ndarray
            The array of y-values (e.g., measurements corresponding to x).
        window : int, optional
            The size of the moving average window. Must be at least 1. Default is 5.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple containing:
                - x values corresponding to valid (finite) moving average points.
                - The moving average values (trend) for the valid points.
                - The detrended y values for the valid points.

        Raises
        ------
        ValueError
            If the window size is less than 1.

        """
        if window < 1:
            msg = "Window size must be at least 1."
            raise ValueError(msg)
        y_series = pd.Series(y)
        ma_fit = y_series.rolling(window=window, center=True).mean().to_numpy()
        good = np.isfinite(x) & np.isfinite(ma_fit)
        y_detrended = y[good] - ma_fit[good] + np.mean(y[good])
        return x[good], ma_fit[good], y_detrended

    def get_trend(
        self,
        x: np.ndarray,
        y: np.ndarray,
        method: str = "linear",
        window: int = 5,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes the trend of the given data using the specified method.

        Parameters
        ----------
        x : np.ndarray
            The independent variable data (e.g., time points).
        y : np.ndarray
            The dependent variable data (e.g., measurements).
        method : str, optional
            The detrending method to use. Options are:
                - "none": No detrending, returns x and zeros for y.
                - "linear": Applies a linear trend.
                - "poly2": Applies a quadratic (2nd degree polynomial) trend.
                - "moving_average": Applies a moving average trend.
            Default is "linear".
        window : int, optional
            The window size for the moving average method. Default is 5.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple containing:
                - The x values (possibly unchanged).
                - The trend values corresponding to y.
                - The detrended y values.

        Raises
        ------
        ValueError
            If an unknown detrending method is specified.

        """
        if method == "none":
            return np.asarray(x, float), np.zeros_like(np.asarray(y, float)), np.asarray(y, float)
        if method == "linear":
            return self.linear_trend(x, y)
        if method == "poly2":
            return self.poly2_trend(x, y)
        if method == "moving_average":
            return self.moving_average_trend(x, y, window=window)
        msg = f"Unknown detrending method: {method}"
        raise ValueError(msg)

    def plot_group_data(
        self,
        group: str,
        method: str = "linear",
        window: int = 5,
        m: int = 5,
        plot_style: str = "scatter",
    ) -> tuple[plt.Figure, str]:
        """
        Plots data for a specified group, with options for trend fitting and plot style.

        Parameters
        ----------
        group : str
            The group to plot, must be either 'group1' or 'group2'.
        method : str, optional
            The method used for trend fitting. Default is "linear".
            Use "none" to skip trend fitting.
        window : int, optional
            The window size for trend fitting (if applicable). Default is 5.
        m : int, optional
            Number of columns in the subplot grid. Default is 5.
        plot_style : str, optional
            The style of the plot, either "scatter" or "line". Default is "scatter".

        Returns
        -------
        fig : matplotlib.figure.Figure
            The matplotlib Figure object containing the plots.
        tmp_path : str
            The path to the saved temporary PDF file of the figure.

        Raises
        ------
        ValueError
            If the group argument is not 'group1' or 'group2'.

        """
        if group == "group1":
            ids, _replicates, data = self.get_group1_ids_replicates_data()
            color = self.color_group1
            group_label = self.group1_label
        elif group == "group2":
            ids, _replicates, data = self.get_group2_ids_replicates_data()
            color = self.color_group2
            group_label = self.group2_label
        else:
            msg = "group must be 'group1' or 'group2'"
            raise ValueError(msg)

        n_group = len(np.unique(ids))
        n_cols = m
        n_rows = int(np.ceil(n_group / n_cols))

        study_list = np.unique(ids).tolist()
        fig = plt.figure(figsize=(5 * n_cols / 2.54, 5 * n_rows / 2.54))

        for i, id_curr in enumerate(study_list):
            mask = np.array(ids) == id_curr
            n_reps = int(np.sum(mask))
            ax = fig.add_subplot(n_rows, n_cols, i + 1)

            for j in range(n_reps):
                x = self.time
                y = data[:, mask][:, j]

                if plot_style == "scatter":
                    ax.scatter(x, y, s=4, alpha=0.8, color=color)
                else:
                    ax.plot(x, y, color=color)

                valid_mask = ~np.isnan(y)
                x_fit = x[valid_mask]
                y_fit = y[valid_mask]

                x_processed, trend, _y_detrended = self.get_trend(
                    x_fit,
                    y_fit,
                    method=method,
                    window=window,
                )
                if method != "none":
                    ax.plot(
                        x_processed,
                        trend,
                        color="black",
                        linestyle="--",
                        linewidth=0.8,
                    )

            ax.set_title(f"ID: {id_curr} (n={n_reps}) - {group_label}")
            ax.set_xlabel("Time (h)")
            if i % n_cols == 0:
                ax.set_ylabel("Expression")

        plt.tight_layout()

        fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)

        fig.savefig(tmp_path)

        return fig, tmp_path
