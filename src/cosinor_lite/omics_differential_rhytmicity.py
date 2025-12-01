from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass as std_dataclass
from typing import Self, cast

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from pydantic import ConfigDict, field_validator, model_validator
from pydantic.dataclasses import dataclass as pydantic_dataclass
from tqdm import tqdm

from cosinor_lite.omics_dataset import OmicsDataset

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


W: float = 2 * np.pi / 24.0
RAD2H: float = 24.0 / (2.0 * np.pi)


def phase_from_ab(a: float, b: float) -> float:
    """
    Convert cosine and sine coefficients to a phase value in hours.

    Parameters
    ----------
    a : float
        Cosine-term regression coefficient.
    b : float
        Sine-term regression coefficient.

    Returns
    -------
    float
        Phase value constrained to the ``[0, 24)`` hour interval.

    """
    return (np.arctan2(b, a) * RAD2H) % 24.0


def amp_from_ab(a: float, b: float) -> float:
    """
    Compute the amplitude implied by cosine and sine coefficients.

    Parameters
    ----------
    a : float
        Cosine-term regression coefficient.
    b : float
        Sine-term regression coefficient.

    Returns
    -------
    float
        Non-negative amplitude derived from the Euclidean norm.

    """
    return float(np.hypot(a, b))


def bic(llf: float, k: int, n: int) -> float:
    """
    Evaluate the Bayesian Information Criterion for a fitted model.

    Parameters
    ----------
    llf : float
        Log-likelihood of the fitted model.
    k : int
        Number of estimated parameters.
    n : int
        Number of observations used during fitting.

    Returns
    -------
    float
        BIC score, where lower values indicate preferred models.

    """
    return k * np.log(n) - 2.0 * llf


@std_dataclass
class ModelResult:
    """Store parameter estimates and metrics for dual-condition models."""

    name: int
    llf: float
    bic: float
    alpha_phase: float
    alpha_amp: float
    beta_phase: float
    beta_amp: float


@std_dataclass
class ModelResultOneCondition:
    """Store parameter estimates and metrics for single-condition models."""

    name: int
    llf: float
    bic: float
    phase: float
    amp: float


class BaseModel(ABC):
    """Template for two-condition cosinor models sharable across datasets."""

    name: int
    k: int
    formula: str

    def fit(self, df: pd.DataFrame) -> ModelResult:
        """
        Fit the configured statsmodels OLS formula to the supplied design matrix.

        Parameters
        ----------
        df : pandas.DataFrame
            Design matrix containing predictors and response values.

        Returns
        -------
        ModelResult
            Bundle of model metadata, BIC, and derived phase/amplitude values.

        """
        model = smf.ols(self.formula, data=df).fit()
        n: int = len(df)
        alpha_phase, alpha_amp, beta_phase, beta_amp = self.extract(model.params)
        return ModelResult(
            name=self.name,
            llf=model.llf,
            bic=bic(model.llf, self.k, n),
            alpha_phase=alpha_phase,
            alpha_amp=alpha_amp,
            beta_phase=beta_phase,
            beta_amp=beta_amp,
        )

    @abstractmethod
    def extract(self, params: pd.Series) -> tuple[float, float, float, float]: ...


class BaseModelOneCondition(ABC):
    """Template for single-condition cosinor models."""

    name: int
    k: int
    formula: str

    def fit(self, df: pd.DataFrame) -> ModelResultOneCondition:
        """
        Fit a single-condition OLS cosinor model on the provided design matrix.

        Parameters
        ----------
        df : pandas.DataFrame
            Design matrix containing predictors and response values.

        Returns
        -------
        ModelResultOneCondition
            Bundle of model metadata plus the fitted phase and amplitude.

        """
        model = smf.ols(self.formula, data=df).fit()
        n = len(df)
        phase, amp = self.extract(model.params)
        return ModelResultOneCondition(
            name=self.name,
            llf=model.llf,
            bic=bic(model.llf, self.k, n),
            phase=phase,
            amp=amp,
        )

    @abstractmethod
    def extract(self, params: pd.Series) -> tuple[float, float]: ...


@std_dataclass
class M0:
    """Fallback result used when all cosinor models fall below the weight cutoff."""

    name: int = 0
    alpha_phase: float = np.nan
    alpha_amp: float = np.nan
    beta_phase: float = np.nan
    beta_amp: float = np.nan
    amp: float = np.nan
    phase: float = np.nan
    bic: float = np.nan


class M1(BaseModel):
    name: int = 1
    k: int = 3
    formula: str = "y ~ is_alpha:constant + is_beta:constant -1"

    def extract(self, params: pd.Series) -> tuple[float, float, float, float]:  # noqa: ARG002
        return (np.nan, np.nan, np.nan, np.nan)


class M1OneCondition(BaseModelOneCondition):
    name: int = 1
    k: int = 1
    formula: str = "y ~ 1"

    def extract(self, params: pd.Series) -> tuple[float, float]:  # noqa: ARG002
        return (np.nan, np.nan)


class M2(BaseModel):
    name: int = 2
    k: int = 5
    formula: str = "y ~ is_alpha:constant + is_beta:constant + is_beta:cos_wt + is_beta:sin_wt -1"

    def extract(self, params: pd.Series) -> tuple[float, float, float, float]:
        a: float = cast(float, params["is_beta:cos_wt"])
        b: float = cast(float, params["is_beta:sin_wt"])
        beta_phase: float = phase_from_ab(a, b)
        beta_amp: float = amp_from_ab(a, b)
        return (np.nan, np.nan, beta_phase, beta_amp)


class M3(BaseModel):
    name: int = 3
    k: int = 5
    formula: str = "y ~ is_alpha:constant + is_beta:constant + is_alpha:cos_wt + is_alpha:sin_wt -1"

    def extract(self, params: pd.Series) -> tuple[float, float, float, float]:
        a: float = cast(float, params["is_alpha:cos_wt"])
        b: float = cast(float, params["is_alpha:sin_wt"])
        alpha_phase: float = phase_from_ab(a, b)
        alpha_amp: float = amp_from_ab(a, b)
        return (alpha_phase, alpha_amp, np.nan, np.nan)


class MOscOneCondition(BaseModelOneCondition):
    name: int = 3
    k: int = 3
    formula: str = "y ~ 1 + cos_wt + sin_wt"

    def extract(self, params: pd.Series) -> tuple[float, float]:
        a: float = cast(float, params["cos_wt"])
        b: float = cast(float, params["sin_wt"])
        phase: float = phase_from_ab(a, b)
        amp: float = amp_from_ab(a, b)
        return (phase, amp)


class M4(BaseModel):
    name: int = 4
    k: int = 5
    formula: str = "y ~ is_alpha:constant + is_beta:constant + cos_wt + sin_wt -1"

    def extract(self, params: pd.Series) -> tuple[float, float, float, float]:
        a: float = cast(float, params["cos_wt"])
        b: float = cast(float, params["sin_wt"])
        ph: float = phase_from_ab(a, b)
        am: float = amp_from_ab(a, b)
        return (ph, am, ph, am)


class M5(BaseModel):
    name: int = 5
    k: int = 7
    formula: str = (
        "y ~ is_alpha:constant + is_beta:constant + "
        "is_alpha:cos_wt + is_alpha:sin_wt + is_beta:cos_wt + is_beta:sin_wt -1"
    )

    def extract(self, params: pd.Series) -> tuple[float, float, float, float]:
        a_cond1: float = cast(float, params["is_alpha:cos_wt"])
        b_cond1: float = cast(float, params["is_alpha:sin_wt"])
        a_cond2: float = cast(float, params["is_beta:cos_wt"])
        b_cond2: float = cast(float, params["is_beta:sin_wt"])

        alpha_phase: float = phase_from_ab(a_cond1, b_cond1)
        alpha_amp: float = amp_from_ab(a_cond1, b_cond1)
        beta_phase: float = phase_from_ab(a_cond2, b_cond2)
        beta_amp: float = amp_from_ab(a_cond2, b_cond2)
        return (alpha_phase, alpha_amp, beta_phase, beta_amp)


MODELS: tuple[BaseModel, ...] = (M1(), M2(), M3(), M4(), M5())
MODELS_ONE_CONDITION: tuple[BaseModelOneCondition, ...] = (
    M1OneCondition(),
    MOscOneCondition(),
)


def akaike_weights_from_bics(bics: np.ndarray) -> np.ndarray:
    """
    Convert BIC scores into normalized Akaike weights.

    Parameters
    ----------
    bics : numpy.ndarray
        Array of BIC values where lower entries indicate better fits.

    Returns
    -------
    numpy.ndarray
        Non-negative weights that sum to one after exponentiation and scaling.

    """
    d: np.ndarray = bics - np.nanmin(bics)
    w: np.ndarray = np.exp(-0.5 * d)
    return w / np.nansum(w)


def _weight_mapping(model_ids: Sequence[int], weights: np.ndarray) -> dict[str, float]:
    """
    Align Akaike weights with ``w_model{n}`` keys expected in result tables.

    Parameters
    ----------
    model_ids : collections.abc.Sequence[int]
        Sequence of model identifiers that produced ``weights``.
    weights : numpy.ndarray
        Akaike weights in positional order matching ``model_ids``.

    Returns
    -------
    dict[str, float]
        Mapping of weight column names to values (or ``nan`` when absent).

    """
    mapping: dict[str, float] = {f"w_model{i}": np.nan for i in range(1, len(MODELS) + 1)}
    for idx, model_id in enumerate(model_ids):
        if idx < len(weights):
            mapping[f"w_model{model_id}"] = float(weights[idx])
    return mapping


def build_design(
    alpha_vals: np.ndarray,
    beta_vals: np.ndarray,
    t_cond1: np.ndarray,
    t_cond2: np.ndarray,
) -> pd.DataFrame:
    """
    Construct the full design matrix for simultaneously fitting both conditions.

    Parameters
    ----------
    alpha_vals : numpy.ndarray
        Expression values for condition 1 samples.
    beta_vals : numpy.ndarray
        Expression values for condition 2 samples.
    t_cond1 : numpy.ndarray
        Zeitgeber time points corresponding to ``alpha_vals``.
    t_cond2 : numpy.ndarray
        Zeitgeber time points corresponding to ``beta_vals``.

    Returns
    -------
    pandas.DataFrame
        Design matrix including indicator, cosine, and sine predictors.

    """
    df_cond1: pd.DataFrame = pd.DataFrame(
        {"y": alpha_vals, "time": t_cond1, "dataset": "alpha"},
    ).dropna()
    df_cond2: pd.DataFrame = pd.DataFrame(
        {"y": beta_vals, "time": t_cond2, "dataset": "beta"},
    ).dropna()
    df: pd.DataFrame = pd.concat([df_cond1, df_cond2], ignore_index=True)
    df["constant"] = 1.0
    df["cos_wt"] = np.cos(W * df["time"].to_numpy().astype(float))
    df["sin_wt"] = np.sin(W * df["time"].to_numpy().astype(float))
    df["is_alpha"] = (df["dataset"] == "alpha").astype(int)
    df["is_beta"] = (df["dataset"] == "beta").astype(int)
    return df


def build_design_cond1(
    alpha_vals: np.ndarray,
    t_cond1: np.ndarray,
) -> pd.DataFrame:
    """
    Construct the design matrix for condition 1 only.

    Parameters
    ----------
    alpha_vals : numpy.ndarray
        Expression values for condition 1 samples.
    t_cond1 : numpy.ndarray
        Zeitgeber time points associated with ``alpha_vals``.

    Returns
    -------
    pandas.DataFrame
        Single-condition design matrix with cosine and sine predictors.

    """
    df_cond1: pd.DataFrame = pd.DataFrame(
        {"y": alpha_vals, "time": t_cond1, "dataset": "alpha"},
    ).dropna()
    df: pd.DataFrame = pd.concat([df_cond1], ignore_index=True)
    df["constant"] = 1.0
    df["cos_wt"] = np.cos(W * df["time"].to_numpy().astype(float))
    df["sin_wt"] = np.sin(W * df["time"].to_numpy().astype(float))
    return df


def build_design_cond2(
    beta_vals: np.ndarray,
    t_cond2: np.ndarray,
) -> pd.DataFrame:
    """
    Construct the design matrix for condition 2 only.

    Parameters
    ----------
    beta_vals : numpy.ndarray
        Expression values for condition 2 samples.
    t_cond2 : numpy.ndarray
        Zeitgeber time points associated with ``beta_vals``.

    Returns
    -------
    pandas.DataFrame
        Single-condition design matrix with cosine and sine predictors.

    """
    df_cond2: pd.DataFrame = pd.DataFrame(
        {"y": beta_vals, "time": t_cond2, "dataset": "beta"},
    ).dropna()
    df: pd.DataFrame = pd.concat([df_cond2], ignore_index=True)
    df["constant"] = 1.0
    df["cos_wt"] = np.cos(W * df["time"].to_numpy().astype(float))
    df["sin_wt"] = np.sin(W * df["time"].to_numpy().astype(float))
    return df


@pydantic_dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class DifferentialRhythmicity:
    """
    Coordinate cosinor model selection across an :class:`OmicsDataset`.

    Parameters
    ----------
    dataset : OmicsDataset
        Dataset containing expression values and metadata required for fitting.
    BIC_cutoff : float, optional
        Minimum Akaike weight threshold for accepting a rhythmic model, by default ``0.5``.

    """

    dataset: OmicsDataset
    BIC_cutoff: float = 0.5

    @property
    def df(self) -> pd.DataFrame:
        """pandas.DataFrame: Alias to the underlying dataset table."""
        return self.dataset.df

    @property
    def columns_cond1(self) -> list[str]:
        """list[str]: Column names belonging to condition 1."""
        return self.dataset.columns_cond1

    @property
    def columns_cond2(self) -> list[str]:
        """list[str]: Column names belonging to condition 2."""
        return self.dataset.columns_cond2

    @property
    def t_cond1(self) -> np.ndarray:
        """numpy.ndarray: Zeitgeber time points for condition 1."""
        return self.dataset.t_cond1

    @property
    def t_cond2(self) -> np.ndarray:
        """numpy.ndarray: Zeitgeber time points for condition 2."""
        return self.dataset.t_cond2

    @property
    def cond1_label(self) -> str:
        """str: Human-readable label for condition 1."""
        return self.dataset.cond1_label

    @property
    def cond2_label(self) -> str:
        """str: Human-readable label for condition 2."""
        return self.dataset.cond2_label

    def rhythmic_genes_expressed_both(self, progress: gr.Progress | None = None) -> pd.DataFrame:
        """
        Evaluate rhythmicity for genes expressed in both conditions.

        Parameters
        ----------
        progress : gradio.Progress | None, optional
            Progress reporter used when running inside a Gradio app, by default ``None``.

        Returns
        -------
        pandas.DataFrame
            Per-gene summaries containing model identity, weights, and fitted parameters.

        """
        df: pd.DataFrame = self.df
        mask: pd.Series = (df["is_expressed_cond1"]) & (df["is_expressed_cond2"])
        df_to_analyse: pd.DataFrame = df[mask].reset_index(drop=True)

        progress_manager = progress or gr.Progress()
        model_ids: list[int] = [model.name for model in MODELS]
        rows: list[dict] = []
        for gene, row in progress_manager.tqdm(
            df_to_analyse.set_index("Genes").iterrows(),
            total=len(df_to_analyse),
            desc="Fitting models to genes expressed in both conditions",
        ):
            alpha_vec: np.ndarray = row[self.columns_cond1].to_numpy(float)
            beta_vec: np.ndarray = row[self.columns_cond2].to_numpy(float)
            design: pd.DataFrame = build_design(
                alpha_vec,
                beta_vec,
                self.t_cond1,
                self.t_cond2,
            )
            results: list[ModelResult] = [model.fit(design) for model in MODELS]
            bics: np.ndarray = np.array([result.bic for result in results], dtype=float)
            weights: np.ndarray = akaike_weights_from_bics(bics)
            best_result: ModelResult | M0
            if float(np.nanmax(weights)) < self.BIC_cutoff:
                best_result = M0()
                chosen_model_biw: float = np.nan
                model: int = 0
            else:
                pick: int = int(np.nanargmax(weights))
                best_result = results[pick]
                chosen_model_biw = float(weights[pick])
                model = int(best_result.name)
            weight_columns = _weight_mapping(model_ids, weights)
            rows.append(
                {
                    "gene": gene,
                    "model": model,
                    "chosen_model_bicw": chosen_model_biw,
                    # "weight": weights[pick],
                    **weight_columns,
                    "alpha_phase": best_result.alpha_phase,
                    "alpha_amp": best_result.alpha_amp,
                    "beta_phase": best_result.beta_phase,
                    "beta_amp": best_result.beta_amp,
                },
            )

        df_results: pd.DataFrame = pd.DataFrame(rows)
        df_results["subclass"] = "c"

        return df_results

    def rhythmic_genes_expressed_cond1(self) -> pd.DataFrame:
        """
        Evaluate rhythmicity for genes uniquely expressed in condition 1.

        Returns
        -------
        pandas.DataFrame
            Per-gene summaries containing model identity, weights, and fitted parameters.

        """
        df: pd.DataFrame = self.df
        mask: pd.Series = (df["is_expressed_cond1"]) & ~(df["is_expressed_cond2"])
        df_to_analyse: pd.DataFrame = df[mask].reset_index(drop=True)

        model_ids: list[int] = [model.name for model in MODELS_ONE_CONDITION]
        rows: list[dict] = []
        for gene, row in tqdm(
            df_to_analyse.set_index("Genes").iterrows(),
            total=len(df_to_analyse),
            desc="Fitting models to genes expressed in cond1 only",
        ):
            alpha_vec: np.ndarray = row[self.columns_cond1].to_numpy(float)
            design: pd.DataFrame = build_design_cond1(alpha_vec, self.t_cond1)
            results: list[ModelResultOneCondition] = [model.fit(design) for model in MODELS_ONE_CONDITION]
            bics: np.ndarray = np.array([result.bic for result in results], dtype=float)
            weights: np.ndarray = akaike_weights_from_bics(bics)
            best_result: ModelResultOneCondition | M0
            if float(np.nanmax(weights)) < self.BIC_cutoff:
                best_result = M0()
                chosen_model_biw: float = np.nan
                model: int = 0
            else:
                pick: int = int(np.nanargmax(weights))
                best_result = results[pick]
                chosen_model_biw = float(weights[pick])
                model = int(best_result.name)
            weight_columns = _weight_mapping(model_ids, weights)
            rows.append(
                {
                    "gene": gene,
                    "model": model,
                    "chosen_model_bicw": chosen_model_biw,
                    # "weight": weights[pick],
                    **weight_columns,
                    "alpha_phase": getattr(best_result, "phase", np.nan),
                    "alpha_amp": getattr(best_result, "amp", np.nan),
                    "beta_phase": np.nan,
                    "beta_amp": np.nan,
                },
            )

        df_results: pd.DataFrame = pd.DataFrame(rows)
        df_results["subclass"] = "a"

        return df_results

    def rhythmic_genes_expressed_cond2(self) -> pd.DataFrame:
        """
        Evaluate rhythmicity for genes uniquely expressed in condition 2.

        Returns
        -------
        pandas.DataFrame
            Per-gene summaries containing model identity, weights, and fitted parameters.

        """
        df: pd.DataFrame = self.df
        mask: pd.Series = ~(df["is_expressed_cond1"]) & (df["is_expressed_cond2"])
        df_to_analyse: pd.DataFrame = df[mask].reset_index(drop=True)

        model_ids: list[int] = [model.name for model in MODELS_ONE_CONDITION]
        rows: list[dict] = []
        for gene, row in tqdm(
            df_to_analyse.set_index("Genes").iterrows(),
            total=len(df_to_analyse),
            desc="Fitting models to genes expressed in cond2 only",
        ):
            beta_vec: np.ndarray = row[self.columns_cond2].to_numpy(float)
            design: pd.DataFrame = build_design_cond2(beta_vec, self.t_cond2)
            results: list[ModelResultOneCondition] = [model.fit(design) for model in MODELS_ONE_CONDITION]
            bics: np.ndarray = np.array([result.bic for result in results], dtype=float)
            weights: np.ndarray = akaike_weights_from_bics(bics)
            best_result: ModelResultOneCondition | M0
            if float(np.nanmax(weights)) < self.BIC_cutoff:
                best_result = M0()
                chosen_model_biw: float = np.nan
                model: int = 0
            else:
                pick: int = int(np.nanargmax(weights))
                best_result = results[pick]
                chosen_model_biw = float(weights[pick])
                model = int(best_result.name)
            weight_columns = _weight_mapping(model_ids, weights)
            rows.append(
                {
                    "gene": gene,
                    "model": model,
                    "chosen_model_bicw": chosen_model_biw,
                    # "weight": weights[pick],
                    **weight_columns,
                    "alpha_phase": np.nan,
                    "alpha_amp": np.nan,
                    "beta_phase": getattr(best_result, "phase", np.nan),
                    "beta_amp": getattr(best_result, "amp", np.nan),
                },
            )

        df_results: pd.DataFrame = pd.DataFrame(rows)
        df_results["subclass"] = "b"

        return df_results

    def extract_all_circadian_params(self) -> pd.DataFrame:
        """
        Combine rhythmicity results from all subsets into a single table.

        Returns
        -------
        pandas.DataFrame
            Dataset joined with cosinor parameters and model weights.

        """
        rhythmic_analysis_expressed_both = self.rhythmic_genes_expressed_both()
        rhythmic_analysis_cond1 = self.rhythmic_genes_expressed_cond1()

        rhythmic_analysis_cond2 = self.rhythmic_genes_expressed_cond2()

        results_total = pd.concat(
            [
                rhythmic_analysis_expressed_both,
                rhythmic_analysis_cond1,
                rhythmic_analysis_cond2,
            ],
        )

        df_pre_export = self.df.merge(
            results_total,
            left_on="Genes",
            right_on="gene",
            how="right",
        )
        column_list: list = [
            "Genes",
            *self.columns_cond1,
            *self.columns_cond2,
            "w_model1",
            "w_model2",
            "w_model3",
            "w_model4",
            "w_model5",
            "model",
            "subclass",
            "mean_cond1",
            "alpha_amp",
            "alpha_phase",
            "is_expressed_cond1",
            "mean_cond2",
            "beta_amp",
            "beta_phase",
            "is_expressed_cond2",
        ]

        df_export = df_pre_export[column_list]
        return df_export


@pydantic_dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class OmicsHeatmap:
    """
    Generate clustered heatmaps highlighting rhythmic expression patterns.

    Parameters
    ----------
    df : pandas.DataFrame
        Source table with model classifications and expression data.
    columns_cond1 : list[str]
        Column names for condition 1 replicates.
    columns_cond2 : list[str]
        Column names for condition 2 replicates.
    t_cond1 : numpy.ndarray
        Zeitgeber time points for condition 1 samples.
    t_cond2 : numpy.ndarray
        Zeitgeber time points for condition 2 samples.
    cond1_label : str, optional
        Human-readable label for condition 1, by default ``"cond1"``.
    cond2_label : str, optional
        Human-readable label for condition 2, by default ``"cond2"``.
    show_unexpressed : bool, optional
        Whether to include non-expressed entries when plotting, by default ``True``.

    """

    df: pd.DataFrame

    columns_cond1: list[str]
    columns_cond2: list[str]
    t_cond1: np.ndarray
    t_cond2: np.ndarray
    cond1_label: str = "cond1"
    cond2_label: str = "cond2"

    show_unexpressed: bool = True

    @field_validator("t_cond1", "t_cond2", mode="before")
    @classmethod
    def _to_1d_f64(cls, v: object) -> np.ndarray:
        """
        Coerce input to a one-dimensional ``float64`` NumPy array.

        Parameters
        ----------
        v : object
            Sequence-like structure representing time points.

        Returns
        -------
        numpy.ndarray
            Flattened array in ``float64`` precision.

        Raises
        ------
        ValueError
            If ``v`` cannot be interpreted as a one-dimensional array.

        """
        a = np.asarray(v, dtype=np.float64)
        if a.ndim != 1:
            text: str = "expected 1D array"
            raise ValueError(text)
        return a

    @model_validator(mode="after")
    def _check_columns(self) -> Self:
        """
        Confirm that required expression columns exist in the DataFrame.

        Returns
        -------
        Self
            The validated :class:`OmicsHeatmap` instance.

        Raises
        ------
        ValueError
            If any specified column names are absent.

        """
        missing = [c for c in (self.columns_cond1 + self.columns_cond2) if c not in self.df.columns]
        if missing:
            text = f"Missing columns: {missing}"
            raise ValueError(text)
        return self

    def timepoint_means(
        self,
        df: pd.DataFrame,
        columns: list[str],
        times: np.ndarray,
    ) -> np.ndarray:
        """
        Average replicates at identical time points for a collection of genes.

        Parameters
        ----------
        df : pandas.DataFrame
            Table containing the expression measurements.
        columns : list[str]
            Subset of column names whose values should be aggregated.
        times : numpy.ndarray
            Zeitgeber time labels corresponding to ``columns``.

        Returns
        -------
        numpy.ndarray
            Array where each column represents the mean expression at a unique time.

        Raises
        ------
        ValueError
            If the number of columns differs from the number of time labels provided.

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
        """
        Z-score normalize each gene profile independently.

        Parameters
        ----------
        arr : numpy.ndarray
            Matrix of expression values where rows correspond to genes.

        Returns
        -------
        numpy.ndarray
            Z-score normalized matrix preserving the original shape.

        """
        means: np.ndarray = np.mean(arr, axis=1, keepdims=True)
        stds: np.ndarray = np.std(arr, axis=1, keepdims=True)
        safe_stds: np.ndarray = np.where(stds == 0, 1, stds)
        return (arr - means) / safe_stds

    def plot_heatmap(self, cmap: str = "bwr") -> plt.Figure:  # noqa: PLR0915
        """
        Render the full alpha/beta heatmap split by rhythmic subclasses.

        Parameters
        ----------
        cmap : str, optional
            Matplotlib colormap name used for heat intensity, by default ``"bwr"``.

        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the subplot grid of heatmaps.

        """
        df: pd.DataFrame = self.df
        df_sorted: pd.DataFrame = df.sort_values(by=["alpha_phase", "beta_phase"]).reset_index(
            drop=True,
        )

        t_unique: np.ndarray = np.unique(self.t_cond1).astype(int)

        mean_cond1: np.ndarray = self.timepoint_means(
            df_sorted,
            self.columns_cond1,
            self.t_cond1,
        )
        mean_cond2: np.ndarray = self.timepoint_means(
            df_sorted,
            self.columns_cond2,
            self.t_cond2,
        )

        z_cond1: np.ndarray = self.get_z_score(mean_cond1)
        z_cond2: np.ndarray = self.get_z_score(mean_cond2)

        total_rows: int = (df_sorted["model"].isin([2, 3, 4, 5])).sum()

        m2: int = 2
        m3: int = 3
        m4: int = 4
        m5: int = 5

        n_m2a: int = ((df_sorted["model"] == m2) & (df_sorted["subclass"] == "b")).sum()
        n_m2b: int = ((df_sorted["model"] == m2) & (df_sorted["subclass"] == "c")).sum()
        n_m3a: int = ((df_sorted["model"] == m3) & (df_sorted["subclass"] == "a")).sum()
        n_m3b: int = ((df_sorted["model"] == m3) & (df_sorted["subclass"] == "c")).sum()
        n_m4: int = (df_sorted["model"] == m4).sum()
        n_m5: int = (df_sorted["model"] == m5).sum()

        fig, axes = plt.subplots(
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
                "width_ratios": [1, 1],
            },
            figsize=(12 / 2.54, 18 / 2.54),
        )

        vmin_global = -2.5
        vmax_global = 2.5

        mask = (df_sorted["model"] == m2) & (df_sorted["subclass"] == "b")
        z_cond1_filtered = z_cond1[mask.to_numpy()].copy()
        if not self.show_unexpressed:
            z_cond1_filtered[:] = 0
        axes[0, 0].imshow(
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

        axes[0, 0].text(
            -0.2,
            0.5,
            "M2a",
            transform=axes[0, 0].transAxes,
            ha="center",
            va="center",
            rotation=0,
            fontsize=8,
        )

        mask = (df_sorted["model"] == m2) & (df_sorted["subclass"] == "c")
        z_cond1_filtered = z_cond1[mask.to_numpy()]
        axes[1, 0].imshow(
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

        axes[1, 0].text(
            -0.2,
            0.5,
            "M2b",
            transform=axes[1, 0].transAxes,
            ha="center",
            va="center",
            rotation=0,
            fontsize=8,
        )
        mask = (df_sorted["model"] == m3) & (df_sorted["subclass"] == "a")
        z_cond1_filtered = z_cond1[mask.to_numpy()]
        axes[2, 0].imshow(
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

        axes[2, 0].text(
            -0.2,
            0.5,
            "M3a",
            transform=axes[2, 0].transAxes,
            ha="center",
            va="center",
            rotation=0,
            fontsize=8,
        )
        mask = (df_sorted["model"] == m3) & (df_sorted["subclass"] == "c")
        z_cond1_filtered = z_cond1[mask.to_numpy()]
        axes[3, 0].imshow(
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

        axes[3, 0].text(
            -0.2,
            0.5,
            "M3b",
            transform=axes[3, 0].transAxes,
            ha="center",
            va="center",
            rotation=0,
            fontsize=8,
        )

        mask = df_sorted["model"] == m4
        z_cond1_filtered = z_cond1[mask.to_numpy()]
        axes[4, 0].imshow(
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
            "M4",
            transform=axes[4, 0].transAxes,
            ha="center",
            va="center",
            rotation=0,
            fontsize=8,
        )

        mask = df_sorted["model"] == m5
        z_cond1_filtered = z_cond1[mask.to_numpy()]
        axes[5, 0].imshow(
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
            "M5",
            transform=axes[5, 0].transAxes,
            ha="center",
            va="center",
            rotation=0,
            fontsize=8,
        )

        mask = (df_sorted["model"] == m2) & (df_sorted["subclass"] == "b")
        z_cond2_filtered = z_cond2[mask.to_numpy()]
        axes[0, 1].imshow(
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

        mask = (df_sorted["model"] == m2) & (df_sorted["subclass"] == "c")
        z_cond2_filtered = z_cond2[mask.to_numpy()]
        axes[1, 1].imshow(
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

        mask = (df_sorted["model"] == m3) & (df_sorted["subclass"] == "a")
        z_cond2_filtered = z_cond2[mask.to_numpy()].copy()
        if not self.show_unexpressed:
            z_cond2_filtered[:] = 0
        axes[2, 1].imshow(
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

        mask = (df_sorted["model"] == m3) & (df_sorted["subclass"] == "c")
        z_cond2_filtered = z_cond2[mask.to_numpy()]
        axes[3, 1].imshow(
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

        mask = df_sorted["model"] == m4
        z_cond2_filtered = z_cond2[mask.to_numpy()]
        axes[4, 1].imshow(
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

        mask = df_sorted["model"] == m5
        z_cond2_filtered = z_cond2[mask.to_numpy()]
        axes[5, 1].imshow(
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

        return fig


@pydantic_dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class TimeSeriesExample:
    """
    Plot raw observations alongside cosinor model predictions for genes.

    Parameters
    ----------
    df : pandas.DataFrame
        Table containing expression measurements and model metadata.
    columns_cond1 : list[str]
        Column names for condition 1 samples.
    columns_cond2 : list[str]
        Column names for condition 2 samples.
    t_cond1 : numpy.ndarray
        Zeitgeber times corresponding to condition 1 measurements.
    t_cond2 : numpy.ndarray
        Zeitgeber times corresponding to condition 2 measurements.
    cond1_label : str, optional
        Display label for condition 1, by default ``"cond1"``.
    cond2_label : str, optional
        Display label for condition 2, by default ``"cond2"``.
    deduplicate_on_init : bool, optional
        Whether the underlying dataset should deduplicate genes immediately, by default ``False``.

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
    def _to_1d_f64(cls, v: object) -> np.ndarray:
        """
        Convert an iterable of time points into a one-dimensional ``float64`` array.

        Parameters
        ----------
        v : object
            Sequence-like input representing time points.

        Returns
        -------
        numpy.ndarray
            Flattened array of times in ``float64`` precision.

        Raises
        ------
        ValueError
            If the supplied object cannot be interpreted as one-dimensional.

        """
        a = np.asarray(v, dtype=np.float64)
        if a.ndim != 1:
            text: str = "expected 1D array"
            raise ValueError(text)
        return a

    @model_validator(mode="after")
    def _check_columns(self) -> Self:
        """
        Ensure both condition-specific column sets appear in the DataFrame.

        Returns
        -------
        Self
            The validated :class:`TimeSeriesExample` instance.

        Raises
        ------
        ValueError
            If any required column is missing from ``df``.

        """
        missing = [c for c in (self.columns_cond1 + self.columns_cond2) if c not in self.df.columns]
        if missing:
            text = f"Missing columns: {missing}"
            raise ValueError(text)
        return self

    def plot_time_series(self, gene: str, xticks: np.ndarray | None = None) -> None:
        """
        Visualize observed expression values with corresponding model fits.

        Parameters
        ----------
        gene : str
            Gene identifier to visualize.
        xticks : numpy.ndarray | None, optional
            Custom x-axis tick positions, by default ``None`` for automatic values.

        Raises
        ------
        ValueError
            If the requested gene is absent from the DataFrame.

        """
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

        plt.title(f"Time Series for {gene}")
        plt.xlabel("Time (h)")
        plt.subplot(1, 2, 2)
        plt.scatter(self.t_cond2, beta_vec, s=4, color="r")
        plt.plot(t_test, y_test_cond2, color="r")
        plt.title(f"Time Series for {gene}")
        plt.xlabel("Time (h)")

        plt.xticks(xticks)
        plt.xlim(
            xticks[0] - 0.05 * (xticks[-1] - xticks[0]),
            xticks[-1] + 0.05 * (xticks[-1] - xticks[0]),
        )
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
        """
        Produce fitted curves for genes expressed in both conditions.

        Parameters
        ----------
        alpha_vec : numpy.ndarray
            Expression values for condition 1 samples.
        beta_vec : numpy.ndarray
            Expression values for condition 2 samples.
        t_cond1 : numpy.ndarray
            Zeitgeber times for ``alpha_vec``.
        t_cond2 : numpy.ndarray
            Zeitgeber times for ``beta_vec``.
        model : int
            Model identifier selected for the gene.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
            Common time grid plus fitted values for condition 1 and condition 2.

        """
        design: pd.DataFrame = build_design(alpha_vec, beta_vec, t_cond1, t_cond2)
        t_test: np.ndarray = np.linspace(0, 24, 100)
        y_test_cond1: np.ndarray
        y_test_cond2: np.ndarray
        if model == 0:
            print("No model selected")
            y_test_cond1 = np.full_like(t_test, np.nan)
            y_test_cond2 = np.full_like(t_test, np.nan)
        else:
            model_formula: str = MODELS[model - 1].formula
            res = smf.ols(model_formula, data=design).fit()
            df_cond1: pd.DataFrame = pd.DataFrame(
                {"time": t_test, "dataset": "alpha"},
            ).dropna()
            df_cond2: pd.DataFrame = pd.DataFrame(
                {"time": t_test, "dataset": "beta"},
            ).dropna()
            df_test: pd.DataFrame = pd.concat([df_cond1, df_cond2], ignore_index=True)
            df_test["constant"] = 1.0
            df_test["cos_wt"] = np.cos(W * df_test["time"].to_numpy().astype(float))
            df_test["sin_wt"] = np.sin(W * df_test["time"].to_numpy().astype(float))
            df_test["is_alpha"] = (df_test["dataset"] == "alpha").astype(int)
            df_test["is_beta"] = (df_test["dataset"] == "beta").astype(int)
            y_test = res.predict(exog=df_test)
            y_test_cond1 = y_test[df_test["dataset"] == "alpha"].to_numpy()
            y_test_cond2 = y_test[df_test["dataset"] == "beta"].to_numpy()
        return t_test, y_test_cond1, y_test_cond2

    def get_test_function_expressed_cond1(
        self,
        alpha_vec: np.ndarray,
        t_cond1: np.ndarray,
        model: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Produce fitted curves for genes expressed only in condition 1.

        Parameters
        ----------
        alpha_vec : numpy.ndarray
            Expression values for condition 1 samples.
        t_cond1 : numpy.ndarray
            Zeitgeber times for ``alpha_vec``.
        model : int
            Model identifier selected for the gene.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
            Common time grid, fitted values for condition 1, and ``nan`` placeholders for condition 2.

        """
        design: pd.DataFrame = build_design_cond1(alpha_vec, t_cond1)
        t_test: np.ndarray = np.linspace(0, 24, 100)
        y_test_cond1: np.ndarray
        y_test_cond2: np.ndarray
        if model == 0:
            print("No model selected")
            y_test_cond1 = np.full_like(t_test, np.nan)
            y_test_cond2 = np.full_like(t_test, np.nan)
        else:
            print(f"Fitting model {model}")
            print(MODELS_ONE_CONDITION)
            model_index: int = 0 if model == 1 else 1
            model_formula: str = MODELS_ONE_CONDITION[model_index].formula
            res = smf.ols(model_formula, data=design).fit()
            df_test: pd.DataFrame = pd.DataFrame(
                {"time": t_test, "dataset": "alpha"},
            ).dropna()
            df_test["constant"] = 1.0
            df_test["cos_wt"] = np.cos(W * df_test["time"].to_numpy().astype(float))
            df_test["sin_wt"] = np.sin(W * df_test["time"].to_numpy().astype(float))
            y_test = res.predict(exog=df_test).to_numpy()
            y_test_cond1 = y_test
            y_test_cond2 = np.full_like(t_test, np.nan)
        return t_test, y_test_cond1, y_test_cond2

    def get_test_function_expressed_cond2(
        self,
        beta_vec: np.ndarray,
        t_cond2: np.ndarray,
        model: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Produce fitted curves for genes expressed only in condition 2.

        Parameters
        ----------
        beta_vec : numpy.ndarray
            Expression values for condition 2 samples.
        t_cond2 : numpy.ndarray
            Zeitgeber times for ``beta_vec``.
        model : int
            Model identifier selected for the gene.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
            Common time grid, ``nan`` placeholders for condition 1, and fitted values for condition 2.

        """
        design: pd.DataFrame = build_design_cond2(beta_vec, t_cond2)
        t_test: np.ndarray = np.linspace(0, 24, 100)
        y_test_cond1: np.ndarray
        y_test_cond2: np.ndarray
        if model == 0:
            print("No model selected")
            y_test_cond1 = np.full_like(t_test, np.nan)
            y_test_cond2 = np.full_like(t_test, np.nan)
        else:
            print(f"Fitting model {model}")
            print(MODELS_ONE_CONDITION)
            model_index: int = 0 if model == 1 else 1
            model_formula: str = MODELS_ONE_CONDITION[model_index].formula
            res = smf.ols(model_formula, data=design).fit()
            df_test: pd.DataFrame = pd.DataFrame(
                {"time": t_test, "dataset": "alpha"},
            ).dropna()
            df_test["constant"] = 1.0
            df_test["cos_wt"] = np.cos(W * df_test["time"].to_numpy().astype(float))
            df_test["sin_wt"] = np.sin(W * df_test["time"].to_numpy().astype(float))
            y_test = res.predict(exog=df_test).to_numpy()
            y_test_cond1 = np.full_like(t_test, np.nan)
            y_test_cond2 = y_test

        return t_test, y_test_cond1, y_test_cond2
