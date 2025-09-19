import numpy as np
import pandas as pd

W: float = 2 * np.pi / 24.0
RAD2H: float = 24.0 / (2.0 * np.pi)


def phase_from_ab(a: float, b: float) -> float:
    return (np.arctan2(b, a) * RAD2H) % 24.0


def amp_from_ab(a: float, b: float) -> float:
    return float(np.hypot(a, b))


def bic(llf: float, k: int, n: int) -> float:
    return k * np.log(n) - 2.0 * llf


def akaike_weights_from_bics(bics: np.ndarray) -> np.ndarray:
    d: np.ndarray = bics - np.nanmin(bics)
    w: np.ndarray = np.exp(-0.5 * d)
    return w / np.nansum(w)


def build_design(
    alpha_vals: np.ndarray,
    beta_vals: np.ndarray,
    t_cond1: np.ndarray,
    t_cond2: np.ndarray,
) -> pd.DataFrame:
    df_cond1: pd.DataFrame = pd.DataFrame({"y": alpha_vals, "time": t_cond1, "dataset": "alpha"}).dropna()
    df_cond2: pd.DataFrame = pd.DataFrame({"y": beta_vals, "time": t_cond2, "dataset": "beta"}).dropna()
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
    df_cond1: pd.DataFrame = pd.DataFrame({"y": alpha_vals, "time": t_cond1, "dataset": "alpha"}).dropna()
    df: pd.DataFrame = pd.concat([df_cond1], ignore_index=True)
    df["constant"] = 1.0
    df["cos_wt"] = np.cos(W * df["time"].to_numpy().astype(float))
    df["sin_wt"] = np.sin(W * df["time"].to_numpy().astype(float))
    return df


def build_design_cond2(
    beta_vals: np.ndarray,
    t_cond2: np.ndarray,
) -> pd.DataFrame:
    df_cond2: pd.DataFrame = pd.DataFrame({"y": beta_vals, "time": t_cond2, "dataset": "beta"}).dropna()
    df: pd.DataFrame = pd.concat([df_cond2], ignore_index=True)
    df["constant"] = 1.0
    df["cos_wt"] = np.cos(W * df["time"].to_numpy().astype(float))
    df["sin_wt"] = np.sin(W * df["time"].to_numpy().astype(float))
    return df
