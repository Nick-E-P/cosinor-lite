from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic.dataclasses import dataclass

from cosinor_lite.base_models import BaseModel, BaseModelOneCondition
from cosinor_lite.utils import amp_from_ab, phase_from_ab


@dataclass
class M0:
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

    def extract(self, params: pd.Series) -> tuple[float, float, float, float]:
        # No rhythmic terms → phases/amps are NaN
        return (np.nan, np.nan, np.nan, np.nan)


class M1OneCondition(BaseModelOneCondition):
    name: int = 1
    k: int = 1
    formula: str = "y ~ 1"

    def extract(self, params: pd.Series) -> tuple[float, float]:
        # No rhythmic terms → phases/amps are NaN
        return (np.nan, np.nan)


class M2(BaseModel):
    name: int = 2
    k: int = 5
    formula: str = "y ~ is_alpha:constant + is_beta:constant + is_beta:cos_wt + is_beta:sin_wt -1"

    def extract(self, params: pd.Series) -> tuple[float, float, float, float]:
        # beta has rhythmic terms, alpha does not
        a: float = params["is_beta:cos_wt"]
        b: float = params["is_beta:sin_wt"]
        beta_phase: float = phase_from_ab(a, b)
        beta_amp: float = amp_from_ab(a, b)
        return (np.nan, np.nan, beta_phase, beta_amp)


class M3(BaseModel):
    name: int = 3
    k: int = 5
    formula: str = "y ~ is_alpha:constant + is_beta:constant + is_alpha:cos_wt + is_alpha:sin_wt -1"

    def extract(self, params: pd.Series) -> tuple[float, float, float, float]:
        # alpha has rhythmic terms, beta does not
        a: float = params["is_alpha:cos_wt"]
        b: float = params["is_alpha:sin_wt"]
        alpha_phase: float = phase_from_ab(a, b)
        alpha_amp: float = amp_from_ab(a, b)
        return (alpha_phase, alpha_amp, np.nan, np.nan)


class MOscOneCondition(BaseModelOneCondition):
    name: int = 3
    k: int = 3
    formula: str = "y ~ 1 + cos_wt + sin_wt"

    def extract(self, params: pd.Series) -> tuple[float, float]:
        # alpha has rhythmic terms, beta does not
        a: float = params["cos_wt"]
        b: float = params["sin_wt"]
        phase: float = phase_from_ab(a, b)
        amp: float = amp_from_ab(a, b)
        return (phase, amp)


class M4(BaseModel):
    name: int = 4
    k: int = 5
    formula: str = "y ~ is_alpha:constant + is_beta:constant + cos_wt + sin_wt -1"

    def extract(self, params: pd.Series) -> tuple[float, float, float, float]:
        # shared rhythmic terms for both alpha and beta
        a: float = params["cos_wt"]
        b: float = params["sin_wt"]
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
        a_cond1: float = params["is_alpha:cos_wt"]
        b_cond1: float = params["is_alpha:sin_wt"]
        a_cond2: float = params["is_beta:cos_wt"]
        b_cond2: float = params["is_beta:sin_wt"]

        alpha_phase: float = phase_from_ab(a_cond1, b_cond1)
        alpha_amp: float = amp_from_ab(a_cond1, b_cond1)
        beta_phase: float = phase_from_ab(a_cond2, b_cond2)
        beta_amp: float = amp_from_ab(a_cond2, b_cond2)
        return (alpha_phase, alpha_amp, beta_phase, beta_amp)
