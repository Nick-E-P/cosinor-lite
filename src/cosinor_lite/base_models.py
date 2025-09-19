from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd
import statsmodels.formula.api as smf
from pydantic.dataclasses import dataclass


@dataclass
class ModelResult:
    name: int
    llf: float
    bic: float
    alpha_phase: float
    alpha_amp: float
    beta_phase: float
    beta_amp: float


@dataclass
class ModelResultOneCondition:
    name: int
    llf: float
    bic: float
    phase: float
    amp: float


class BaseModel(ABC):
    name: str
    k: int
    formula: str

    def fit(self, df: pd.DataFrame) -> ModelResult:
        # df must already contain: y, constant, cos_wt, sin_wt, is_alpha, is_beta
        model = smf.ols(self.formula, data=df).fit()
        n = len(df)
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
    name: str
    k: int
    formula: str

    def fit(self, df: pd.DataFrame) -> ModelResultOneCondition:
        # df must already contain: y, constant, cos_wt, sin_wt, is_alpha, is_beta
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
