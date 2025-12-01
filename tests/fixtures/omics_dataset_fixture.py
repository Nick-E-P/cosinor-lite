from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cosinor_lite.omics_dataset import OmicsDataset
from tests.consts import PROJECT_DIR


def _columns_for_condition(data: pd.DataFrame, suffix: str) -> list[str]:
    """Return expression columns that contain the given condition suffix."""
    return [column for column in data.columns if f"_{suffix}_" in column]


def _extract_timepoints(columns: list[str]) -> np.ndarray:
    """Extract zeitgeber time values from column names."""
    return np.asarray([float(column.split("_")[1]) for column in columns], dtype=float)


@pytest.fixture(scope="session")
def omics_dataset() -> OmicsDataset:
    """Return an ``OmicsDataset`` populated with alpha/beta expression data."""
    file = PROJECT_DIR / "tests" / "test_data" / "GSE95156_Alpha_Beta.csv"
    dataframe = pd.read_csv(file)
    columns_cond1 = _columns_for_condition(dataframe, "a")
    columns_cond2 = _columns_for_condition(dataframe, "b")

    dataset = OmicsDataset(
        df=dataframe.copy(),
        columns_cond1=columns_cond1,
        columns_cond2=columns_cond2,
        t_cond1=_extract_timepoints(columns_cond1),
        t_cond2=_extract_timepoints(columns_cond2),
        cond1_label="Alpha",
        cond2_label="Beta",
    )

    return dataset
