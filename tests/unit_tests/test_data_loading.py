# %%
from pathlib import Path

import pandas as pd
from pandas import DataFrame

DATA_DIR = Path(__file__).parent.parent / "data"
sample_csv = DATA_DIR / "Bioluminescence_test_data_1.csv"


def test_sample_csv_exists() -> None:
    if not sample_csv.exists():
        raise FileNotFoundError(sample_csv)


def test_bioluminescence_csv_can_be_loaded() -> None:
    df_bioluminescence: DataFrame = pd.read_csv(sample_csv, header=None)
    if df_bioluminescence.empty:
        msg = "Loaded DataFrame is empty"
        raise ValueError(msg)
    if df_bioluminescence.shape[1] <= 0:
        msg = "Loaded DataFrame has no columns"
        raise ValueError(msg)
