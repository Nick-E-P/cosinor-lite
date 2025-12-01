# tests/fixtures/livecell_dataset_fixture.py
import pandas as pd
import pytest

from cosinor_lite.livecell_dataset import LiveCellDataset
from tests.consts import PROJECT_DIR


@pytest.fixture(
    scope="session",
    params=[
        "bioluminescence_example.csv",
        "qpcr_example.csv",
        "single_cell_example.csv",
    ],
)
def bioluminescence_dataset(request: pytest.FixtureRequest) -> LiveCellDataset:
    """Return a LiveCellDataset built from several example files."""
    filename = request.param
    file = PROJECT_DIR / "tests" / "test_data" / filename

    df_data: pd.DataFrame = pd.read_csv(file, index_col=0, header=None)

    participant_id: pd.Series[str] = df_data.iloc[0, :].astype(str)
    replicate: pd.Series[int] = df_data.iloc[1, :].astype(int)
    group: pd.Series[str] = df_data.iloc[2, :].astype(str)
    time = df_data.index[3:].to_numpy(dtype=float)
    time_rows = df_data.iloc[3:].apply(pd.to_numeric)

    return LiveCellDataset(
        ids=participant_id.tolist(),
        group=group.tolist(),
        replicate=replicate.tolist(),
        time_series=time_rows.to_numpy(dtype=float),
        time=time,
        group1_label="Group 1",
        group2_label="Group 2",
    )
