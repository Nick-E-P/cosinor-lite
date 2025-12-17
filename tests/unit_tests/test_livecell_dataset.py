import numpy as np
import pytest

from cosinor_lite.livecell_dataset import LiveCellDataset


def test_to_2d_f64_valid(bioluminescence_dataset: LiveCellDataset) -> None:
    arr = bioluminescence_dataset.time_series
    result = LiveCellDataset._to_2d_f64(arr)  # noqa: SLF001
    assert result.shape == arr.shape  # noqa: S101
    assert result.dtype == np.float64  # noqa: S101


def test_to_2d_f64_invalid() -> None:
    arr = np.ones(3)
    with pytest.raises(ValueError, match="expected 2D array"):
        LiveCellDataset._to_2d_f64(arr)  # noqa: SLF001


def test_check_columns_valid(bioluminescence_dataset: LiveCellDataset) -> None:
    ds = bioluminescence_dataset
    assert ds.time_series.shape[1] == len(ds.ids)  # noqa: S101
    assert ds.time_series.shape[1] == len(ds.group)  # noqa: S101
    assert ds.time_series.shape[1] == len(ds.replicate)  # noqa: S101


def test_check_columns_invalid() -> None:
    ids = ["A", "B"]
    group = ["group1", "group2"]
    replicate = [1, 2]
    time_series = np.ones((10, 3))
    time = np.arange(10)
    with pytest.raises(ValueError, match="Length of ids must match number of columns in time_series"):
        LiveCellDataset(
            ids=ids,
            group=group,
            replicate=replicate,
            time_series=time_series,
            time=time,
        )


def test_get_group1_ids_replicates_data(bioluminescence_dataset: LiveCellDataset) -> None:
    ds = bioluminescence_dataset
    ids, replicates, data = ds.get_group1_ids_replicates_data()
    assert set(ids).issubset(set(ds.ids))  # noqa: S101
    group_values = set(np.array(ds.group)[np.isin(ds.ids, ids)])
    assert group_values <= {ds.group1_label}  # noqa: S101
    assert data.shape[1] == len(ids)  # noqa: S101
    assert len(replicates) == len(ids)  # noqa: S101


def test_get_group2_ids_replicates_data(bioluminescence_dataset: LiveCellDataset) -> None:
    ds = bioluminescence_dataset
    ids, replicates, data = ds.get_group2_ids_replicates_data()
    assert set(ids).issubset(set(ds.ids))  # noqa: S101
    group_values = set(np.array(ds.group)[np.isin(ds.ids, ids)])
    assert group_values <= {ds.group2_label}  # noqa: S101
    assert data.shape[1] == len(ids)  # noqa: S101
    assert len(replicates) == len(ids)  # noqa: S101


def test_linear_trend(bioluminescence_dataset: LiveCellDataset) -> None:
    ds = bioluminescence_dataset
    x = ds.time
    y = ds.time_series[:, 0]
    mask = np.isfinite(x) & np.isfinite(y)
    x_valid = x[mask]
    y_valid = y[mask]
    x_fit, y_fit, y_detrended = ds.linear_trend(x_valid, y_valid)
    assert x_fit.shape == y_fit.shape == x_valid.shape  # noqa: S101
    assert np.all(np.isfinite(y_fit))  # noqa: S101
    assert y_detrended.shape == y_valid.shape  # noqa: S101
    assert np.all(np.isfinite(y_detrended))  # noqa: S101


def test_poly2_trend(bioluminescence_dataset: LiveCellDataset) -> None:
    ds = bioluminescence_dataset
    x = ds.time
    y = ds.time_series[:, 0]
    mask = np.isfinite(x) & np.isfinite(y)
    x_valid = x[mask]
    y_valid = y[mask]
    x_fit, y_fit, y_detrended = ds.poly2_trend(x_valid, y_valid)
    expected = np.polyval(np.polyfit(x_valid, y_valid, 2), x_valid)
    assert x_fit.shape == y_fit.shape == x_valid.shape  # noqa: S101
    assert np.allclose(y_fit, expected)  # noqa: S101
    assert y_detrended.shape == y_valid.shape  # noqa: S101
    assert np.all(np.isfinite(y_detrended))  # noqa: S101


def test_moving_average_trend(bioluminescence_dataset: LiveCellDataset) -> None:
    ds = bioluminescence_dataset
    x = ds.time
    y = ds.time_series[:, 0]
    mask = np.isfinite(x) & np.isfinite(y)
    x_ma, y_ma, y_detrended = ds.moving_average_trend(x[mask], y[mask], window=3)
    assert len(x_ma) == len(y_ma)  # noqa: S101
    assert len(x_ma) > 0  # noqa: S101
    assert np.all(np.isfinite(y_ma))  # noqa: S101
    assert y_detrended.shape == y_ma.shape  # noqa: S101
    assert np.all(np.isfinite(y_detrended))  # noqa: S101


def test_moving_average_trend_invalid_window(bioluminescence_dataset: LiveCellDataset) -> None:
    ds = bioluminescence_dataset
    x = ds.time
    y = ds.time_series[:, 0]
    with pytest.raises(ValueError, match="Window size must be at least 1"):
        ds.moving_average_trend(x, y, window=0)


@pytest.mark.parametrize("method", ["none", "linear", "poly2", "moving_average"])
def test_get_trend_methods(method: str, bioluminescence_dataset: LiveCellDataset) -> None:
    ds = bioluminescence_dataset
    x = ds.time
    y = ds.time_series[:, 0]
    mask = np.isfinite(x) & np.isfinite(y)
    x_valid = x[mask]
    y_valid = y[mask]
    x_out, y_out, y_detrended = ds.get_trend(x_valid, y_valid, method=method, window=3)
    assert isinstance(x_out, np.ndarray)  # noqa: S101
    assert isinstance(y_out, np.ndarray)  # noqa: S101
    assert x_out.shape == y_out.shape  # noqa: S101
    assert isinstance(y_detrended, np.ndarray)  # noqa: S101
    assert y_detrended.shape == y_out.shape  # noqa: S101
    if method != "moving_average":
        assert x_out.shape == x_valid.shape  # noqa: S101
    else:
        assert x_out.size <= x_valid.size  # noqa: S101


def test_get_trend_invalid_method(bioluminescence_dataset: LiveCellDataset) -> None:
    ds = bioluminescence_dataset
    x = ds.time
    y = ds.time_series[:, 0]
    mask = np.isfinite(x) & np.isfinite(y)
    with pytest.raises(ValueError, match="Unknown detrending method"):
        ds.get_trend(x[mask], y[mask], method="unknown")


@pytest.mark.parametrize("group", ["group1", "group2"])
def test_plot_group_data(bioluminescence_dataset: LiveCellDataset, group: str) -> None:
    fig, tmp_path, csv_path = bioluminescence_dataset.plot_group_data(
        group=group,
        method="linear",
        plot_style="scatter",
    )
    assert hasattr(fig, "savefig")  # noqa: S101
    assert tmp_path.endswith(".pdf")  # noqa: S101
    assert csv_path.endswith(".csv")  # noqa: S101


def test_plot_group_data_invalid_group(bioluminescence_dataset: LiveCellDataset) -> None:
    ds = bioluminescence_dataset
    with pytest.raises(ValueError, match="group must be 'group1' or 'group2'"):
        ds.plot_group_data("invalid_group")
