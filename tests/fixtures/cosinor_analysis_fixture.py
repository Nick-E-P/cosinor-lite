"""Unit tests for cosinor analysis utilities."""

from __future__ import annotations

import pytest

from cosinor_lite.livecell_cosinor_analysis import CosinorAnalysis
from cosinor_lite.livecell_dataset import LiveCellDataset


@pytest.fixture(scope="session")
def cosinor_analysis(bioluminescence_dataset: LiveCellDataset) -> CosinorAnalysis:
    """Return a ``CosinorAnalysis`` instance seeded from the dataset fixture."""
    return CosinorAnalysis(
        ids=bioluminescence_dataset.ids,
        group=bioluminescence_dataset.group,
        replicate=bioluminescence_dataset.replicate,
        time_series=bioluminescence_dataset.time_series,
        time=bioluminescence_dataset.time,
        group1_label=bioluminescence_dataset.group1_label,
        group2_label=bioluminescence_dataset.group2_label,
        color_group1=bioluminescence_dataset.color_group1,
        color_group2=bioluminescence_dataset.color_group2,
    )
