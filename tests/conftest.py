"""Register pytest plugins, fixtures, and hooks to be used during test execution."""

# module import paths to python files containing fixtures
pytest_plugins = [
    "tests.fixtures.livecell_dataset_fixture",
    "tests.fixtures.cosinor_analysis_fixture",
    "tests.fixtures.omics_dataset_fixture",
]
