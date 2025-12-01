# cosinor-lite

`cosinor-lite` is a lightweight toolkit for exploring circadian oscillations across multiomics and live-cell imaging experiments. It relies on widely used statistical models and offers two modes of use.

- **Interactive Gradio dashboard** to drag & drop your data files.
- **Python package** for fine-grained customisation beyond the settings offered in the Gradio UI.

🚀 Try the interactive analysis pipeline on
**[Hugging Face Spaces → cosinor-lite](https://huggingface.co/spaces/nick-e-p/cosinor-lite)**.

## Why `cosinor-lite?`

- **Unified workflow for live cell data**: Perform detrending and fit cosinor models with either a fixed 24 h, free-period or damped oscillation with consistent APIs across data modalities (i.e. bioluminescence, cytokines etc.).
- **Differential rhythmicity across any omics type**: Compare two conditions for differential rhythmicity for any type of omics data using BIC model-selection strategies with widespread use in the circadian literature.
- **Create publication-ready analysis with a simple UI**: Launch a browser-based analysis console for visual quality control, plots, and parameter export.

## Extracting circadian parameters from live-cell data

For live-cell data (see tutorial in [`notebooks/live_cell_tutorial`](./notebooks/live_cell_tutorial.ipynb)), `cosinor-lite` offers the choice between three different cosinor models:

- a) Fixed 24-h period

- b) Free period

- c) Free period with damped amplitude

<div style="text-align: center;">
  <img src="./images/live_cell_fitting-01.png" alt="Model selection" style="width:100%;"/>
</div>

Once a model is chosen, it is fitted independently to each sample. The fitted parameters can be exported for downstream statistical analysis.

## Differential rhytmicity analysis of omics datasets

`cosinor-lite` includes a toolbox for performing differential rhytmicity analysis of omics datasets (ee [`notebooks/omics_tutorial`](./notebooks/omics_tutorial.ipynb)) for tutorial. The details of the method are nicely explained in the article:

> Pelikan A, Herzel H, Kramer A, Ananthasubramaniam B. 2022. Venn diagram analysis overestimates the extent of circadian rhythm reprogramming. The FEBS Journal 289:6605–6621. doi:10.1111/febs.16095

Here is an adaptation of their figure explaining the methdology:

<div style="text-align: center;">
  <img src="./images/model_selection.png" alt="Model selection" style="width:60%;"/>
</div>

For condition 1 (i.e. alpha cells) and condition 2 (i.e. beta cells), we fit 5 different models:

- Model 1) Arrhythmic in alpha and beta cells

- Model 2) Rhythmic in beta cells only

- Model 3) Rhythmic in alpha cells only

- Model 4) Rhythmic in alpha and beta cells with the same rhythmic parameters (i.e. phase and amplitude)

- Model 5) Rhythmic in both but with differential rhythmicity in alpha vs beta cells

## Installation

```bash
# clone the repository
git clone https://github.com/nick-e-p/cosinor-lite.git
cd cosinor-lite

# create and populate the local uv environment
uv sync

# activate the environment
source .venv/bin/activate
```

> **Tip:** The project targets Python 3.11+. Using `uv` keeps package resolution reproducible via `uv.lock`.

## Quick Start

### Launch the interactive app

```bash
uv run python app.py
```

Open the printed local URL to explore:

- **Live cell**: Upload CSVs with participant, replicate, and time-series data, then compare detrending strategies or cosinor fits across groups.
- **Omics**: Load expression matrices, build time vectors automatically or manually, compute differential rhythmicity, and download publication-ready plots.

### Use the Python API

```python
from cosinor_lite.livecell_dataset import LiveCellDataset
from cosinor_lite.livecell_cosinor_analysis import CosinorAnalysis

dataset = LiveCellDataset(
    ids=["mouse_a", "mouse_b"],
    group=["treated", "control"],
    replicate=[1, 1],
    time_series=my_expression_matrix,
    time=my_timepoints,
)

analysis = CosinorAnalysis(dataset=dataset)
fit = analysis.fit(group="treated", model="damped")
print(fit.parameters)
```

## Repository Layout

```
src/cosinor_lite/        # Core library modules and models
app.py                   # Gradio application entry point
tests/                   # Pytest suites (unit, integration, fixtures)
data/                    # Example datasets for the app
images/                  # Illustration assets used in the UI
```

## Development Workflow

```bash
# run style, type, and security hooks
pre-commit run --all-files

# execute unit tests
uv run pytest
```

## Sample Datasets

The `data/` directory includes curated examples to help you get started:

- `bioluminescence_example.csv`: Bioluminescence time series for live-cell analysis.
- `cytokine_example.csv`: An alternative CSV with the same schema for cosinor fitting using cytokine data.
- `GSE95156_Alpha_Beta.txt`: RNA-seq data used in the omics differential rhythmicity workflow.

Each file is formatted to drag & drop into `app.py` as well as the library APIs.

## Contributing

1. Fork the repository and create a feature branch.
2. Install dependencies with `uv sync` and activate the environment.
3. Implement your changes with tests.
4. Verify via `pre-commit run --all-files` and `uv run pytest`.
5. Open a pull request describing the motivation and results.

Bug reports and feature proposals are welcome through GitHub issues. Please include reproducible examples whenever possible.

## License

Licensed under the Apache License 2.0. See `LICENSE` for details.
