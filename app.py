from __future__ import annotations

import re
import tempfile
from collections.abc import Sequence
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cosinor_lite.livecell_cosinor_analysis import CosinorAnalysis
from cosinor_lite.livecell_dataset import LiveCellDataset
from cosinor_lite.omics_dataset import OmicsDataset
from cosinor_lite.omics_differential_rhytmicity import DifferentialRhythmicity, OmicsHeatmap

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 8,
        "pdf.fonttype": 42,
    },
)
plt.style.use("seaborn-v0_8-ticks")

# --- (your rcParams / styles as-is) ---

APP_DIR = Path(__file__).parent
bioluminescence_file = str(APP_DIR / "data" / "bioluminescence_example.csv")
cytokine_file = str(APP_DIR / "data" / "cytokine_example.csv")
qpcr_file = str(APP_DIR / "data" / "qpcr_example.csv")
omics_example = str(APP_DIR / "data" / "GSE95156_Alpha_Beta.txt")
method_img = str(APP_DIR / "images" / "live_cell_fitting-01.png")
model_selection_img = str(APP_DIR / "images" / "model_selection.png")

with gr.Blocks(title="Cosinor Analysis — Live Cell & Omics") as demo:
    gr.Markdown(
        """
    # Cosinor Analysis App

    `cosinor-lite`: a simple app for circadian cosinor analysis & biostatistics.

    Choose between the `Live cell` and `Omics` tabs for two different types of analysis:
    - `Live cell`: inferring rhythmic properties of live cell data using three different cosinor models
    - `Omics`: differential rhythmicity analysis of omics datasets between two conditions

    """,
    )

    with gr.Tabs() as tabs:
        with gr.Tab("Live cell", id=0):
            gr.Image(
                value=method_img,
                label="Choosing a cosinor model and fitting parameters",
                interactive=False,
                show_label=True,
                height=600,  # adjust as needed; or remove to use natural size
            )

            gr.Markdown(
                """
            # Fitting live cell data

            This section allows inference of parameters describing circadian oscillations in live cell data.
            Once inferred, the extracted parameters can be compared between groups in downstream analyses. Methods are included at the bottom of the page for easy copy-pasting.

            There are three types of cosinor model to choose from:
            - 24h period cosinor
            - Free period (constrained within 20-28h) cosinor
            - Damped cosinor (equivalent to Chronostar analysis), with an additional dampening coefficient

            ## Input data format

            There are many valid ways to organise the underlying live cell data file. Here we assume a specific format to facilitate data processing.
            If in doubt, you can download the example files and match the format of your input data accordingly.

            - Row 1: contains a unique identifier for the participant ID, mouse ID etc.
            - Row 2: replicate number. If there's only one replicate per unique ID, this can just be a row of 1's
            - Row 3: the group to which each measurement belongs (maximum of two groups allowed)
            - Left column; the left column contains the time (going down)

            ## Recommended workflow for examples

            - Bioluminescence: Detrending method: Moving average | Cosinor model: Damped cosinor (Chronostar)
            - Cytokine: Detrending method: Linear | Cosinor model: Free period cosinor
            - qPCR: Detrending method: None | Cosinor model: 24h period cosinor

            """,
            )

            file = gr.File(label="Upload CSV", file_types=[".csv"], type="filepath")

            gr.Examples(
                examples=[bioluminescence_file, cytokine_file, qpcr_file],
                inputs=file,
                label="Example input",
            )

            status = gr.Textbox(label="CSV status", interactive=False)

            gr.Markdown(
                "Provide the names of your two groups below, as they appear in row 3 of your CSV file.",
            )

            with gr.Row():
                group1_label = gr.Textbox(label="Group 1 label", value="Group 1")
                group2_label = gr.Textbox(label="Group 2 label", value="Group 2")

            st_participant_id = gr.State()
            st_replicate = gr.State()
            st_group = gr.State()
            st_time = gr.State()
            st_time_rows = gr.State()

            def process_csv(
                fpath: str | Path,
            ) -> tuple[
                str,
                pd.Series,
                pd.Series,
                pd.Series,
                np.ndarray,
                pd.DataFrame,
            ]:
                df_data = pd.read_csv(fpath, index_col=0, header=None)

                participant_id = df_data.iloc[0, :].astype(str)
                replicate = df_data.iloc[1, :].astype(int)
                group = df_data.iloc[2, :].astype(str)

                time_index = df_data.index[3:]
                try:
                    time = time_index.astype(float).to_numpy()
                except (TypeError, ValueError) as error:
                    msg = f"Time index not numeric from row 4 onward: {list(time_index)}"
                    raise ValueError(msg) from error

                time_rows = df_data.iloc[3:].apply(pd.to_numeric, errors="raise")

                shape_info = f"Total loaded shape of metadata + data: {df_data.shape} | {len(participant_id)} samples x {len(time)} time points"
                return shape_info, participant_id, replicate, group, time, time_rows

            for evt in (file.upload, file.change):
                evt(
                    process_csv,
                    inputs=file,
                    outputs=[
                        status,
                        st_participant_id,
                        st_replicate,
                        st_group,
                        st_time,
                        st_time_rows,
                    ],
                )

            group_choice = gr.Radio(
                choices=[("Group 1", "group1"), ("Group 2", "group2")],
                value="group1",
                label="Select group to analyse",
            )
            m_slider = gr.Slider(
                1,
                10,
                value=5,
                step=1,
                label="Number of columns for plot",
            )

            def make_plot(  # noqa: PLR0913
                ids_series: pd.Series | None,
                group_series: pd.Series | None,
                repl_series: pd.Series | None,
                time_array: np.ndarray | None,
                time_rows_df: pd.DataFrame | None,
                g1: str,
                g2: str,
                which_group: str,
                method: str,
                window: float,
                m: float,
                plot_style: str,
            ) -> tuple[plt.Figure | None, str | None, str | None]:
                if (
                    ids_series is None
                    or group_series is None
                    or repl_series is None
                    or time_array is None
                    or time_rows_df is None
                ):
                    return None, None, None
                ds = LiveCellDataset(
                    ids=list(ids_series),
                    group=list(group_series),
                    replicate=[int(x) for x in list(repl_series)],
                    time_series=time_rows_df.to_numpy(dtype=float),
                    time=np.asarray(time_array, dtype=float),
                    group1_label=g1,
                    group2_label=g2,
                )
                if method == "moving_average":
                    fig, pdf_path, csv_path = ds.plot_group_data(
                        which_group,
                        method=method,
                        m=int(m),
                        window=int(window),
                        plot_style=plot_style,
                    )
                else:
                    fig, pdf_path, csv_path = ds.plot_group_data(
                        which_group,
                        method=method,
                        m=int(m),
                        plot_style=plot_style,
                    )
                return fig, pdf_path, csv_path

            method_choice = gr.Radio(
                choices=[
                    ("None (no detrending)", "none"),
                    ("Linear", "linear"),
                    ("Linear + quadratic", "poly2"),
                    ("Moving average", "moving_average"),
                ],
                value="none",
                label="Detrending method",
            )

            window_slider = gr.Slider(
                1,
                1000,
                value=144,
                step=1,
                label="Window size for moving average (if used)",
            )
            gr.Markdown(
                "⚠️ Note: Window size is in **data points** (measurement intervals), not hours.",
            )

            plot_style_choice = gr.Radio(
                choices=[("Line", "line"), ("Scatter", "scatter")],
                value="line",
                label="Plot style for raw data",
            )

            build_btn = gr.Button("Plot raw data and trend", variant="primary")
            plot = gr.Plot(label="Preview")
            download = gr.File(label="Download plot")
            detrended_download = gr.File(label="Download detrended CSV")

            build_btn.click(
                make_plot,
                inputs=[
                    st_participant_id,
                    st_group,
                    st_replicate,
                    st_time,
                    st_time_rows,
                    group1_label,
                    group2_label,
                    group_choice,
                    method_choice,
                    window_slider,
                    m_slider,
                    plot_style_choice,
                ],
                outputs=[plot, download, detrended_download],
            )

            cosinor_model = gr.Radio(
                choices=[
                    ("24h period cosinor", "cosinor_24"),
                    ("Free period (20-28h) cosinor", "cosinor_free_period"),
                    ("Damped cosinor (Chronostar)", "cosinor_damped"),
                ],
                value="cosinor_24",  # must match one of the VALUES above
                label="Cosinor model",
            )

            t_lower_slider = gr.Slider(
                0,
                1000,
                value=0,
                step=1,
                label="Remove data below this time limit (hours)",
            )
            t_upper_slider = gr.Slider(
                0,
                1000,
                value=1000,
                step=1,
                label="Remove data above this time limit (hours)",
            )

            build_btn_cosinor = gr.Button("Build cosinor fits", variant="primary")
            plot_cosinor = gr.Plot(label="Model fit preview")
            pdf_export = gr.File(label="Download figure")
            table_export = gr.Dataframe()
            download_export = gr.File(label="Download fitted parameters")

            def make_cosinor_fits(  # noqa: PLR0913
                ids_series: pd.Series | None,
                group_series: pd.Series | None,
                repl_series: pd.Series | None,
                time_array: np.ndarray | None,
                time_rows_df: pd.DataFrame | None,
                g1: str,
                g2: str,
                which_group: str,
                method: str,
                window: float,
                cosinor_model: str,
                t_lower: float,
                t_upper: float,
                m: float,
                plot_style: str,
            ) -> tuple[
                plt.Figure | None,
                str | None,
                pd.DataFrame | None,
                str | None,
            ]:
                if (
                    ids_series is None
                    or group_series is None
                    or repl_series is None
                    or time_array is None
                    or time_rows_df is None
                ):
                    return None, None, None, None
                ds = CosinorAnalysis(
                    ids=list(ids_series),
                    group=list(group_series),
                    replicate=[int(x) for x in list(repl_series)],
                    time_series=time_rows_df.to_numpy(dtype=float),
                    time=np.asarray(time_array, dtype=float),
                    group1_label=g1,
                    group2_label=g2,
                    t_lower=t_lower,
                    t_upper=t_upper,
                )
                if method == "moving_average":
                    df_export, csv_path, fig, pdf_path = ds.fit_cosinor(
                        which_group,
                        method=method,
                        window=int(window),
                        cosinor_model=cosinor_model,
                        m=int(m),
                        plot_style=plot_style,
                    )
                else:
                    df_export, csv_path, fig, pdf_path = ds.fit_cosinor(
                        which_group,
                        method=method,
                        cosinor_model=cosinor_model,
                        m=int(m),
                        plot_style=plot_style,
                    )
                return fig, pdf_path, df_export, csv_path

            build_btn_cosinor.click(
                make_cosinor_fits,
                inputs=[
                    st_participant_id,
                    st_group,
                    st_replicate,
                    st_time,
                    st_time_rows,
                    group1_label,
                    group2_label,
                    group_choice,
                    method_choice,
                    window_slider,
                    cosinor_model,
                    t_lower_slider,
                    t_upper_slider,
                    m_slider,
                    plot_style_choice,  # <-- add
                ],
                outputs=[plot_cosinor, pdf_export, table_export, download_export],
            )
            gr.Markdown(
                """
            # Fitting Live Cell Data Methods (to paste and adapt as needed)

            Live cell time series from X data were first detrended using:

            - **Linear regression**: ordinary least-squares (statsmodels 0.14.1) was fit to the raw signal versus time, and the fitted mean-centered trend was subtracted to yield detrended residuals.
            - **Quadratic regression**: a second-order polynomial was fit to the raw signal versus time (NumPy 1.26), and the fitted mean-centered trend was subtracted to yield detrended residuals.
            - **Centered moving average**: a rolling mean (pandas 2.1.4) with a window of X hours was computed, and detrended values were obtained by subtracting the mean-centered trend from the raw data.


            We extracted periodic parameters from the cell line data by fitting a cosinor model of the form (CHOOSE APPROPRIATE MODEL):

            **24-h cosinor model**

            $$
            y(t) = \\text{mesor} + \\text{amplitude} \\cdot \\cos\\!\\left(\\frac{2\\pi (t - \\text{acrophase})}{24}\\right),
            $$

            where $y(t)$ represents the X signal and $t$ represents time in hours. The parameters mesor, amplitude, and acrophase correspond to the mean level, oscillation amplitude, and phase of peak expression, respectively. Model fitting was performed by minimizing the least-squares error using the `curve_fit` function from SciPy (v1.11.4).

            **Free-period cosinor model**

            $$
            y(t) = \\text{mesor} + \\text{amplitude} \\cdot \\cos\\!\\left(\\frac{2\\pi (t - \\text{acrophase})}{\\text{period}}\\right),
            $$

            where $y(t)$ represents the X signal and $t$ represents time in hours. The parameters mesor, amplitude, acrophase, and period correspond to the mean level, oscillation amplitude, phase of peak expression, and oscillation period, respectively. Model fitting was performed by minimizing the least-squares error using the `curve_fit` function from SciPy (v1.11.4).

            **Damped cosinor model**

            To account for potential attenuation of oscillatory amplitude over time, we also considered a damped cosinor model with exponential decay,

            $$
            y(t) = \\text{mesor} + \\text{amplitude} \\cdot e^{-\\lambda t}
            \\cos\\!\\left(\\frac{2\\pi (t - \\text{acrophase})}{\\text{period}}\\right),
            $$

            where $y(t)$ represents the X signal and $t$ represents time in hours. The parameters mesor, amplitude, acrophase, and period correspond to the mean level, oscillation amplitude, phase of peak expression, and oscillation period, respectively. The parameter $\\lambda$ is the damping coefficient controlling the rate of exponential decay of the oscillation amplitude. Model fitting was performed by minimizing the least-squares error using the `curve_fit` function from SciPy (v1.11.4).

            """,
                latex_delimiters=[
                    {"left": "$$", "right": "$$", "display": True},
                    {"left": "$", "right": "$", "display": False},
                ],
            )
        with gr.Tab("Omics", id=1):
            gr.Markdown(
                """
            # Differential rhythmicity analysis of omics datasets

            Here we perform differential rhythmicity analysis between two conditions using a model selection approach on omics data.
            The example data includes published RNA-seq data, but in theory any types of omics data (RNA-seq, proteomics, metabolomics, lipidomics) could be used. The dataset is from the following publication:

            > Petrenko V, Saini C, Giovannoni L, Gobet C, Sage D, Unser M, Heddad Masson M, Gu G, Bosco D, Gachon F, Philippe J, Dibner C. 2017. Pancreatic alpha- and beta-cellular clocks have distinct molecular properties and impact on islet hormone secretion and gene expression. Genes Dev 31:383-398. doi:10.1101/gad.290379.116.
                """,
            )

            gr.Image(
                value=model_selection_img,
                label="Differential rhytmicity analysis with model selection",
                interactive=False,
                show_label=True,
                height=600,
            )

            gr.Markdown(
                """
            ## How does the method actually work?

            Methods are included at the bottom of the page for easy copy-pasting. The details of the method are nicely explained in the article:
            > Pelikan A, Herzel H, Kramer A, Ananthasubramaniam B. 2022. Venn diagram analysis overestimates the extent of circadian rhythm reprogramming. The FEBS Journal 289:6605–6621. doi:10.1111/febs.16095

            See the above adaptation of their figure explaining the methodology.

            For condition 1 (i.e. cell type 1) and condition 2 (i.e. cell type 2), we fit five different models:

            - Model 1) Arrhythmic in cell type 1 and cell type 2
            - Model 2) Rhythmic in cell type 2 only
            - Model 3) Rhythmic in cell type 1 only
            - Model 4) Rhythmic in cell type 1 and cell type 2 with the same rhythmic parameters (i.e. phase and amplitude)
            - Model 5) Rhythmic in both but with differential rhythmicity in cell type 1 vs cell type 2

            A degree of confidence is calculated for each model (called model weight, which sums to 1 across all models), and a model is chosen if the model weight exceeds a threshold (for this tutorial we will use 0.5). If no model exceeds this threshold, then the model is unclassified, which we define as Model 0.

            - Model 0) unclassified

                """,
            )

            omics_file = gr.File(
                label="Upload Omics TXT/TSV",
                file_types=[".txt", ".tsv", ".csv"],
                type="filepath",
            )
            gr.Examples(
                examples=[omics_example],
                inputs=omics_file,
                label="Example input",
            )

            omics_status = gr.Textbox(label="File status", interactive=False)

            columns_cond1_dd = gr.Dropdown(
                choices=[],
                multiselect=True,
                label="Condition A columns (e.g., ZT_*_a_*)",
            )
            columns_cond2_dd = gr.Dropdown(
                choices=[],
                multiselect=True,
                label="Condition B columns (e.g., ZT_*_b_*)",
            )

            override_time = gr.Checkbox(
                label="Override time vectors manually?",
                value=False,
            )
            t_cond1_tb = gr.Textbox(label="t_cond1 (comma-separated)", visible=False)
            t_cond2_tb = gr.Textbox(label="t_cond2 (comma-separated)", visible=False)
            log2_choice = gr.Radio(
                choices=[("No", "no"), ("Yes", "yes")],
                value="no",
                label="Apply log2 transform to expression data?",
            )

            st_df_rna = gr.State()

            omics_preview = gr.Code(label="Planned class inputs", language="python")
            build_omics_btn = gr.Button("Build Omics inputs", variant="primary")

            sample1_dd = gr.Dropdown(choices=[], label="Sample 1 (x-axis)")
            sample2_dd = gr.Dropdown(choices=[], label="Sample 2 (y-axis)")
            scatter_btn = gr.Button(
                "Generate replicate scatterplot",
                variant="secondary",
            )
            scatter_plot = gr.Plot(label="Replicate scatterplot")
            scatter_download = gr.File(label="Download scatterplot")

            hist_btn = gr.Button("Generate histogram", variant="primary")
            omics_plot = gr.Plot(label="Expression histogram")
            omics_download = gr.File(label="Download histogram")

            def _guess_cols(cols: Sequence[str]) -> tuple[list[str], list[str]]:
                cols = [str(c) for c in cols]
                a_guess = [c for c in cols if re.search(r"_a_", str(c))]
                b_guess = [c for c in cols if re.search(r"_b_", str(c))]
                return a_guess, b_guess

            def _pick_default_samples(
                cols: Sequence[str],
            ) -> tuple[str | None, str | None]:
                s1 = next((c for c in cols if re.search(r"ZT_0_.*_1$", str(c))), None)
                s2 = next((c for c in cols if re.search(r"ZT_0_.*_2$", str(c))), None)
                if s1 and s2 and s1 != s2:
                    return s1, s2
                cols = [str(c) for c in cols]
                return (cols[0] if cols else None, cols[1] if len(cols) > 1 else None)

            def _build_time_vec(n_cols: int, manual_text: str | None) -> list[float]:
                if manual_text:
                    try:
                        return [float(x.strip()) for x in manual_text.split(",") if x.strip()]
                    except (AttributeError, ValueError):
                        pass
                base = [0, 4, 8, 12, 16, 20]
                reps = max(1, (n_cols + len(base) - 1) // len(base))
                return [float(value) for value in (base * reps)[:n_cols]]

            def load_omics(
                fpath: str | Path,
            ) -> tuple[object, object, object, pd.DataFrame, object, object, object, object]:
                dataframe = pd.read_csv(fpath, sep="\t")

                if dataframe.shape[1] > 0:
                    dataframe = dataframe.drop(dataframe.columns[0], axis=1)

                if "gene_name" in dataframe.columns:
                    dataframe["Genes"] = dataframe["gene_name"].astype(str).str.split("|").str[1]

                status = f"Loaded {dataframe.shape[0]} rows x {dataframe.shape[1]} columns."
                choices = dataframe.columns.tolist()
                a_guess, b_guess = _guess_cols(choices)
                s1, s2 = _pick_default_samples(choices)
                default_cycle = "0,4,8,12,16,20"

                return (
                    status,
                    gr.update(choices=choices, value=a_guess),
                    gr.update(choices=choices, value=b_guess),
                    dataframe,
                    gr.update(value=default_cycle),
                    gr.update(value=default_cycle),
                    gr.update(choices=choices, value=s1),
                    gr.update(choices=choices, value=s2),
                )

            for evt in (omics_file.upload, omics_file.change):
                evt(
                    load_omics,
                    inputs=omics_file,
                    outputs=[
                        omics_status,
                        columns_cond1_dd,
                        columns_cond2_dd,
                        st_df_rna,
                        t_cond1_tb,
                        t_cond2_tb,
                        sample1_dd,
                        sample2_dd,
                    ],
                )

            def toggle_time_fields(checked: object) -> tuple[object, object]:
                is_checked = bool(checked)
                return gr.update(visible=is_checked), gr.update(visible=is_checked)

            override_time.change(
                toggle_time_fields,
                inputs=override_time,
                outputs=[t_cond1_tb, t_cond2_tb],
            )

            def build_omics_inputs(  # noqa: PLR0913
                df: pd.DataFrame | None,
                cols_a: Sequence[str] | None,
                cols_b: Sequence[str] | None,
                use_manual_time: object,
                t_a_text: str | None,
                t_b_text: str | None,
            ) -> str:
                if df is None:
                    return "# Upload or select an example file first."

                cols_a = list(cols_a or [])
                cols_b = list(cols_b or [])

                manual_flag = bool(use_manual_time)

                t_a = _build_time_vec(
                    len(cols_a),
                    t_a_text if manual_flag else None,
                )
                t_b = _build_time_vec(
                    len(cols_b),
                    t_b_text if manual_flag else None,
                )

                snippet = f"""# Planned inputs for the Omics dataset class
        columns_cond1 = {cols_a}
        columns_cond2 = {cols_b}
        t_cond1 = {t_a}
        t_cond2 = {t_b}
        """
                return snippet

            build_omics_btn.click(
                build_omics_inputs,
                inputs=[
                    st_df_rna,
                    columns_cond1_dd,
                    columns_cond2_dd,
                    override_time,
                    t_cond1_tb,
                    t_cond2_tb,
                ],
                outputs=omics_preview,
            )

            def run_histogram(  # noqa: PLR0913
                df: pd.DataFrame | None,
                cols_a: Sequence[str] | None,
                cols_b: Sequence[str] | None,
                use_manual_time: object,
                t_a_text: str | None,
                t_b_text: str | None,
                log2_option: str,
            ) -> tuple[plt.Figure | None, str | None]:
                if df is None:
                    return None, None

                cols_a = list(cols_a or [])
                cols_b = list(cols_b or [])
                manual_flag = bool(use_manual_time)

                t_a = _build_time_vec(len(cols_a), t_a_text if manual_flag else None)
                t_b = _build_time_vec(len(cols_b), t_b_text if manual_flag else None)

                t_a_array = np.asarray(t_a, dtype=float)
                t_b_array = np.asarray(t_b, dtype=float)
                apply_log2 = log2_option == "yes"

                rna_data = OmicsDataset(
                    df=df,
                    columns_cond1=cols_a,
                    columns_cond2=cols_b,
                    t_cond1=t_a_array,
                    t_cond2=t_b_array,
                    deduplicate_on_init=True,
                    log2_transform=apply_log2,
                )

                fig = rna_data.expression_histogram()

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
                    fig.savefig(tmpfile.name)
                    tmp_path = tmpfile.name
                plt.close(fig)
                return fig, tmp_path

            hist_btn.click(
                run_histogram,
                inputs=[
                    st_df_rna,
                    columns_cond1_dd,
                    columns_cond2_dd,
                    override_time,
                    t_cond1_tb,
                    t_cond2_tb,
                    log2_choice,
                ],
                outputs=[omics_plot, omics_download],
            )

            def run_replicate_scatter(  # noqa: PLR0913
                df: pd.DataFrame | None,
                cols_a: Sequence[str] | None,
                cols_b: Sequence[str] | None,
                use_manual_time: object,
                t_a_text: str | None,
                t_b_text: str | None,
                sample1: str | None,
                sample2: str | None,
                log2_option: str,
            ) -> tuple[plt.Figure | None, str | None]:
                if df is None or not sample1 or not sample2:
                    return None, None

                cols_a = list(cols_a or [])
                cols_b = list(cols_b or [])
                manual_flag = bool(use_manual_time)

                t_a = _build_time_vec(len(cols_a), t_a_text if manual_flag else None)
                t_b = _build_time_vec(len(cols_b), t_b_text if manual_flag else None)

                t_a_array = np.asarray(t_a, dtype=float)
                t_b_array = np.asarray(t_b, dtype=float)
                apply_log2 = log2_option == "yes"

                rna_data = OmicsDataset(
                    df=df,
                    columns_cond1=cols_a,
                    columns_cond2=cols_b,
                    t_cond1=t_a_array,
                    t_cond2=t_b_array,
                    deduplicate_on_init=True,
                    log2_transform=apply_log2,
                )

                fig = rna_data.replicate_scatterplot(sample1=sample1, sample2=sample2)

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    fig.savefig(tmp.name)
                    tmp_path = tmp.name
                plt.close(fig)
                return fig, tmp_path

            scatter_btn.click(
                run_replicate_scatter,
                inputs=[
                    st_df_rna,
                    columns_cond1_dd,
                    columns_cond2_dd,
                    override_time,
                    t_cond1_tb,
                    t_cond2_tb,
                    sample1_dd,
                    sample2_dd,
                    log2_choice,
                ],
                outputs=[scatter_plot, scatter_download],
            )
            gr.Markdown(
                "Define a cutoff for expressed genes, based on mean expression or number of detected samples.",
            )
            with gr.Row():
                cond1_label_tb = gr.Textbox(
                    label="Condition 1 label",
                    value="Alpha cells",
                )
                cond2_label_tb = gr.Textbox(
                    label="Condition 2 label",
                    value="Beta cells",
                )
                mean_min_num = gr.Number(
                    label="mean_min (for is_expressed)",
                    value=0,
                    precision=0,
                )
                num_detected_num = gr.Number(
                    label="num_detected_min (for is_expressed)",
                    value=0,
                    precision=0,
                )

            compute_dr_btn = gr.Button(
                "Compute differential rhythmicity & heatmap",
                variant="primary",
            )

            heatmap_plot = gr.Plot(label="Heatmap preview")
            heatmap_download = gr.File(label="Download heatmap (PDF)")
            params_preview = gr.Dataframe(
                label="Rhythmic parameters (preview)",
                interactive=False,
            )
            params_download = gr.File(label="Download rhythmic parameters (CSV)")

            def run_dr_and_heatmap(  # noqa: PLR0913
                df: pd.DataFrame | None,
                cols_a: Sequence[str] | None,
                cols_b: Sequence[str] | None,
                use_manual_time: object,
                t_a_text: str | None,
                t_b_text: str | None,
                cond1_label: str,
                cond2_label: str,
                mean_min: float | None,
                num_detected_min: float | None,
                log2_option: str,
            ) -> tuple[plt.Figure | None, str | None, pd.DataFrame | None, str | None]:
                if df is None or not cols_a or not cols_b:
                    return None, None, None, None

                cols_a = list(cols_a or [])
                cols_b = list(cols_b or [])
                manual_flag = bool(use_manual_time)

                t_a = _build_time_vec(len(cols_a), t_a_text if manual_flag else None)
                t_b = _build_time_vec(len(cols_b), t_b_text if manual_flag else None)

                t_a_array = np.asarray(t_a, dtype=float)
                t_b_array = np.asarray(t_b, dtype=float)
                apply_log2 = log2_option == "yes"

                rna_data = OmicsDataset(
                    df=df,
                    columns_cond1=cols_a,
                    columns_cond2=cols_b,
                    t_cond1=t_a_array,
                    t_cond2=t_b_array,
                    deduplicate_on_init=True,
                    log2_transform=apply_log2,
                )

                try:
                    mean_min_value = float(mean_min) if mean_min is not None else 0.0
                except (TypeError, ValueError):
                    mean_min_value = 0.0
                try:
                    num_detected_value = int(num_detected_min) if num_detected_min is not None else None
                except (TypeError, ValueError):
                    num_detected_value = None
                rna_data.add_is_expressed(mean_min=mean_min_value, num_detected_min=num_detected_value)

                dr = DifferentialRhythmicity(dataset=rna_data)
                rhythmic_all = dr.extract_all_circadian_params()  # pandas DataFrame

                heatmap = OmicsHeatmap(
                    df=rhythmic_all,
                    columns_cond1=cols_a,
                    columns_cond2=cols_b,
                    t_cond1=t_a_array,
                    t_cond2=t_b_array,
                    cond1_label=cond1_label or "Condition 1",
                    cond2_label=cond2_label or "Condition 2",
                    show_unexpressed=False,
                )
                fig = heatmap.plot_heatmap()  # should return a Matplotlib Figure

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                    fig.savefig(tmp_pdf.name)
                    tmp_pdf_path = tmp_pdf.name

                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_csv:
                    rhythmic_all.to_csv(tmp_csv.name, index=False)
                    tmp_csv_path = tmp_csv.name

                preview_df = rhythmic_all.head(20)

                return fig, tmp_pdf_path, preview_df, tmp_csv_path

            compute_dr_btn.click(
                run_dr_and_heatmap,
                inputs=[
                    st_df_rna,
                    columns_cond1_dd,
                    columns_cond2_dd,
                    override_time,
                    t_cond1_tb,
                    t_cond2_tb,
                    cond1_label_tb,
                    cond2_label_tb,
                    mean_min_num,
                    num_detected_num,
                    log2_choice,
                ],
                outputs=[
                    heatmap_plot,
                    heatmap_download,
                    params_preview,
                    params_download,
                ],
            )
            gr.Markdown(
                """
            ## Differential Rhythmicity Analysis Methods (to paste and adapt as needed)

            Differential rhythmicity analysis was performed using harmonic linear regression and model selection based on the Bayesian Information Criterion (BIC) (Atger et al., 2015, Pelikan et al., 2022).

            We used the harmonic linear regression model

            $$
            y(t) = m + a \\cos(\\omega t) + b \\sin(\\omega t),
            $$

            where $y(t)$ represents the log2-transformed values of either transcript or protein levels, $m$ is the mean level, $\\omega$ is the frequency of the oscillations (fixed to give a period of 24 h), $t$ is the time, and $a$ and $b$ are the regression coefficients of the cosine and sine terms, respectively.

            We proposed five different models to represent different rhythmic categories. Model $M_1$ corresponds to arrhythmic behavior in both Condition 1 and Condition 2 cells. Model $M_2$ corresponds to rhythmicity in Condition 2 cells only, whereas model $M_3$ corresponds to rhythmicity in Condition 1 cells only. Model $M_4$ corresponds to rhythmicity in both Condition 1 and Condition 2 cells with identical rhythmic parameters (i.e. identical coefficients $a$ and $b$). Model $M_5$ corresponds to rhythmicity in both cell types with differential rhythmicity between Condition 1 and Condition 2 cells (i.e. different coefficients $a$ and $b$).

            For each model $M_i$, the regression was performed and the BIC was calculated as

            $$
            \\mathrm{BIC}_i = k_i \\ln(n) - 2 \\ln(\\hat{L}_i),
            $$

            where $k_i$ is the number of parameters in model $M_i$, $n$ is the sample size, and $\\hat{L}_i$ is the maximized value of the likelihood function for model $M_i$.

            For model comparison, the difference between the BIC of each model and the lowest BIC value among all models (denoted $\\mathrm{BIC}_{\\min}$) was computed as

            $$
            \\Delta_i = \\mathrm{BIC}_i - \\mathrm{BIC}_{\\min}.
            $$

            The Schwarz weight $w_i$ of each model was then calculated as

            $$
            w_i = \\frac{\\exp\\left(-\\tfrac{1}{2}\\Delta_i\\right)}{\\sum_i \\exp\\left(-\\tfrac{1}{2}\\Delta_i\\right)}.
            $$

            Genes were considered as detected if the mean log2 RPKM level was ≥ X, and proteins were considered as detected if at least Y out of Z samples had a detectable level. Detected status was denoted using several subclasses: subclass $a$, detected in Condition 1 cells only; subclass $b$, detected in Condition 2 cells only; and subclass $c$, detected in both Condition 1 and Condition 2 cells.

            For each model there is an associated set of possible subclasses: model $M_1$ allows subclasses $a$, $b$, and $c$; model $M_2$ allows subclasses $b$ and $c$; model $M_3$ allows subclasses $a$ and $c$; model $M_4$ allows subclass $c$ only; model $M_5$ allows subclass $c$ only; and model $M_0$ allows subclass $c$ only.

            To generate lists of oscillatory genes or proteins, we summed the model weights corresponding to rhythmic expression for the condition of interest. If this summed weight exceeded the Schwarz weight cutoff of 0.5, the feature was classified as oscillatory. For example, for Condition 1 cells, oscillatory features were those supported by models $M_3$, $M_4$, and $M_5$.

            **References**

            > Atger F, Gobet C, Marquis J, Martin E, Wang J, Weger B, Lefebvre G, Descombes P, Naef F, Gachon F. 2015. Circadian and feeding rhythms differentially affect rhythmic mRNA transcription and translation in mouse liver. Proceedings of the National Academy of Sciences 112:E6579–E6588. doi:10.1073/pnas.1515308112

            > Pelikan A, Herzel H, Kramer A, Ananthasubramaniam B. 2022. Venn diagram analysis overestimates the extent of circadian rhythm reprogramming. The FEBS Journal 289:6605–6621. doi:10.1111/febs.16095
            """,
                latex_delimiters=[
                    {"left": "$$", "right": "$$", "display": True},
                    {"left": "$", "right": "$", "display": False},
                ],
            )


if __name__ == "__main__":
    demo.launch()
