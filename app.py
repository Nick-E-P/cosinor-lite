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
omics_example = str(APP_DIR / "data" / "GSE95156_Alpha_Beta.txt")
method_img = str(APP_DIR / "images" / "live_cell_fitting-01.png")
model_selection_img = str(APP_DIR / "images" / "model_selection.png")

with gr.Blocks(title="Cosinor Analysis — Live Cell & Omics") as demo:
    gr.Markdown(
        """
    # Cosinor Analysis App

    A simple app for circadian cosinor analysis & biostatistics.

    Choose between:
    - inferring rhythmic properties of live cell data using three different cosinor models
    - differential rhythmicity analysis of omics datasets

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

            This section allows the use to infer parameters describing circadian oscillations in live cell data.
            Once inferred, the extracted parameters can be compared between groups in downstream analyses. There are three types of cosinor model to choose from:
            - 24h period cosinor
            - Free period (constrained within 20-28h) cosinor
            - Damped cosinor (equivalent to Chronostar analysis), with an additional dampening coefficient

            There are many valid ways to organise the underlying live cell data file. Here we assume a specific format to facilitate data processing

            - Row 1: contains a unique identifier for the participant, mouse etc.
            - Row 2: replicate number. If there's only one replicate per unique ID, this can just be a row of 1's
            - Row 3: the group to which each measurement belongs
            - Left column; the left column contains the time (going down)

            """,
            )

            file = gr.File(label="Upload CSV", file_types=[".csv"], type="filepath")

            gr.Examples(
                examples=[bioluminescence_file, cytokine_file],
                inputs=file,
                label="Example input",
            )

            status = gr.Textbox(label="CSV status", interactive=False)

            with gr.Row():
                group1_label = gr.Textbox(label="Group 1 label", value="Group 1")
                group2_label = gr.Textbox(label="Group 2 label", value="Group 2")

            # State (values will be pandas / numpy objects, not components)
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
                # If needed, normalize FileData dict -> str path here
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

                shape_info = (
                    f"Loaded shape: {df_data.shape} | participants: {len(participant_id)} | time points: {len(time)}"
                )
                return shape_info, participant_id, replicate, group, time, time_rows

            # Trigger on both upload and change (Examples sets the value via change)
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
            ) -> tuple[plt.Figure | None, str | None]:
                if (
                    ids_series is None
                    or group_series is None
                    or repl_series is None
                    or time_array is None
                    or time_rows_df is None
                ):
                    return None, None
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
                    fig, pdf_path = ds.plot_group_data(
                        which_group,
                        method=method,
                        m=int(m),
                        window=int(window),
                        plot_style=plot_style,
                    )
                else:
                    fig, pdf_path = ds.plot_group_data(
                        which_group,
                        method=method,
                        m=int(m),
                        plot_style=plot_style,
                    )
                return fig, pdf_path

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
                outputs=[plot, download],
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
        with gr.Tab("Omics", id=1):
            gr.Markdown(
                """
            # Differential rhytmicity analysis of omics datasets

            Here we perform differential rhythmicity analysis on omics data using a model selection approach.
            The example data includes published RNA-seq data, but in theory any types of omics data (RNA-seq, proteomics, metabolomics, lipidomics) could be used. The dataset is from the following publication:

            Petrenko V, Saini C, Giovannoni L, Gobet C, Sage D, Unser M, Heddad Masson M, Gu G, Bosco D, Gachon F, Philippe J, Dibner C. 2017. Pancreatic alpha- and beta-cellular clocks have distinct molecular properties and impact on islet hormone secretion and gene expression. Genes Dev 31:383-398. doi:10.1101/gad.290379.116.
                """,
            )

            gr.Image(
                value=model_selection_img,
                label="Differential rhytmicity analysis with model selection",
                interactive=False,
                show_label=True,
                height=600,  # adjust as needed; or remove to use natural size
            )

            gr.Markdown(
                """
            ## How does the method actually work?

            The details of the method are nicely explained in the article:

            Pelikan A, Herzel H, Kramer A, Ananthasubramaniam B. 2022. Venn diagram analysis overestimates the extent of circadian rhythm reprogramming. The FEBS Journal 289:6605-6621. doi:10.1111/febs.16095

            See the above adaptation of their figure explaining the methodology.

            For condition 1 (i.e. alpha cells) and condition 2 (i.e. beta cells), we fit five different models:

            - Model 1) Arrhythmic in alpha and beta cells
            - Model 2) Rhythmic in beta cells only
            - Model 3) Rhythmic in alpha cells only
            - Model 4) Rhythmic in alpha and beta cells with the same rhythmic parameters (i.e. phase and amplitude)
            - Model 5) Rhythmic in both but with differential rhythmicity in alpha vs beta cells

            A degree of confidence is calculated for each model (called model weight, which sums to 1 across all models), and a model is chosen if the model weight exceeds a threshold (for this tutorial we will use 0.5). If no model exceeds this threshold, then the model is unclassified, which we define as Model 0.

            - Model 0) unclassified

                """,
            )

            # File input + example
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

            # Multiselects for choosing columns by header names
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

            # Optional manual time vectors
            override_time = gr.Checkbox(
                label="Override time vectors manually?",
                value=False,
            )
            t_cond1_tb = gr.Textbox(label="t_cond1 (comma-separated)", visible=False)
            t_cond2_tb = gr.Textbox(label="t_cond2 (comma-separated)", visible=False)

            # Hidden state to stash the DataFrame
            st_df_rna = gr.State()

            # Preview planned inputs for your class
            omics_preview = gr.Code(label="Planned class inputs", language="python")
            build_omics_btn = gr.Button("Build Omics inputs", variant="primary")

            # Sample dropdowns for replicate scatterplot
            sample1_dd = gr.Dropdown(choices=[], label="Sample 1 (x-axis)")
            sample2_dd = gr.Dropdown(choices=[], label="Sample 2 (y-axis)")
            scatter_btn = gr.Button(
                "Generate replicate scatterplot",
                variant="secondary",
            )
            scatter_plot = gr.Plot(label="Replicate scatterplot")
            scatter_download = gr.File(label="Download scatterplot")

            # Histogram outputs
            hist_btn = gr.Button("Generate histogram", variant="primary")
            omics_plot = gr.Plot(label="Expression histogram")
            omics_download = gr.File(label="Download histogram")

            # ---------- helpers ----------
            def _guess_cols(cols: Sequence[str]) -> tuple[list[str], list[str]]:
                cols = [str(c) for c in cols]
                a_guess = [c for c in cols if re.search(r"_a_", str(c))]
                b_guess = [c for c in cols if re.search(r"_b_", str(c))]

                def zt_key(col: str) -> int:
                    m = re.search(r"ZT_(\d+)", str(col))
                    return int(m.group(1)) if m else 0

                a_guess = sorted(a_guess, key=zt_key)
                b_guess = sorted(b_guess, key=zt_key)
                return a_guess, b_guess

            def _pick_default_samples(
                cols: Sequence[str],
            ) -> tuple[str | None, str | None]:
                # Try ZT_0 ... _1 and ZT_0 ... _2 as a sensible default pair
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

            # ---------- loaders & toggles ----------
            def load_omics(
                fpath: str | Path,
            ) -> tuple[object, object, object, pd.DataFrame, object, object, object, object]:
                # Read TSV (tab-separated). Pandas also handles CSV if present.
                dataframe = pd.read_csv(fpath, sep="\t")
                # Drop first column by index (your pipeline)
                if dataframe.shape[1] > 0:
                    dataframe = dataframe.drop(dataframe.columns[0], axis=1)
                # Create Genes column from gene_name if present
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

            # ---------- preview inputs for your future class ----------
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

                snippet = f"""# Planned inputs for your Omics class
        columns_cond1 = {cols_a}
        columns_cond2 = {cols_b}
        t_cond1 = {t_a}
        t_cond2 = {t_b}

        # Example construction (later):
        # rna_data = OmicsDataset(
        #     df=df_rna,
        #     columns_cond1=columns_cond1,
        #     columns_cond2=columns_cond2,
        #     t_cond1=t_cond1,
        #     t_cond2=t_cond2,
        #     deduplicate_on_init=True,
        # )
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

            # ---------- histogram ----------
            def run_histogram(  # noqa: PLR0913
                df: pd.DataFrame | None,
                cols_a: Sequence[str] | None,
                cols_b: Sequence[str] | None,
                use_manual_time: object,
                t_a_text: str | None,
                t_b_text: str | None,
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

                rna_data = OmicsDataset(
                    df=df,
                    columns_cond1=cols_a,
                    columns_cond2=cols_b,
                    t_cond1=t_a_array,
                    t_cond2=t_b_array,
                    deduplicate_on_init=True,
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
                ],
                outputs=[omics_plot, omics_download],
            )

            # ---------- replicate scatterplot ----------
            def run_replicate_scatter(  # noqa: PLR0913
                df: pd.DataFrame | None,
                cols_a: Sequence[str] | None,
                cols_b: Sequence[str] | None,
                use_manual_time: object,
                t_a_text: str | None,
                t_b_text: str | None,
                sample1: str | None,
                sample2: str | None,
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

                rna_data = OmicsDataset(
                    df=df,
                    columns_cond1=cols_a,
                    columns_cond2=cols_b,
                    t_cond1=t_a_array,
                    t_cond2=t_b_array,
                    deduplicate_on_init=True,
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
                ],
                outputs=[scatter_plot, scatter_download],
            )

            # ------------------------------------------------------------
            # Differential rhythmicity + heatmap
            # (Paste this inside your existing `with gr.Tab("Omics", id=1):` block)
            # Re-uses helpers: _build_time_vec, and states/components you already created.
            # ------------------------------------------------------------

            # Labels for the heatmap and expressed-threshold
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

            compute_dr_btn = gr.Button(
                "Compute differential rhythmicity & heatmap",
                variant="primary",
            )

            # Outputs
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
            ) -> tuple[plt.Figure | None, str | None, pd.DataFrame | None, str | None]:
                if df is None or not cols_a or not cols_b:
                    # Nothing to do yet
                    return None, None, None, None

                cols_a = list(cols_a or [])
                cols_b = list(cols_b or [])
                manual_flag = bool(use_manual_time)

                # Build time vectors (reuse your helper)
                t_a = _build_time_vec(len(cols_a), t_a_text if manual_flag else None)
                t_b = _build_time_vec(len(cols_b), t_b_text if manual_flag else None)

                t_a_array = np.asarray(t_a, dtype=float)
                t_b_array = np.asarray(t_b, dtype=float)

                # Construct dataset
                rna_data = OmicsDataset(
                    df=df,
                    columns_cond1=cols_a,
                    columns_cond2=cols_b,
                    t_cond1=t_a_array,
                    t_cond2=t_b_array,
                    deduplicate_on_init=True,
                )

                # Mark expressed genes
                try:
                    mean_min_value = float(mean_min) if mean_min is not None else 0.0
                except (TypeError, ValueError):
                    mean_min_value = 0.0
                rna_data.add_is_expressed(mean_min=mean_min_value)

                # Differential rhythmicity
                dr = DifferentialRhythmicity(dataset=rna_data)
                rhythmic_all = dr.extract_all_circadian_params()  # pandas DataFrame

                # Build heatmap
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

                # Save outputs for download

                # Heatmap PDF
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                    fig.savefig(tmp_pdf.name)
                    tmp_pdf_path = tmp_pdf.name

                # Rhythmic params CSV
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_csv:
                    rhythmic_all.to_csv(tmp_csv.name, index=False)
                    tmp_csv_path = tmp_csv.name

                # For preview table, show up to 200 rows to keep UI snappy
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
                ],
                outputs=[
                    heatmap_plot,
                    heatmap_download,
                    params_preview,
                    params_download,
                ],
            )


if __name__ == "__main__":
    demo.launch()
