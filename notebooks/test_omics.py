# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cosinor_lite.omics_dataset import OmicsDataset
from cosinor_lite.omics_differential_rhytmicity import DifferentialRhythmicity, OmicsHeatmap, TimeSeriesExample

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

# %%

file: Path = Path.cwd().parent / "data" / "GSE95156_Alpha_Beta.txt"

df_rna: pd.DataFrame = pd.read_csv(file, sep="\t")
df_rna: pd.DataFrame = df_rna.drop(df_rna.columns[0], axis=1)

# %%

df_rna["Genes"] = df_rna["gene_name"].str.split("|").str[1]

columns_cond1: list[str] = [
    "ZT_0_a_1",
    "ZT_4_a_1",
    "ZT_8_a_1",
    "ZT_12_a_1",
    "ZT_16_a_1",
    "ZT_20_a_1",
    "ZT_0_a_2",
    "ZT_4_a_2",
    "ZT_8_a_2",
    "ZT_12_a_2",
    "ZT_16_a_2",
    "ZT_20_a_2",
]

columns_cond2: list[str] = [
    "ZT_0_b_1",
    "ZT_4_b_1",
    "ZT_8_b_1",
    "ZT_12_b_1",
    "ZT_16_b_1",
    "ZT_20_b_1",
    "ZT_0_b_2",
    "ZT_4_b_2",
    "ZT_8_b_2",
    "ZT_12_b_2",
    "ZT_16_b_2",
    "ZT_20_b_2",
]

t_cond1: np.ndarray = np.array(
    [
        0,
        4,
        8,
        12,
        16,
        20,
        0,
        4,
        8,
        12,
        16,
        20,
    ],
)

t_cond2: np.ndarray = t_cond1.copy()

# %%

rna_data: OmicsDataset = OmicsDataset(
    df=df_rna,
    columns_cond1=columns_cond1,
    columns_cond2=columns_cond2,
    t_cond1=t_cond1,
    t_cond2=t_cond2,
    deduplicate_on_init=True,
    log2_transform=False,
)
fig = rna_data.expression_histogram()

# %%

fig = rna_data.replicate_scatterplot(sample1="ZT_0_a_1", sample2="ZT_0_a_2")

# %%

rna_data.add_is_expressed(mean_min=0, num_detected_min=2)

# %%

dr = DifferentialRhythmicity(dataset=rna_data)
rhythmic_all = dr.extract_all_circadian_params()

# %%

heatmap: OmicsHeatmap = OmicsHeatmap(
    df=rhythmic_all,
    columns_cond1=columns_cond1,
    columns_cond2=columns_cond2,
    t_cond1=t_cond1,
    t_cond2=t_cond2,
    cond1_label="Alpha cells",
    cond2_label="Beta cells",
    show_unexpressed=False,
)
fig = heatmap.plot_heatmap()

# %%

example = TimeSeriesExample(
    df=rhythmic_all,
    columns_cond1=columns_cond1,
    columns_cond2=columns_cond2,
    t_cond1=t_cond1,
    t_cond2=t_cond2,
    cond1_label="Alpha cells",
    cond2_label="Beta cells",
)

# example.plot_time_series("Scg2", xticks=np.arange(0, 25, 4))
# example.plot_time_series("Upk3a", xticks=np.arange(0, 25, 4))
# example.plot_time_series("Kcnk3", xticks=np.arange(0, 25, 4))

example.plot_time_series("Ntrk1", xticks=np.arange(0, 25, 4))

# %%
# Create a Genes column


# file: Path = Path.cwd().parent / "data" / "GSE95156_Alpha_Beta.txt"

# df_rna: pd.DataFrame = pd.read_csv(file, sep="\t")
# df_rna: pd.DataFrame = df_rna.drop(df_rna.columns[0], axis=1)


# df_rna["Genes"] = df_rna["gene_name"].str.split("|").str[1]
# df_rna = df_rna.drop(columns="gene_name")
# df_rna = df_rna[["Genes"] + [col for col in df_rna.columns if col != "Genes"]]

# df_rna.to_csv(Path.cwd().parent / "data" / "GSE95156_Alpha_Beta.csv", index=False)


# %%
