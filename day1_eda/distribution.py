import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from battery_common.config import BATCH_FILES
from battery_common.artifacts import display


def analyze_distribution(data_dir, writer):

    life = {}
    counts = {}
    for batch_id, filename in BATCH_FILES.items():
        with h5py.File(data_dir / filename, "r") as f:
            refs = f["batch"]["cycle_life"][()].ravel()
            life[batch_id] = pd.Series([f[ref][()].item() for ref in refs])
            counts[batch_id] = len(refs)

    life = pd.DataFrame(life)
    stats = life.describe().T
    stats["missing"] = pd.Series(counts) - stats["count"]
    display(stats.round(1))

    bins = np.linspace(150, 2300, 22)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True, sharey=True)
    for ax, batch_id in zip(axes, life.columns):
        ax.hist(life[batch_id].dropna(), bins=bins, edgecolor="white")
        ax.axvline(life[batch_id].median(), color="red", linestyle="--", label="Median")
        ax.set(title=batch_id, xlabel="Cycle Life", ylabel="Battery count")
        ax.set_xlim(150, 2300)
        ax.legend()
    plt.tight_layout()
    writer.figure(plt.gcf(), "figure_01.png")

    life.boxplot(figsize=(8, 4))
    plt.ylabel("Cycle Life")
    plt.title("Cycle Life by Batch")
    writer.figure(plt.gcf(), "figure_02.png")
    return life, stats


def summarize_lifetimes(data_dir, life):

    life_rows, outlier_rows = [], []
    for bid in list(BATCH_FILES) + ["All"]:
        values = life.stack().dropna() if bid == "All" else life[bid].dropna()

        raw_n = 0
        for b, filename in BATCH_FILES.items():
            if bid == "All" or bid == b:
                with h5py.File(data_dir / filename, "r") as f:
                    raw_n += f["batch"]["cycle_life"].size
        life_rows.append(
            {
                "batch": bid,
                "labelled": len(values),
                "missing": raw_n - len(values),
                "short_n": int((values < 500).sum()),
                "short_pct": 100 * (values < 500).mean(),
                "middle_n": int(values.between(500, 1000).sum()),
                "long_n": int((values > 1000).sum()),
                "long_pct": 100 * (values > 1000).mean(),
                "outside_plot": int(((values < 150) | (values > 2300)).sum()),
            }
        )
        if bid != "All":
            q1, q3 = values.quantile([0.25, 0.75])
            low, high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
            for cell, value in values.items():
                outlier_rows.append(
                    {
                        "batch": bid,
                        "cell": cell,
                        "cycle_life": value,
                        "lower_fence": low,
                        "upper_fence": high,
                        "iqr_outlier": "low"
                        if value < low
                        else "high"
                        if value > high
                        else "none",
                    }
                )
    life_ratio_df = pd.DataFrame(life_rows)
    outlier_df = pd.DataFrame(outlier_rows)
    display(life_ratio_df.round(2))
    display(outlier_df.loc[outlier_df.iqr_outlier != "none"])
    print("배치별 짧은 수명 IQR 이상치 수:")
    print(
        outlier_df.assign(low=outlier_df.iqr_outlier.eq("low"))
        .groupby("batch")
        .low.sum()
    )
    return life_ratio_df, outlier_df
