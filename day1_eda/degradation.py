import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from battery_common.config import BATCH_FILES
from battery_common.artifacts import display


def plot_capacity_curves(data_dir, writer):

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

    for ax, (batch_id, filename) in zip(axes, BATCH_FILES.items()):
        with h5py.File(data_dir / filename, "r") as f:
            refs = f["batch"]["summary"][()].ravel()
            for ref in refs:
                summary = f[ref]
                cycle = summary["cycle"][()].ravel()
                qd = summary["QDischarge"][()].ravel().astype(float)
                qd[(qd <= 0) | ~np.isfinite(qd)] = np.nan
                ax.plot(cycle, qd, linewidth=0.8, alpha=0.6)
        ax.set(title=batch_id, xlabel="Cycle", ylabel="Discharge capacity (Ah)")
        ax.grid(alpha=0.2)

    plt.tight_layout()
    writer.figure(plt.gcf(), "figure_03.png")
    return None


def fit_knee_candidate(cycle, capacity):
    valid = np.isfinite(cycle) & np.isfinite(capacity) & (cycle >= 10) & (capacity > 0)
    series = (
        pd.Series(capacity[valid], index=cycle[valid])
        .groupby(level=0)
        .median()
        .sort_index()
    )
    if len(series) < 100 or series.index[-1] - series.index[0] < 200:
        return {"status": "insufficient data"}, None
    smooth = series.rolling(11, center=True, min_periods=1).median()
    idx = np.unique(np.linspace(0, len(series) - 1, min(400, len(series))).astype(int))
    x, y = smooth.index.to_numpy()[idx], smooth.to_numpy()[idx]

    t = (x - x[0]) / 100
    line_design = np.column_stack([np.ones(len(t)), t])
    line_coef = np.linalg.lstsq(line_design, y, rcond=None)[0]
    line_sse = np.sum((y - line_design @ line_coef) ** 2)
    best = None
    for k in np.linspace(t[0] + 0.2 * np.ptp(t), t[0] + 0.8 * np.ptp(t), 61):
        if min(np.sum(t <= k), np.sum(t > k)) < 30:
            continue
        design = np.column_stack([line_design, np.maximum(0, t - k)])
        coef = np.linalg.lstsq(design, y, rcond=None)[0]
        sse = np.sum((y - design @ coef) ** 2)
        if best is None or sse < best[0]:
            best = (sse, k, coef, design @ coef)
    if best is None:
        return {"status": "insufficient segments"}, None
    sse, k, coef, fitted = best
    before, after = -coef[1], -(coef[1] + coef[2])
    gain = 1 - sse / line_sse if line_sse > 1e-15 else 0.0
    ratio = after / before if before > 1e-8 else np.nan
    candidate = bool(before > 1e-8 and after > 0 and ratio >= 2 and gain >= 0.30)
    result = {
        "status": "candidate" if candidate else "not flagged",
        "split_cycle": x[0] + 100 * k,
        "knee_cycle": x[0] + 100 * k if candidate else np.nan,
        "rate_before": before,
        "rate_after": after,
        "rate_ratio": ratio,
        "sse_improvement": gain,
    }
    return result, (series, x, fitted)


def analyze_knees(data_dir, writer):

    knee_rows, knee_curves = [], {}
    for batch_id, filename in BATCH_FILES.items():
        with h5py.File(data_dir / filename, "r") as f:
            for i, ref in enumerate(f["batch"]["summary"][()].ravel()):
                summary = f[ref]
                result, curve = fit_knee_candidate(
                    summary["cycle"][()].ravel(), summary["QDischarge"][()].ravel()
                )
                knee_rows.append({"batch": batch_id, "cell": i, **result})
                if curve is not None:
                    knee_curves[(batch_id, i)] = curve
    knee_df = pd.DataFrame(knee_rows)
    print("Candidate counts (cell is a zero-based record index)")
    display(pd.crosstab(knee_df["batch"], knee_df["status"]))
    print("Candidate knee locations: descriptive summary, not confidence intervals")
    display(
        knee_df.groupby("batch")["knee_cycle"]
        .agg(["count", "median", "min", "max"])
        .round(1)
    )
    display(knee_df.round(4))

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for col, batch_id in enumerate(BATCH_FILES):
        group = knee_df.loc[knee_df["batch"] == batch_id]
        ax = axes[0, col]
        for _, row in group.iterrows():
            curve = knee_curves.get((batch_id, row["cell"]))
            if curve is None:
                continue
            series, x, fitted = curve
            ax.plot(series.index, series, alpha=0.25, linewidth=0.7)
            if row["status"] == "candidate":
                ax.scatter(
                    row["knee_cycle"],
                    np.interp(row["knee_cycle"], x, fitted),
                    color="black",
                    s=12,
                )
        ax.set(
            title=f"{batch_id}: candidate knees (black dots)",
            xlabel="Cycle",
            ylabel="Qd (Ah)",
        )
        candidates = group.loc[group["status"] == "candidate"]
        ax = axes[1, col]
        if len(candidates):
            chosen = candidates.loc[
                (candidates["sse_improvement"] - candidates["sse_improvement"].median())
                .abs()
                .idxmin()
            ]
            series, x, fitted = knee_curves[(batch_id, chosen["cell"])]
            ax.plot(series.index, series, alpha=0.5, label="Observed")
            ax.plot(x, fitted, color="black", label="Two-segment fit")
            ax.axvline(
                chosen["knee_cycle"],
                color="red",
                linestyle="--",
                label=f"Candidate: {chosen['knee_cycle']:.0f}",
            )
            ax.set_title(
                f"{batch_id}, cell {int(chosen['cell'])}: illustrative candidate"
            )
            ax.legend(fontsize=8)
        else:
            ax.set_title(f"{batch_id}: no candidate")
        ax.set(xlabel="Cycle", ylabel="Qd (Ah)")
    for ax in axes.ravel():
        ax.grid(alpha=0.2)
    plt.tight_layout()
    writer.figure(plt.gcf(), "figure_04.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    for batch_id, group in knee_df.dropna(subset=["rate_before", "rate_after"]).groupby(
        "batch"
    ):
        ax.scatter(
            group["rate_before"], group["rate_after"], alpha=0.65, label=batch_id
        )
    limits = ax.get_xlim() + ax.get_ylim()
    lo, hi = min(limits), max(limits)
    ax.plot([lo, hi], [lo, hi], "k--", label="Equal rates")
    ax.set(
        xlabel="Before split: capacity loss (Ah / 100 cycles)",
        ylabel="After split: capacity loss (Ah / 100 cycles)",
        title="Above diagonal: faster capacity loss after split",
    )
    ax.legend()
    plt.tight_layout()
    writer.figure(plt.gcf(), "figure_05.png")
    return knee_df
