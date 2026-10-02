import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from battery_common.config import BATCH_FILES
from battery_common.artifacts import display

DELTA_FEATURES = ["delta_mean", "delta_min", "delta_range", "log_var_deltaQ"]


def analyze_delta_curves(data_dir, writer):

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    delta_records = []
    for ax, (batch_id, filename) in zip(axes, BATCH_FILES.items()):
        skipped = 0
        with h5py.File(data_dir / filename, "r") as f:
            batch = f["batch"]
            for i, ref in enumerate(batch["cycles"][()].ravel()):
                cycles = f[ref]
                summary = f[batch["summary"][()].ravel()[i]]
                numbers = summary["cycle"][()].ravel()
                positions = [np.flatnonzero(numbers == c) for c in [10, 100]]
                if any(len(pos) != 1 for pos in positions):
                    skipped += 1
                    continue
                qrefs = cycles["Qdlin"][()].ravel()
                if max(pos[0] for pos in positions) >= len(qrefs):
                    skipped += 1
                    continue
                nodes = [f[qrefs[pos[0]]] for pos in positions]
                voltage = f[batch["Vdlin"][()].ravel()[i]][()].ravel()
                q10, q100 = [node[()].ravel() for node in nodes]
                if any(node.attrs.get("MATLAB_empty", 0) for node in nodes) or not (
                    len(q10) == len(q100) == len(voltage)
                ):
                    skipped += 1
                    continue
                delta = q100 - q10
                if not np.isfinite(delta).all() or not np.isfinite(voltage).all():
                    skipped += 1
                    continue
                ax.plot(voltage, delta, linewidth=0.8, alpha=0.6)
                variance = np.var(delta)
                lifespan = f[batch["cycle_life"][()].ravel()[i]][()].item()
                delta_records.append(
                    [
                        batch_id,
                        i,
                        np.log10(variance) if variance > 0 else np.nan,
                        lifespan,
                    ]
                )
        print(f"{batch_id}: ΔQ 제외 {skipped}개")
        ax.set(title=batch_id, xlabel="Voltage (V)", ylabel="Q100 - Q10 (Ah)")
        ax.grid(alpha=0.2)
    plt.tight_layout()
    writer.figure(plt.gcf(), "figure_06.png")

    delta_df = pd.DataFrame(
        delta_records, columns=["batch", "cell", "log_var_deltaQ", "cycle_life"]
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    for batch_id, group in delta_df.groupby("batch"):
        ax.scatter(
            group["log_var_deltaQ"], group["cycle_life"], label=batch_id, alpha=0.7
        )
    ax.set(xlabel="log10(var(ΔQ))", ylabel="Cycle Life")
    ax.legend()
    plt.tight_layout()
    writer.figure(plt.gcf(), "figure_07.png")
    return delta_df


def compare_relative_lifetimes(data_dir, writer):

    delta_feature_rows, delta_curve_groups, voltage_grids = [], {}, {}
    delta_exclusions = []
    for batch_id, filename in BATCH_FILES.items():
        with h5py.File(data_dir / filename, "r") as f:
            batch = f["batch"]
            for i, ref in enumerate(batch["cycles"][()].ravel()):
                cycles = f[ref]
                summary = f[batch["summary"][()].ravel()[i]]
                numbers = summary["cycle"][()].ravel()
                positions = [np.flatnonzero(numbers == number) for number in [10, 100]]
                reason = None
                if any(len(pos) != 1 for pos in positions):
                    reason = "missing or duplicate cycle 10/100"
                else:
                    qrefs = cycles["Qdlin"][()].ravel()
                    if max(pos[0] for pos in positions) >= len(qrefs):
                        reason = "Qdlin index missing"
                    else:
                        nodes = [f[qrefs[pos[0]]] for pos in positions]
                        q10, q100 = [node[()].ravel() for node in nodes]
                        voltage = f[batch["Vdlin"][()].ravel()[i]][()].ravel()
                        if (
                            any(node.attrs.get("MATLAB_empty", 0) for node in nodes)
                            or not (len(q10) == len(q100) == len(voltage))
                            or len(voltage) < 2
                        ):
                            reason = "empty or mismatched arrays"
                        elif not all(
                            np.isfinite(a).all() for a in [q10, q100, voltage]
                        ):
                            reason = "nonfinite data"
                if reason:
                    delta_exclusions.append(
                        {"batch": batch_id, "cell": i, "reason": reason}
                    )
                    continue
                order = np.argsort(voltage)
                voltage, delta = voltage[order], (q100 - q10)[order]
                if batch_id in voltage_grids:
                    grid = voltage_grids[batch_id]
                    if grid.shape != voltage.shape or not np.allclose(
                        grid, voltage, rtol=0, atol=1e-8
                    ):
                        raise ValueError(f"Voltage grid mismatch: {batch_id}, cell {i}")
                else:
                    voltage_grids[batch_id] = voltage
                variance = np.var(delta)
                lifespan = f[batch["cycle_life"][()].ravel()[i]][()].item()
                delta_feature_rows.append(
                    {
                        "batch": batch_id,
                        "cell": i,
                        "cycle_life": lifespan,
                        "delta_mean": delta.mean(),
                        "delta_min": delta.min(),
                        "delta_range": np.ptp(delta),
                        "log_var_deltaQ": np.log10(variance)
                        if variance > 0
                        else np.nan,
                    }
                )
                delta_curve_groups[(batch_id, i)] = delta

    delta_features_df = pd.DataFrame(delta_feature_rows)
    delta_features_df["life_cutoff"] = delta_features_df.groupby("batch")[
        "cycle_life"
    ].transform("median")
    delta_features_df["life_group"] = "Missing label"
    labelled = delta_features_df["cycle_life"].notna()
    delta_features_df.loc[labelled, "life_group"] = np.where(
        delta_features_df.loc[labelled, "cycle_life"]
        >= delta_features_df.loc[labelled, "life_cutoff"],
        "Long",
        "Short",
    )
    print("Batch-specific life cutoffs and group counts")
    display(delta_features_df.groupby("batch")["life_cutoff"].first().to_frame())
    display(pd.crosstab(delta_features_df["batch"], delta_features_df["life_group"]))
    print(f"Curve exclusions: {len(delta_exclusions)}")
    if delta_exclusions:
        display(pd.DataFrame(delta_exclusions))

    comparison = delta_features_df.loc[labelled].copy()
    group_colors = {"Short": "#E07B39", "Long": "#2878B5"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)
    for ax, batch_id in zip(axes, BATCH_FILES):
        for name in ["Short", "Long"]:
            group = comparison.loc[
                (comparison["batch"] == batch_id) & (comparison["life_group"] == name)
            ]
            if group.empty:
                continue
            curves = np.stack(
                [delta_curve_groups[(batch_id, i)] for i in group["cell"]]
            )
            q25, median, q75 = np.quantile(curves, [0.25, 0.5, 0.75], axis=0)
            voltage = voltage_grids[batch_id]
            ax.plot(
                voltage,
                median,
                color=group_colors[name],
                label=f"{name} (n={len(group)})",
            )
            ax.fill_between(voltage, q25, q75, color=group_colors[name], alpha=0.2)
        ax.axhline(0, color="gray", linewidth=0.7)
        ax.set(title=batch_id, xlabel="Voltage (V)", ylabel="Q100 - Q10 (Ah)")
        ax.legend()
        ax.grid(alpha=0.2)
    plt.tight_layout()
    writer.figure(plt.gcf(), "figure_08.png")

    print("Group feature distributions")
    display(
        comparison.groupby(["batch", "life_group"])[DELTA_FEATURES]
        .agg(["median", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)])
        .rename(columns={"<lambda_0>": "q25", "<lambda_1>": "q75"})
        .round(5)
    )
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    for row, batch_id in enumerate(BATCH_FILES):
        for col, feature in enumerate(DELTA_FEATURES):
            ax = axes[row, col]
            for position, name in enumerate(["Short", "Long"]):
                values = (
                    comparison.loc[
                        (comparison["batch"] == batch_id)
                        & (comparison["life_group"] == name),
                        feature,
                    ]
                    .dropna()
                    .to_numpy()
                )
                if len(values):
                    ax.boxplot(
                        [values], positions=[position], widths=0.45, showfliers=False
                    )
                    ax.scatter(
                        position + np.linspace(-0.12, 0.12, len(values)),
                        values,
                        color=group_colors[name],
                        alpha=0.6,
                        s=15,
                    )
            ax.set_xticks([0, 1], ["Short", "Long"])
            ax.set_title(f"{batch_id}: {feature}", fontsize=10)
            ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    writer.figure(plt.gcf(), "figure_09.png")

    delta_relation_rows = []
    for batch_id in list(BATCH_FILES) + ["All"]:
        group = (
            comparison
            if batch_id == "All"
            else comparison.loc[comparison["batch"] == batch_id]
        )
        for feature in DELTA_FEATURES:
            pair = group[[feature, "cycle_life"]].dropna()
            delta_relation_rows.append(
                {
                    "batch": batch_id,
                    "feature": feature,
                    "n": len(pair),
                    "Pearson": pair[feature].corr(pair["cycle_life"]),
                    "Spearman": pair[feature].corr(
                        pair["cycle_life"], method="spearman"
                    ),
                }
            )
    delta_feature_correlations = pd.DataFrame(delta_relation_rows)
    display(delta_feature_correlations.set_index(["batch", "feature"]).round(3))
    print("Feature overlap: pooled Pearson correlations (batch effects may contribute)")
    display(comparison[DELTA_FEATURES].corr().round(3))
    return (
        delta_features_df,
        delta_curve_groups,
        voltage_grids,
        delta_feature_correlations,
    )


def compare_absolute_lifetimes(
    delta_features_df, delta_curve_groups, voltage_grids, writer
):

    absolute_delta = delta_features_df.copy()
    absolute_delta["absolute_group"] = np.select(
        [absolute_delta.cycle_life < 500, absolute_delta.cycle_life > 1000],
        ["Short <500", "Long >1000"],
        default="Middle / missing",
    )
    display(pd.crosstab(absolute_delta.batch, absolute_delta.absolute_group))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)
    for ax, bid in zip(axes, BATCH_FILES):
        for name, color in [("Short <500", "#E07B39"), ("Long >1000", "#2878B5")]:
            group = absolute_delta.loc[
                (absolute_delta.batch == bid) & (absolute_delta.absolute_group == name)
            ]
            if group.empty:
                ax.plot([], [], color=color, label=f"{name}: n=0 (unavailable)")
                continue
            curves = np.stack([delta_curve_groups[(bid, i)] for i in group.cell])
            q25, median, q75 = np.quantile(curves, [0.25, 0.5, 0.75], axis=0)
            ax.plot(
                voltage_grids[bid], median, color=color, label=f"{name}: n={len(group)}"
            )
            ax.fill_between(voltage_grids[bid], q25, q75, color=color, alpha=0.2)
        ax.set(title=bid, xlabel="Voltage (V)", ylabel="Q100 - Q10 (Ah)")
        ax.legend(fontsize=8)
    plt.tight_layout()
    writer.figure(plt.gcf(), "figure_10.png")
    absolute_delta_stats = (
        absolute_delta.loc[absolute_delta.absolute_group != "Middle / missing"]
        .groupby(["batch", "absolute_group"])[DELTA_FEATURES]
        .agg(["count", "median"])
    )
    display(absolute_delta_stats.round(5))
    return absolute_delta_stats
