import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from battery_common.config import BATCH_FILES
from battery_common.artifacts import display


def analyze_policies(data_dir, writer):
    records = []
    for batch_id, filename in BATCH_FILES.items():
        with h5py.File(data_dir / filename, "r") as f:
            batch = f["batch"]
            for i, ref in enumerate(batch["cycle_life"][()].ravel()):
                lifespan = f[ref][()].item()
                policy_node = f[batch["policy_readable"][()].ravel()[i]]
                policy = "".join(chr(int(x)) for x in policy_node[()].ravel() if x)
                records.append([batch_id, i, policy, lifespan])
    policy_df = pd.DataFrame(records, columns=["batch", "cell", "policy", "cycle_life"])
    policy_stats = policy_df.groupby(["batch", "policy"])["cycle_life"].agg(
        ["count", "mean", "std"]
    )
    display(policy_stats.round(1))

    fig, axes = plt.subplots(1, 3, figsize=(18, 9), sharex=True)
    for ax, batch_id in zip(axes, BATCH_FILES):
        stats = policy_stats.loc[batch_id].dropna(subset=["mean"]).sort_values("mean")
        labels = [f"{p} (n={int(c)})" for p, c in zip(stats.index, stats["count"])]
        ax.barh(labels, stats["mean"], xerr=stats["std"].fillna(0), alpha=0.7)
        ax.set(title=batch_id, xlabel="Mean Cycle Life ± SD")
        ax.tick_params(axis="y", labelsize=7)
    plt.tight_layout()
    writer.figure(plt.gcf(), "figure_11.png")
    return policy_df, policy_stats


def analyze_protocols(data_dir, writer):
    import re

    pattern = re.compile(
        r"^(?P<C1>\d+(?:\.\d+)?)C\((?P<switch_pct>\d+(?:\.\d+)?)%\)-(?P<C2>\d+(?:\.\d+)?)C(?P<suffix>-newstructure)?$"
    )
    protocol_rows = []
    for batch_id, filename in BATCH_FILES.items():
        with h5py.File(data_dir / filename, "r") as f:
            batch = f["batch"]
            for i, ref in enumerate(batch["summary"][()].ravel()):
                policy = "".join(
                    chr(int(x))
                    for x in f[batch["policy_readable"][()].ravel()[i]][()].ravel()
                    if x
                )
                match = pattern.fullmatch(policy)
                summary = f[ref]
                cycle = summary["cycle"][()].ravel()
                qd = summary["QDischarge"][()].ravel()
                capacities = []
                for number in [100, 300]:
                    positions = np.flatnonzero(cycle == number)
                    value = qd[positions[0]] if len(positions) == 1 else np.nan
                    capacities.append(
                        value if np.isfinite(value) and value > 0 else np.nan
                    )
                q100, q300 = capacities
                row = {
                    "batch": batch_id,
                    "cell": i,
                    "policy": policy,
                    "Q100": q100,
                    "Q300": q300,
                    "protocol_valid": match is not None,
                    "group": "unparsed",
                    "C1": np.nan,
                    "switch_pct": np.nan,
                    "C2": np.nan,
                    "fade_pct_per_100": 100 * (q100 - q300) / q100 / 2,
                }
                if match:
                    row.update(
                        {key: float(match[key]) for key in ["C1", "switch_pct", "C2"]}
                    )
                    row["group"] = "newstructure" if match["suffix"] else "no suffix"
                protocol_rows.append(row)

    protocol_df = pd.DataFrame(protocol_rows)
    protocol_df["usable"] = protocol_df["protocol_valid"] & np.isfinite(
        protocol_df["fade_pct_per_100"]
    )
    quality = protocol_df.groupby("batch").agg(
        total=("cell", "size"),
        parsed=("protocol_valid", "sum"),
        usable=("usable", "sum"),
    )
    quality["excluded"] = quality["total"] - quality["usable"]
    display(quality)
    protocol_valid = protocol_df.loc[protocol_df["usable"]].copy()
    print("Excluded records (invalid policy format or missing endpoint capacity)")
    display(
        protocol_df.loc[
            ~protocol_df["usable"], ["batch", "cell", "policy", "Q100", "Q300"]
        ]
    )

    protocol_features = ["C1", "switch_pct", "C2"]
    correlation_rows = []
    for (batch_id, group_name), group in protocol_valid.groupby(["batch", "group"]):
        for feature in protocol_features:
            pair = group[[feature, "fade_pct_per_100"]].dropna()
            valid = len(pair) >= 3 and pair.nunique().min() > 1
            correlation_rows.append(
                {
                    "batch": batch_id,
                    "group": group_name,
                    "feature": feature,
                    "n": len(pair),
                    "Pearson r": pair[feature].corr(pair["fade_pct_per_100"])
                    if valid
                    else np.nan,
                    "Spearman rho": pair[feature].corr(
                        pair["fade_pct_per_100"], method="spearman"
                    )
                    if valid
                    else np.nan,
                }
            )
    protocol_corr = pd.DataFrame(correlation_rows)
    display(protocol_corr.set_index(["batch", "group", "feature"]).round(3))

    fig, axes = plt.subplots(3, 3, figsize=(14, 11), sharey=True)
    for row_index, batch_id in enumerate(BATCH_FILES):
        for col_index, feature in enumerate(protocol_features):
            ax = axes[row_index, col_index]
            for group_name, group in protocol_valid.loc[
                protocol_valid["batch"] == batch_id
            ].groupby("group"):
                ax.scatter(
                    group[feature],
                    group["fade_pct_per_100"],
                    marker="^" if group_name == "newstructure" else "o",
                    alpha=0.7,
                    label=group_name,
                )
            ax.axhline(0, color="gray", linewidth=0.7)
            ax.set(
                title=f"{batch_id}: {feature}",
                xlabel=feature,
                ylabel="Capacity loss (% / 100 cycles)" if col_index == 0 else "",
            )
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
    plt.tight_layout()
    writer.figure(plt.gcf(), "figure_12.png")
    return protocol_df, protocol_corr


def analyze_measured_current(data_dir, protocol_df, policy_df, writer):
    current_rows = []
    for bid, filename in BATCH_FILES.items():
        with h5py.File(data_dir / filename, "r") as f:
            b = f["batch"]
            for i, ref in enumerate(b["cycles"][()].ravel()):
                row = {
                    "batch": bid,
                    "cell": i,
                    "charge_mean_A": np.nan,
                    "charge_max_A": np.nan,
                    "status": "invalid cycle 10",
                }
                numbers = f[b["summary"][()].ravel()[i]]["cycle"][()].ravel()
                pos = np.flatnonzero(numbers == 10)
                c = f[ref]
                if len(pos) == 1 and all(pos[0] < c[k].size for k in ["I", "t"]):
                    nodes = [f[c[k][()].ravel()[pos[0]]] for k in ["I", "t"]]
                    amps, times = [node[()].ravel() for node in nodes]
                    if (
                        not any(node.attrs.get("MATLAB_empty", 0) for node in nodes)
                        and len(amps) == len(times)
                        and len(amps) > 1
                    ):
                        dt = np.diff(times)
                        ok = (
                            (dt > 0)
                            & np.isfinite(dt)
                            & np.isfinite(amps[:-1])
                            & np.isfinite(amps[1:])
                            & (amps[:-1] > 0)
                            & (amps[1:] > 0)
                        )
                        if ok.any():
                            row.update(
                                charge_mean_A=np.sum(
                                    (amps[:-1][ok] + amps[1:][ok]) / 2 * dt[ok]
                                )
                                / dt[ok].sum(),
                                charge_max_A=max(
                                    amps[:-1][ok].max(), amps[1:][ok].max()
                                ),
                                status="ok",
                            )
                        else:
                            row["status"] = "no valid positive-current intervals"
                    else:
                        row["status"] = "empty or mismatched arrays"
                current_rows.append(row)
    current_df = (
        pd.DataFrame(current_rows)
        .merge(
            protocol_df[["batch", "cell", "group", "fade_pct_per_100"]],
            on=["batch", "cell"],
        )
        .merge(policy_df[["batch", "cell", "cycle_life"]], on=["batch", "cell"])
    )
    display(pd.crosstab(current_df.batch, current_df.status))
    current_corr_rows = []
    for (bid, group_name), g in current_df.groupby(["batch", "group"]):
        for feature in ["charge_mean_A", "charge_max_A"]:
            for target in ["fade_pct_per_100", "cycle_life"]:
                pair = g[[feature, target]].replace([np.inf, -np.inf], np.nan).dropna()
                valid = len(pair) >= 3 and pair.nunique().min() > 1
                current_corr_rows.append(
                    {
                        "batch": bid,
                        "group": group_name,
                        "feature": feature,
                        "target": target,
                        "n": len(pair),
                        "Pearson": pair[feature].corr(pair[target])
                        if valid
                        else np.nan,
                        "Spearman": pair[feature].corr(pair[target], method="spearman")
                        if valid
                        else np.nan,
                    }
                )
    current_corr_df = pd.DataFrame(current_corr_rows)
    display(current_corr_df.round(3))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, bid in zip(axes, BATCH_FILES):
        for name, g in current_df.loc[current_df.batch == bid].groupby("group"):
            ax.scatter(g.charge_mean_A, g.fade_pct_per_100, label=name, alpha=0.7)
        ax.set(
            title=bid,
            xlabel="Cycle 10: mean charging current (A)",
            ylabel="Capacity loss (% / 100 cycles)",
        )
        ax.legend()
    plt.tight_layout()
    writer.figure(plt.gcf(), "figure_13.png")
    return current_df, current_corr_df
