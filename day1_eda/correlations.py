import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from battery_common.config import BATCH_FILES
from battery_common.artifacts import display


def analyze_early_correlations(data_dir, delta_df, writer):

    fields = {
        "QD": "QDischarge",
        "IR": "IR",
        "Tavg": "Tavg",
        "Tmax": "Tmax",
        "chargetime": "chargetime",
    }
    records = []
    for batch_id, filename in BATCH_FILES.items():
        with h5py.File(data_dir / filename, "r") as f:
            batch = f["batch"]
            for i, ref in enumerate(batch["summary"][()].ravel()):
                summary = f[ref]
                cycles = summary["cycle"][()].ravel()
                early = (cycles >= 1) & (cycles <= 100)
                row = {
                    "batch": batch_id,
                    "cell": i,
                    "cycle_life": f[batch["cycle_life"][()].ravel()[i]][()].item(),
                }
                for name, field in fields.items():
                    values = summary[field][()].ravel()[early]
                    values = values[np.isfinite(values)]
                    if name in ["QD", "IR", "chargetime"]:
                        values = values[values > 0]
                    row[name] = values.mean() if len(values) else np.nan
                records.append(row)
    early_df = pd.DataFrame(records).merge(
        delta_df[["batch", "cell", "log_var_deltaQ"]], on=["batch", "cell"], how="left"
    )
    features = list(fields) + ["log_var_deltaQ"]
    corr, counts = {}, {}
    for batch_id in list(BATCH_FILES) + ["All"]:
        group = (
            early_df if batch_id == "All" else early_df[early_df["batch"] == batch_id]
        )
        corr[batch_id] = group[features].corrwith(group["cycle_life"])
        counts[batch_id] = group.loc[group["cycle_life"].notna(), features].count()
    corr = pd.DataFrame(corr)
    display(corr.round(3))
    print("지표별 수명과의 유효 표본 수")
    display(pd.DataFrame(counts))

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)), corr.columns)
    ax.set_yticks(range(len(corr.index)), corr.index)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            value = corr.iloc[i, j]
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if abs(value) > 0.5 else "black",
            )
    fig.colorbar(im, ax=ax, label="Pearson r with Cycle Life")
    plt.tight_layout()
    writer.figure(plt.gcf(), "figure_14.png")
    return early_df, features, corr, counts


def analyze_collinearity(early_df, features, corr, counts, writer):

    ranking = corr["All"].rename("Pearson_r").to_frame()
    ranking["abs_r"] = ranking["Pearson_r"].abs()
    display(ranking.sort_values("abs_r", ascending=False).round(3))
    strongest = []
    for batch_id in corr.columns:
        available = corr[batch_id].dropna()
        if len(available):
            feature = available.abs().idxmax()
            strongest.append(
                {
                    "batch": batch_id,
                    "feature": feature,
                    "Pearson_r": available[feature],
                    "n": counts[batch_id][feature],
                }
            )
    display(pd.DataFrame(strongest).round(3))

    vif_rows, pair_rows, sample_rows = [], [], []
    fig, axes = plt.subplots(1, 4, figsize=(22, 5), constrained_layout=True)
    for ax, batch_id in zip(axes, list(BATCH_FILES) + ["All"]):
        group = (
            early_df
            if batch_id == "All"
            else early_df.loc[early_df["batch"] == batch_id]
        )
        complete = (
            group[features + ["cycle_life"]].replace([np.inf, -np.inf], np.nan).dropna()
        )
        X = complete[features]
        sample_rows.append(
            {
                "batch": batch_id,
                "total": len(group),
                "complete": len(X),
                "excluded": len(group) - len(X),
            }
        )
        matrix = X.corr()
        im = ax.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set(title=f"{batch_id} (n={len(X)})")
        ax.set_xticks(range(len(features)), features, rotation=90)
        ax.set_yticks(range(len(features)), features)
        for i, feature in enumerate(features):
            for j, other in enumerate(features):
                value = matrix.iloc[i, j]
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if abs(value) > 0.5 else "black",
                )
                if j > i and abs(value) >= 0.8:
                    pair_rows.append(
                        {
                            "batch": batch_id,
                            "feature_1": feature,
                            "feature_2": other,
                            "r": value,
                            "n": len(X),
                        }
                    )
        varying = [feature for feature in features if X[feature].nunique() > 1]
        Z = (X[varying] - X[varying].mean()) / X[varying].std(ddof=0)
        for feature in features:
            value = np.nan
            if feature in varying and len(X) > len(varying):
                y = Z[feature].to_numpy()
                design = np.column_stack(
                    [np.ones(len(X)), Z.drop(columns=feature).to_numpy()]
                )
                residual = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
                unexplained = np.sum(residual**2) / np.sum(y**2)
                value = np.inf if unexplained <= 1e-12 else 1 / unexplained
            vif_rows.append({"batch": batch_id, "feature": feature, "VIF": value})
    fig.colorbar(im, ax=axes, shrink=0.65, label="Pearson r between features")
    writer.figure(plt.gcf(), "figure_15.png")
    print("Complete-case sample counts")
    display(pd.DataFrame(sample_rows))
    print("Strong feature pairs: |r| >= 0.8")
    display(
        pd.DataFrame(
            pair_rows, columns=["batch", "feature_1", "feature_2", "r", "n"]
        ).round(3)
    )
    vif_table = pd.DataFrame(vif_rows).pivot(
        index="feature", columns="batch", values="VIF"
    )
    print("VIF (>= 5: inspect overlapping information)")
    display(vif_table.round(2))
    return vif_table


def audit_short_lifetimes(
    early_df, delta_features_df, features, outlier_df, policy_df, protocol_df
):

    extended_early = early_df.merge(
        delta_features_df[["batch", "cell", "delta_mean", "delta_min", "delta_range"]],
        on=["batch", "cell"],
        how="left",
    )
    extended_features = features + ["delta_mean", "delta_min", "delta_range"]
    extended_corr = pd.DataFrame(
        {
            bid: (
                extended_early
                if bid == "All"
                else extended_early.loc[extended_early.batch == bid]
            )[extended_features].corrwith(
                (
                    extended_early
                    if bid == "All"
                    else extended_early.loc[extended_early.batch == bid]
                ).cycle_life
            )
            for bid in list(BATCH_FILES) + ["All"]
        }
    )
    display(extended_corr.round(3))
    print("추가 ΔQ 피처를 포함한 전체 절대 상관 순위")
    display(extended_corr.All.abs().sort_values(ascending=False))
    short_audit = (
        outlier_df.merge(policy_df, on=["batch", "cell", "cycle_life"])
        .merge(
            protocol_df[["batch", "cell", "Q100", "Q300", "fade_pct_per_100"]],
            on=["batch", "cell"],
        )
        .merge(
            early_df[["batch", "cell", "Tavg", "Tmax", "IR", "log_var_deltaQ"]],
            on=["batch", "cell"],
        )
    )
    short_audit["bottom3"] = (
        short_audit.groupby("batch").cycle_life.rank(method="first") <= 3
    )
    short_audit = short_audit.loc[
        (short_audit.cycle_life < 500) | short_audit.bottom3
    ].sort_values(["batch", "cycle_life"])
    display(short_audit.round(4))
    print("cell은 0부터 시작하는 원본 기록 번호. 원인 확정이 아니라 후속 점검 대상임.")
    return extended_corr, short_audit
