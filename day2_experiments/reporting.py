import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from battery_common.artifacts import ArtifactWriter, display


def plot_scores(results, writer):
    overall = results["overall_metrics"].set_index("model")
    display(overall.round(3))
    score_columns = [
        "accuracy",
        "recall_short",
        "precision_short",
        "f1_short",
        "f2_short",
        "balanced_accuracy",
    ]
    display(
        results["fold_metrics"]
        .groupby("model")[score_columns]
        .agg(["mean", "std"])
        .round(3)
    )
    display(
        results["fold_metrics"][
            [
                "model",
                "fold",
                "train_f2",
                "f2_short",
                "f2_train_minus_validation",
                "best_params",
            ]
        ]
    )
    ax = overall[score_columns].plot.bar(figsize=(11, 4), ylim=(0, 1.08), rot=15)
    ax.set_ylabel("OOF score (exploratory)")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(writer.directory / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close("all")


def plot_confusion_matrices(results, writer):
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    for ax, (name, group) in zip(
        axes, results["oof_predictions"].groupby("model", sort=False)
    ):
        cm = confusion_matrix(group.y_true, group.y_pred, labels=[0, 1])
        ConfusionMatrixDisplay(cm, display_labels=["Long (0)", "Short (1)"]).plot(
            ax=ax, colorbar=False
        )
        ax.set_title(name)
    plt.tight_layout()
    plt.savefig(writer.directory / "confusion_matrices.png", dpi=150)
    plt.close("all")


def report_errors(results, writer):
    display(results["batch_metrics"].round(3))
    oof = results["oof_predictions"]
    errors = oof.loc[oof.y_true.ne(oof.y_pred)].copy()
    errors["error_type"] = np.where(
        errors.y_true.eq(1), "FN: missed short", "FP: flagged long"
    )
    display(errors)
    errors.to_csv(writer.directory / "errors.csv", index=False)


def plot_feature_weights(results, writer):
    weights = (
        results["feature_weights"]
        .groupby(["model", "feature"])
        .weight.agg(["mean", "std"])
    )
    display(weights.round(4))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, name in zip(axes, ["LogisticRegression", "RandomForest"]):
        table = weights.loc[name].sort_values("mean")
        ax.barh(table.index, table["mean"], xerr=table["std"].fillna(0))
        ax.axvline(0, color="grey", linewidth=0.8)
        ax.set_title(name)
        ax.set_xlabel("Fold mean ± SD")
    plt.tight_layout()
    plt.savefig(writer.directory / "feature_weights.png", dpi=150)
    plt.close("all")


def write_report(results, output_dir):

    writer = ArtifactWriter(output_dir)
    writer.tables(results)
    plot_scores(results, writer)
    plot_confusion_matrices(results, writer)
    report_errors(results, writer)
    plot_feature_weights(results, writer)
