import argparse
from pathlib import Path

from battery_common.config import default_data_dir, validate_data_dir


def run_analysis(data_dir, output_dir):
    from battery_common.artifacts import ArtifactWriter
    from .distribution import analyze_distribution, summarize_lifetimes
    from .degradation import plot_capacity_curves, analyze_knees
    from .delta import (
        analyze_delta_curves,
        compare_relative_lifetimes,
        compare_absolute_lifetimes,
    )
    from .charging import analyze_policies, analyze_protocols, analyze_measured_current
    from .correlations import (
        analyze_early_correlations,
        analyze_collinearity,
        audit_short_lifetimes,
    )

    data_dir = validate_data_dir(data_dir)
    writer = ArtifactWriter(output_dir)
    life, stats = analyze_distribution(data_dir, writer)
    life_ratio, outliers = summarize_lifetimes(data_dir, life)
    plot_capacity_curves(data_dir, writer)
    knees = analyze_knees(data_dir, writer)
    delta = analyze_delta_curves(data_dir, writer)
    delta_features, curves, grids, delta_correlations = compare_relative_lifetimes(
        data_dir, writer
    )
    absolute_delta = compare_absolute_lifetimes(delta_features, curves, grids, writer)
    policies, policy_stats = analyze_policies(data_dir, writer)
    protocols, protocol_correlations = analyze_protocols(data_dir, writer)
    current, current_correlations = analyze_measured_current(
        data_dir, protocols, policies, writer
    )
    early, features, correlations, counts = analyze_early_correlations(
        data_dir, delta, writer
    )
    vif = analyze_collinearity(early, features, correlations, counts, writer)
    extended, short_audit = audit_short_lifetimes(
        early, delta_features, features, outliers, policies, protocols
    )
    tables = {
        "life": life,
        "lifetime_stats": stats,
        "life_ratio_df": life_ratio,
        "outlier_df": outliers,
        "knee_df": knees,
        "delta_df": delta,
        "delta_features_df": delta_features,
        "delta_feature_correlations": delta_correlations,
        "absolute_delta_stats": absolute_delta,
        "policy_df": policies,
        "policy_stats": policy_stats,
        "protocol_df": protocols,
        "protocol_corr": protocol_correlations,
        "current_df": current,
        "current_corr_df": current_correlations,
        "early_df": early,
        "corr": correlations,
        "vif_table": vif,
        "extended_corr": extended,
        "short_audit": short_audit,
    }
    writer.tables(tables, index=True)
    print(f"EDA results saved to {writer.directory}")
    return tables


def main():
    parser = argparse.ArgumentParser(description="Battery EDA")
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).parent / "results"
    )
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    run_analysis(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
