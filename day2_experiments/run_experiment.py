import argparse
import json
from pathlib import Path
import sys

from battery_common.config import default_data_dir, validate_data_dir


def run_experiment(data_dir, output_dir):

    import sklearn
    from battery_common.artifacts import ArtifactWriter
    from .features import FEATURES, load_features
    from .models import SEED
    from .evaluation import evaluate_models
    from .reporting import write_report

    data, audit = load_features(validate_data_dir(data_dir))
    writer = ArtifactWriter(output_dir)
    writer.tables({"record_audit": audit, "early_features": data})
    print(f"EXPLORATORY: {len(data)} raw records; physical-cell identities unverified.")
    results = evaluate_models(data)
    write_report(results, writer.directory)
    run_info = {
        "status": "exploratory_raw_records_not_physical_cell_validated",
        "seed": SEED,
        "python": sys.version.split()[0],
        "sklearn": sklearn.__version__,
        "n_records": len(data),
        "features": FEATURES,
        "outer_folds": 5,
        "inner_folds": 3,
        "selection_metric": "F2 for short class",
        "threshold": "estimator default; no tuning",
    }
    (writer.directory / "run_info.json").write_text(
        json.dumps(run_info, indent=2), encoding="utf-8"
    )
    print(f"Experiment results saved to {writer.directory}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Initial-five-cycle battery classification"
    )
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).parent / "results"
    )
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    run_experiment(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
