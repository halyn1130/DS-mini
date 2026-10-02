from pathlib import Path
import os

BATCH_FILES = {
    "Batch 1": "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    "Batch 2": "2018-02-20_batchdata_updated_struct_errorcorrect.mat",
    "Batch 3": "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
}
BATCH_ORDER = tuple(BATCH_FILES)
BATCH_COLORS = dict(zip(BATCH_ORDER, ["#2878B5", "#E07B39", "#3C956B"]))


def default_data_dir():
    fallback = (
        Path.home()
        / ".cache/kagglehub/datasets/itshpark/data-driven-prediction-of-battery-cycle/versions/1"
    )
    return Path(os.environ.get("BATTERY_DATA_DIR", str(fallback))).expanduser()


def validate_data_dir(data_dir):
    path = Path(data_dir).expanduser()
    missing = [name for name in BATCH_FILES.values() if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing data in {path}: {missing}")
    return path
