from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from battery_common.config import BATCH_FILES, default_data_dir

FEATURES = [
    "qd_mean_2_5",
    "qd_delta_5_2",
    "ir_mean_2_5",
    "tavg_mean_2_5",
    "tmax_mean_2_5",
    "chargetime_mean_2_5",
]
FIELD_MAP = {
    "QDischarge": "qd_mean_2_5",
    "IR": "ir_mean_2_5",
    "Tavg": "tavg_mean_2_5",
    "Tmax": "tmax_mean_2_5",
    "chargetime": "chargetime_mean_2_5",
}


def make_target(life):

    if not np.isfinite(life) or life <= 0:
        return np.nan
    return 1 if life < 500 else (0 if life > 1000 else np.nan)


def early_features(summary):

    cycle = np.asarray(summary["cycle"], dtype=float).ravel()
    positions = [np.flatnonzero(cycle == c) for c in range(2, 6)]
    if any(len(pos) != 1 for pos in positions):
        return {key: np.nan for key in FEATURES}, False
    idx = np.array([pos[0] for pos in positions])
    result, selected = {}, {}
    for field, name in FIELD_MAP.items():
        values = np.asarray(summary[field], dtype=float).ravel()
        if len(values) != len(cycle):
            raise ValueError(f"Summary length mismatch: {field}")
        values = values[idx].copy()
        values[~np.isfinite(values)] = np.nan

        if field in ("QDischarge", "IR", "chargetime"):
            values[values <= 0] = np.nan
        selected[field] = values
        result[name] = (
            float(np.nanmean(values)) if np.isfinite(values).any() else np.nan
        )
    result["qd_delta_5_2"] = selected["QDischarge"][-1] - selected["QDischarge"][0]
    return result, True


def load_features(data_dir=None):

    data_dir = default_data_dir() if data_dir is None else Path(data_dir)
    rows = []
    for batch_name, filename in BATCH_FILES.items():
        path = Path(data_dir) / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}; run python -m day1_eda.download_data or set BATTERY_DATA_DIR."
            )
        with h5py.File(path, "r") as f:
            batch = f["batch"]
            for i, ref in enumerate(batch["summary"][()].ravel()):
                summary = f[ref]
                values = {
                    key: summary[key][()].ravel() for key in ["cycle", *FIELD_MAP]
                }
                features, complete = early_features(values)
                raw_life = f[batch["cycle_life"][()].ravel()[i]][()].ravel()
                life = float(raw_life[0]) if len(raw_life) == 1 else np.nan
                target = make_target(life)
                reason = (
                    "missing_or_invalid_life"
                    if not np.isfinite(life) or life <= 0
                    else "middle_life"
                    if np.isnan(target)
                    else "missing_or_duplicate_cycles_2_5"
                    if not complete
                    else "all_features_missing"
                    if not any(np.isfinite(v) for v in features.values())
                    else "included"
                )
                rows.append(
                    {
                        "record_id": f"b{batch_name[-1]}c{i}",
                        "batch": batch_name,
                        "cell_index": i,
                        "cycle_life": life,
                        "target": target,
                        "status": reason,
                        **features,
                    }
                )
    audit = pd.DataFrame(rows)
    data = audit.loc[audit.status.eq("included")].copy().reset_index(drop=True)
    data["target"] = data.target.astype(int)
    assert data.record_id.is_unique
    return data, audit
