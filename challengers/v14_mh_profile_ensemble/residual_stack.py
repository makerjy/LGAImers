from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import lightgbm as lgb
import numpy as np
import pandas as pd

from challengers.v14_mh_profile_ensemble.temporal_train import (
    ARTIFACT_DIR,
    CACHE_DIR,
    TARGET,
    VALID_SEASONS,
    brier,
)


ROOT = Path(__file__).resolve().parents[2]
V13_PATH = ROOT / "challengers/v11_feature_rebuild/artifacts/main_screen_oof.parquet"
TARGET_BRIER_1500 = 0.2462267860881907
SPECS = (
    ("d2_l4_m10000_t120", 4, 2, 10_000, 120),
    ("d3_l8_m5000_t160", 8, 3, 5_000, 160),
    ("d4_l15_m3000_t200", 15, 4, 3_000, 200),
)
SCALES = (0.25, 0.5, 0.75, 1.0)


def load_frame() -> tuple[pd.DataFrame, list[str], list[str]]:
    mh = pd.read_parquet(ARTIFACT_DIR / "mh_oof.parquet").drop(columns=[TARGET])
    v13 = pd.read_parquet(
        V13_PATH, columns=["row_id", "p_v13_centered_expert_best"]
    )
    raw = pd.read_csv(
        ROOT / "data/raw/train.csv",
        usecols=["row_id", "season", "game_type", "pitcher_id"],
        encoding="utf-8-sig",
    ).rename(columns={"game_type": "raw_game_type"})
    parts = []
    base_feature_columns: list[str] | None = None
    for season in VALID_SEASONS:
        part = pd.read_parquet(CACHE_DIR / f"features_{season}.parquet")
        current = [c for c in part if c not in {"row_id", TARGET}]
        if base_feature_columns is None:
            base_feature_columns = current
        elif current != base_feature_columns:
            raise ValueError("Feature schema changed across cached seasons")
        parts.append(part)
    frame = pd.concat(parts, ignore_index=True)
    frame = frame.merge(mh, on="row_id", validate="one_to_one")
    frame = frame.merge(v13, on="row_id", validate="one_to_one")
    frame = frame.merge(raw, on="row_id", validate="one_to_one")
    if base_feature_columns is None:
        raise RuntimeError("No cached features")
    prediction_columns = [c for c in mh if c.startswith("p_mh_")]
    feature_columns = [
        *base_feature_columns,
        *prediction_columns,
        "p_v13_centered_expert_best",
    ]
    categorical = [
        c
        for c in (
            "top_bottom",
            "game_type",
            "pitcher_hand",
            "batter_hand",
            "pitcher_team_id",
            "batter_team_id",
        )
        if c in feature_columns
    ]
    return frame, feature_columns, categorical


def fit_nested(
    frame: pd.DataFrame,
    feature_columns: list[str],
    categorical: list[str],
    *,
    leaves: int,
    depth: int,
    min_child: int,
    trees: int,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, float | int]]]:
    target = frame[TARGET].to_numpy(dtype="float64")
    base = frame["p_v13_centered_expert_best"].to_numpy(dtype="float64")
    seasons = frame["season"].to_numpy(dtype="int16")
    regular = frame["raw_game_type"].eq("R").to_numpy()
    correction = np.zeros(len(frame), dtype="float64")
    diagnostics = []
    for valid_season in (2023, 2024):
        train = regular & (seasons >= 2022) & (seasons < valid_season)
        valid = regular & (seasons == valid_season)
        x_train = frame.loc[train, feature_columns].copy()
        x_valid = frame.loc[valid, feature_columns].copy()
        residual = target[train] - base[train]
        train_seasons = seasons[train]
        sample_weight = np.power(
            0.7, valid_season - 1 - train_seasons
        ).astype("float32")
        model = lgb.LGBMRegressor(
            objective="regression_l2",
            n_estimators=trees,
            learning_rate=0.02,
            num_leaves=leaves,
            max_depth=depth,
            min_child_samples=min_child,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.75,
            reg_alpha=5.0,
            reg_lambda=200.0,
            random_state=seed + valid_season,
            n_jobs=10,
            verbosity=-1,
        )
        model.fit(
            x_train,
            residual,
            sample_weight=sample_weight,
            categorical_feature=categorical,
        )
        fold = np.clip(model.predict(x_valid), -0.03, 0.03)
        correction[valid] = fold
        diagnostics.append(
            {
                "valid_season": valid_season,
                "train_rows": int(train.sum()),
                "valid_rows": int(valid.sum()),
                "mean_abs_correction": float(np.mean(np.abs(fold))),
                "mean_correction": float(np.mean(fold)),
            }
        )
    return correction, diagnostics


def metric_row(
    frame: pd.DataFrame,
    correction: np.ndarray,
    scale: float,
    name: str,
    diagnostics: list[dict[str, float | int]],
) -> tuple[dict[str, object], np.ndarray]:
    target = frame[TARGET].to_numpy(dtype="float64")
    base = frame["p_v13_centered_expert_best"].to_numpy(dtype="float64")
    seasons = frame["season"].to_numpy(dtype="int16")
    prediction = np.clip(base + scale * correction, 1e-6, 1.0 - 1e-6)
    folds = {
        str(season): {
            "brier": brier(target[seasons == season], prediction[seasons == season]),
            "delta_vs_v13": brier(
                target[seasons == season], prediction[seasons == season]
            )
            - brier(target[seasons == season], base[seasons == season]),
        }
        for season in VALID_SEASONS
    }
    score = brier(target, prediction)
    return (
        {
            "name": name,
            "scale": scale,
            "pooled_brier": score,
            "delta_vs_v13": score - brier(target, base),
            "folds": folds,
            "gap_to_1500": score - TARGET_BRIER_1500,
            "diagnostics": diagnostics,
        },
        prediction.astype("float32"),
    )


def run() -> dict[str, object]:
    frame, feature_columns, categorical = load_frame()
    rows = []
    predictions: dict[str, np.ndarray] = {}
    for index, (name, leaves, depth, minimum, trees) in enumerate(SPECS):
        correction, diagnostics = fit_nested(
            frame,
            feature_columns,
            categorical,
            leaves=leaves,
            depth=depth,
            min_child=minimum,
            trees=trees,
            seed=9200 + index * 100,
        )
        for scale in SCALES:
            row, prediction = metric_row(
                frame, correction, scale, name, diagnostics
            )
            rows.append(row)
            predictions[f"{name}_s{str(scale).replace('.', '')}"] = prediction
            print(
                f"{name} scale={scale}: brier={row['pooled_brier']:.12f} "
                f"delta={row['delta_vs_v13']:+.12f}",
                flush=True,
            )
    best = min(rows, key=lambda row: row["pooled_brier"])
    key = f"{best['name']}_s{str(best['scale']).replace('.', '')}"
    output = frame[["row_id", TARGET, "season", "pitcher_id"]].copy()
    output["p_v14_nested_residual"] = predictions[key]
    output.to_parquet(
        ARTIFACT_DIR / "v14_residual_oof.parquet", index=False, compression="zstd"
    )
    result = {
        "protocol": "2023<-2022, 2024<-2022+2023; R-only residual; F stays V13",
        "feature_count": len(feature_columns),
        "categorical": categorical,
        "experiments": rows,
        "best": best,
    }
    (ARTIFACT_DIR / "residual_results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    print(json.dumps(run()["best"], indent=2))
