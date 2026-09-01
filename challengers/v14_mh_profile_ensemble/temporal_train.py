from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from catboost import CatBoostClassifier, Pool

from challengers.v14_mh_profile_ensemble.features import (
    CAT_COLS,
    DEFAULT_CONFIG,
    build_features,
)
from challengers.v14_mh_profile_ensemble.profiles import (
    build_profiles,
    compute_shrink_priors,
)


ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data/raw/train.csv"
ARTIFACT_DIR = ROOT / "challengers/v14_mh_profile_ensemble/artifacts"
CACHE_DIR = ARTIFACT_DIR / "temporal_features"
PRED_DIR = ARTIFACT_DIR / "predictions"
VALID_SEASONS = (2022, 2023, 2024)
FEATURE_SEASONS = (2020, 2021, 2022, 2023, 2024)
TARGET = "control_success"


CAT_PARAMS = {
    "cat_a": {
        "iterations": 600,
        "learning_rate": 0.02,
        "depth": 7,
        "l2_leaf_reg": 25.0,
        "min_data_in_leaf": 100,
    },
    "cat_b": {
        "iterations": 600,
        "learning_rate": 0.035,
        "depth": 5,
        "l2_leaf_reg": 3.0,
        "min_data_in_leaf": 100,
    },
    "cat_c": {
        "iterations": 1000,
        "learning_rate": 0.02,
        "depth": 4,
        "l2_leaf_reg": 8.0,
        "min_data_in_leaf": 500,
    },
}


def logit(values: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(values, dtype="float64"), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def sigmoid(values: np.ndarray) -> np.ndarray:
    z = np.clip(np.asarray(values, dtype="float64"), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def brier(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean(np.square(target - prediction)))


def feature_config(history: pd.DataFrame) -> dict[str, object]:
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(
        {
            "use_profiles": True,
            "use_shrinkage": True,
            "keep_season": False,
            "keep_raw_ids": False,
            "use_season_to_date": True,
            "std_min_n": 20,
            "shrink_priors": compute_shrink_priors(history),
        }
    )
    return cfg


def ensure_feature_cache(raw: pd.DataFrame, force: bool = False) -> list[str]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    schema: list[str] | None = None
    for season in FEATURE_SEASONS:
        path = CACHE_DIR / f"features_{season}.parquet"
        if path.exists() and not force:
            current = pq.read_schema(path).names
            current = [c for c in current if c not in {"row_id", TARGET}]
            if schema is None:
                schema = current
            elif current != schema:
                raise ValueError(f"Feature schema mismatch in {path}")
            print(f"cache hit season={season} path={path.name}", flush=True)
            continue

        history = raw.loc[raw["season"].lt(season)]
        current = raw.loc[raw["season"].eq(season)]
        if history.empty:
            raise ValueError(f"No strict-past history for season {season}")
        prof = build_profiles(raw, season)
        cfg = feature_config(history)
        features = build_features(current, cfg, prof)
        frame = pd.concat(
            [
                current[["row_id", TARGET]].reset_index(drop=True),
                features.reset_index(drop=True),
            ],
            axis=1,
        )
        if schema is None:
            schema = list(features.columns)
        elif list(features.columns) != schema:
            raise ValueError(f"Feature schema changed in season {season}")
        frame.to_parquet(path, index=False, compression="zstd")
        print(
            f"cache built season={season} rows={len(frame):,} "
            f"features={len(features.columns)} path={path.name}",
            flush=True,
        )
        del features, frame, prof
        gc.collect()
    if schema is None:
        raise RuntimeError("Feature cache was not created")
    (CACHE_DIR / "schema.json").write_text(
        json.dumps(schema, indent=2), encoding="utf-8"
    )
    return schema


def load_fold(
    valid_season: int, schema: list[str]
) -> tuple[
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
    np.ndarray,
    pd.DataFrame,
]:
    first_train = max(2020, valid_season - 4)
    train_seasons = list(range(first_train, valid_season))
    train_parts = []
    weight_parts = []
    latest = valid_season - 1
    for season in train_seasons:
        part = pd.read_parquet(CACHE_DIR / f"features_{season}.parquet")
        train_parts.append(part)
        weight = max(0.25, 1.0 - 0.25 * (latest - season))
        weight_parts.append(np.full(len(part), weight, dtype="float32"))
    train = pd.concat(train_parts, ignore_index=True)
    valid = pd.read_parquet(CACHE_DIR / f"features_{valid_season}.parquet")
    weights = np.concatenate(weight_parts)
    y_train = train.pop(TARGET).to_numpy(dtype="int8")
    train.pop("row_id")
    y_valid = valid[TARGET].to_numpy(dtype="int8")
    meta = valid[["row_id", TARGET]].copy()
    valid = valid.drop(columns=["row_id", TARGET])
    if list(train.columns) != schema or list(valid.columns) != schema:
        raise ValueError("Cached feature order does not match schema")
    print(
        f"fold={valid_season} train={len(train):,} {train_seasons} "
        f"valid={len(valid):,}",
        flush=True,
    )
    return train, y_train, weights, valid, y_valid, meta


def fit_cat(
    model_name: str,
    seed: int,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    weights: np.ndarray,
    x_valid: pd.DataFrame,
    cat_idx: list[int],
) -> np.ndarray:
    params = dict(CAT_PARAMS[model_name])
    params.update(
        {
            "loss_function": "Logloss",
            "eval_metric": "Logloss",
            "random_seed": seed,
            "bootstrap_type": "MVS",
            "subsample": 0.8,
            "thread_count": 10,
            "verbose": 100,
            "allow_writing_files": False,
        }
    )
    model = CatBoostClassifier(**params)
    train_pool = Pool(
        x_train, label=y_train, weight=weights, cat_features=cat_idx
    )
    valid_pool = Pool(x_valid, cat_features=cat_idx)
    model.fit(train_pool)
    return model.predict_proba(valid_pool)[:, 1]


def fit_lgb(
    seed: int,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    weights: np.ndarray,
    x_valid: pd.DataFrame,
    cat_idx: list[int],
) -> np.ndarray:
    dataset = lgb.Dataset(
        x_train,
        label=y_train,
        weight=weights,
        categorical_feature=cat_idx,
        free_raw_data=False,
    )
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.01,
        "num_leaves": 31,
        "min_data_in_leaf": 1500,
        "bagging_fraction": 0.7,
        "bagging_freq": 1,
        "feature_fraction": 0.7,
        "lambda_l2": 20.0,
        "max_bin": 127,
        "seed": seed,
        "bagging_seed": seed,
        "feature_fraction_seed": seed,
        "data_random_seed": seed,
        "num_threads": 10,
        "verbosity": -1,
    }
    model = lgb.train(
        params,
        dataset,
        num_boost_round=600,
        callbacks=[lgb.log_evaluation(period=100)],
    )
    return model.predict(x_valid, num_iteration=600)


def prediction_path(valid_season: int, model_name: str, seed: int) -> Path:
    return PRED_DIR / f"{valid_season}_{model_name}_seed{seed}.npy"


def train_oof(models: list[str], seeds: list[int], force: bool = False) -> None:
    raw = pd.read_csv(RAW_PATH, encoding="utf-8-sig")
    schema = ensure_feature_cache(raw)
    cat_idx = [schema.index(c) for c in CAT_COLS]
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    del raw
    gc.collect()

    for valid_season in VALID_SEASONS:
        pending = [
            (model_name, seed)
            for model_name in models
            for seed in seeds
            if force or not prediction_path(valid_season, model_name, seed).exists()
        ]
        if not pending:
            print(f"all predictions cached fold={valid_season}", flush=True)
            continue
        x_train, y_train, weights, x_valid, y_valid, _ = load_fold(
            valid_season, schema
        )
        for model_name, seed in pending:
            started = time.monotonic()
            print(
                f"training fold={valid_season} model={model_name} seed={seed}",
                flush=True,
            )
            if model_name == "lgb":
                prediction = fit_lgb(
                    seed, x_train, y_train, weights, x_valid, cat_idx
                )
            else:
                prediction = fit_cat(
                    model_name,
                    seed,
                    x_train,
                    y_train,
                    weights,
                    x_valid,
                    cat_idx,
                )
            np.save(
                prediction_path(valid_season, model_name, seed),
                prediction.astype("float32"),
            )
            print(
                f"saved fold={valid_season} model={model_name} seed={seed} "
                f"brier={brier(y_valid, prediction):.12f} "
                f"seconds={time.monotonic() - started:.1f}",
                flush=True,
            )
        del x_train, y_train, weights, x_valid, y_valid
        gc.collect()


def consolidate(models: list[str], seeds: list[int]) -> dict[str, object]:
    schema = json.loads((CACHE_DIR / "schema.json").read_text(encoding="utf-8"))
    del schema
    frames = []
    report: dict[str, object] = {"models": {}, "folds": {}}
    for valid_season in VALID_SEASONS:
        cached = pd.read_parquet(
            CACHE_DIR / f"features_{valid_season}.parquet",
            columns=["row_id", TARGET],
        )
        logits = []
        for model_name in models:
            for seed in seeds:
                path = prediction_path(valid_season, model_name, seed)
                if not path.exists():
                    raise FileNotFoundError(path)
                pred = np.load(path)
                name = f"p_mh_{model_name}_seed{seed}"
                cached[name] = pred
                logits.append(logit(pred))
        raw_blend = sigmoid(np.mean(logits, axis=0))
        cached["p_mh_raw_blend"] = raw_blend.astype("float32")
        cached["p_mh_shift_m005"] = sigmoid(logit(raw_blend) - 0.05).astype("float32")
        report["folds"][str(valid_season)] = {
            "rows": len(cached),
            "raw_blend_brier": brier(cached[TARGET].to_numpy(), raw_blend),
            "shift_m005_brier": brier(
                cached[TARGET].to_numpy(), cached["p_mh_shift_m005"].to_numpy()
            ),
        }
        frames.append(cached)
    oof = pd.concat(frames, ignore_index=True)
    target = oof[TARGET].to_numpy(dtype=float)
    pred_cols = [c for c in oof if c.startswith("p_mh_")]
    report["models"] = {
        col: {"pooled_brier": brier(target, oof[col].to_numpy(dtype=float))}
        for col in pred_cols
    }
    shifts = np.arange(-0.15, 0.151, 0.01)
    base_logit = logit(oof["p_mh_raw_blend"].to_numpy())
    shift_rows = [
        {"shift": float(shift), "brier": brier(target, sigmoid(base_logit + shift))}
        for shift in shifts
    ]
    report["shift_screen"] = shift_rows
    report["best_exploratory_shift"] = min(shift_rows, key=lambda row: row["brier"])
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(ARTIFACT_DIR / "mh_oof.parquet", index=False, compression="zstd")
    (ARTIFACT_DIR / "mh_results.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        default="cat_a,cat_b,cat_c,lgb",
        help="Comma-separated subset of cat_a,cat_b,cat_c,lgb",
    )
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = [name.strip() for name in args.models.split(",") if name.strip()]
    seeds = [int(seed) for seed in args.seeds.split(",") if seed.strip()]
    unknown = sorted(set(models) - {"cat_a", "cat_b", "cat_c", "lgb"})
    if unknown:
        raise ValueError(f"Unknown models: {unknown}")
    if args.cache_only:
        raw = pd.read_csv(RAW_PATH, encoding="utf-8-sig")
        ensure_feature_cache(raw, force=args.force)
        return
    train_oof(models, seeds, force=args.force)
    print(json.dumps(consolidate(models, seeds), indent=2), flush=True)


if __name__ == "__main__":
    main()
