from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from challengers.v14_mh_profile_ensemble.temporal_train import (
    ARTIFACT_DIR,
    CACHE_DIR,
    brier,
)


ROOT = Path(__file__).resolve().parents[2]
V13_PATH = ROOT / "challengers/v11_feature_rebuild/artifacts/main_screen_oof.parquet"
VALID_SEASONS = (2022, 2023, 2024)
TARGET = "control_success"
BASE = "p_v13_centered_expert_best"
ROBUST = "p_v14_uncalibrated"
FINAL = "p_v14_month_calibrated"
TARGET_BRIER_1500 = 0.2462267860881907


def cluster_bootstrap(
    frame: pd.DataFrame,
    pred_a: str,
    pred_b: str,
    repeats: int,
    seed: int,
) -> dict[str, float | int | str]:
    cluster_col = "pitcher_id" if "pitcher_id" in frame.columns else "row_id"
    target = frame[TARGET].to_numpy(dtype="float64")
    work = pd.DataFrame(
        {
            "cluster": frame[cluster_col].astype("string").to_numpy(),
            "err_a": (np.clip(frame[pred_a], 1e-6, 1.0 - 1e-6) - target) ** 2,
            "err_b": (np.clip(frame[pred_b], 1e-6, 1.0 - 1e-6) - target) ** 2,
        }
    )
    grouped = work.groupby("cluster", observed=True).agg(
        err_a=("err_a", "sum"),
        err_b=("err_b", "sum"),
        n=("err_a", "size"),
    )
    rng = np.random.default_rng(seed)
    err_a = grouped["err_a"].to_numpy(dtype="float64")
    err_b = grouped["err_b"].to_numpy(dtype="float64")
    counts = grouped["n"].to_numpy(dtype="float64")
    cluster_indices = np.arange(len(grouped))
    differences = np.empty(repeats, dtype="float64")
    for index in range(repeats):
        sample = rng.choice(cluster_indices, size=len(cluster_indices), replace=True)
        differences[index] = (
            err_a[sample].sum() / counts[sample].sum()
            - err_b[sample].sum() / counts[sample].sum()
        )
    return {
        "cluster_col": cluster_col,
        "repeats": repeats,
        "observed_delta": brier(target, frame[pred_a]) - brier(target, frame[pred_b]),
        "ci_low": float(np.quantile(differences, 0.025)),
        "ci_high": float(np.quantile(differences, 0.975)),
        "p_delta_lt_0": float(np.mean(differences < 0.0)),
    }


def load_frame() -> pd.DataFrame:
    mh = pd.read_parquet(
        ARTIFACT_DIR / "mh_oof.parquet",
        columns=["row_id", TARGET, "p_mh_raw_blend"],
    )
    v13 = pd.read_parquet(V13_PATH, columns=["row_id", BASE])
    raw = pd.read_csv(
        ROOT / "data/raw/train.csv",
        usecols=["row_id", "season", "game_type", "game_month", "pitcher_id"],
        encoding="utf-8-sig",
    )
    tabm_parts = []
    for season in VALID_SEASONS:
        part = pd.read_parquet(
            CACHE_DIR / f"features_{season}.parquet", columns=["row_id"]
        )
        part["p_tabm_seed1234"] = np.load(
            ARTIFACT_DIR / f"tabm/prediction_{season}_seed1234.npy"
        )
        part["p_tabm_seed2345"] = np.load(
            ARTIFACT_DIR
            / f"tabm/prediction_tabm_mini_d256_k16_{season}_seed2345.npy"
        )
        tabm_parts.append(part)
    return (
        mh.merge(v13, on="row_id", validate="one_to_one")
        .merge(raw, on="row_id", validate="one_to_one")
        .merge(pd.concat(tabm_parts, ignore_index=True), on="row_id", validate="one_to_one")
    )


def fit_directions(frame: pd.DataFrame) -> np.ndarray:
    target = frame[TARGET].to_numpy(dtype="float64")
    base = frame[BASE].to_numpy(dtype="float64")
    regular = frame["game_type"].eq("R").to_numpy(dtype="float64")
    matrix = np.column_stack(
        [
            (frame[column].to_numpy(dtype="float64") - base) * regular
            for column in (
                "p_mh_raw_blend",
                "p_tabm_seed1234",
                "p_tabm_seed2345",
            )
        ]
    )
    return np.linalg.solve(matrix.T @ matrix, matrix.T @ (target - base))


def apply_directions(frame: pd.DataFrame, weights: np.ndarray) -> np.ndarray:
    base = frame[BASE].to_numpy(dtype="float64")
    regular = frame["game_type"].eq("R").to_numpy(dtype="float64")
    matrix = np.column_stack(
        [
            (frame[column].to_numpy(dtype="float64") - base) * regular
            for column in (
                "p_mh_raw_blend",
                "p_tabm_seed1234",
                "p_tabm_seed2345",
            )
        ]
    )
    return np.clip(base + matrix @ weights, 1e-6, 1.0 - 1e-6)


def fit_month_calibration(
    frame: pd.DataFrame, prediction: np.ndarray, train_mask: np.ndarray
) -> dict[str, dict[str, float | int]]:
    target = frame[TARGET].to_numpy(dtype="float64")
    groups = frame["game_type"].astype(str) + "_" + frame["game_month"].astype(str)
    policy: dict[str, dict[str, float | int]] = {}
    for group in sorted(groups[train_mask].unique()):
        mask = train_mask & groups.eq(group).to_numpy()
        matrix = np.column_stack(
            [np.ones(int(mask.sum())), prediction[mask] - 0.5]
        )
        coef = np.linalg.lstsq(
            matrix, target[mask] - prediction[mask], rcond=None
        )[0]
        policy[group] = {
            "intercept": float(coef[0]),
            "slope_delta": float(coef[1]),
            "rows": int(mask.sum()),
        }
    return policy


def apply_month_calibration(
    frame: pd.DataFrame,
    prediction: np.ndarray,
    policy: dict[str, dict[str, float | int]],
) -> np.ndarray:
    output = prediction.copy()
    groups = frame["game_type"].astype(str) + "_" + frame["game_month"].astype(str)
    for group, params in policy.items():
        mask = groups.eq(group).to_numpy()
        output[mask] += float(params["intercept"]) + float(
            params["slope_delta"]
        ) * (prediction[mask] - 0.5)
    return np.clip(output, 1e-6, 1.0 - 1e-6)


def metric_block(frame: pd.DataFrame, column: str) -> dict[str, object]:
    target = frame[TARGET].to_numpy(dtype="float64")
    prediction = frame[column].to_numpy(dtype="float64")
    seasons = frame["season"].to_numpy(dtype="int16")
    score = brier(target, prediction)
    return {
        "brier": score,
        "gap_to_1500": score - TARGET_BRIER_1500,
        "folds": {
            str(season): {
                "rows": int((seasons == season).sum()),
                "brier": brier(
                    target[seasons == season], prediction[seasons == season]
                ),
                "prediction_mean": float(prediction[seasons == season].mean()),
                "target_mean": float(target[seasons == season].mean()),
            }
            for season in VALID_SEASONS
        },
    }


def nested_diagnostics(frame: pd.DataFrame) -> dict[str, object]:
    seasons = frame["season"].to_numpy(dtype="int16")
    target = frame[TARGET].to_numpy(dtype="float64")
    prediction = frame[BASE].to_numpy(dtype="float64").copy()
    rows = []
    for season in (2023, 2024):
        train = (seasons >= 2022) & (seasons < season)
        valid = seasons == season
        train_frame = frame.loc[train]
        weights = fit_directions(train_frame)
        robust = apply_directions(frame.loc[valid], weights)
        policy = fit_month_calibration(
            train_frame, apply_directions(train_frame, weights), np.ones(int(train.sum()), dtype=bool)
        )
        calibrated = apply_month_calibration(frame.loc[valid], robust, policy)
        prediction[valid] = calibrated
        rows.append(
            {
                "valid_season": season,
                "direction_weights": weights.tolist(),
                "brier": brier(target[valid], calibrated),
                "delta_vs_v13": brier(target[valid], calibrated)
                - brier(target[valid], frame.loc[valid, BASE].to_numpy()),
            }
        )
    return {
        "protocol": "2023<-2022; 2024<-2022+2023; 2022 remains V13",
        "pooled_brier": brier(target, prediction),
        "folds": rows,
    }


def run() -> dict[str, object]:
    frame = load_frame()
    weights = fit_directions(frame)
    frame[ROBUST] = apply_directions(frame, weights).astype("float32")
    all_rows = np.ones(len(frame), dtype=bool)
    calibration = fit_month_calibration(
        frame, frame[ROBUST].to_numpy(dtype="float64"), all_rows
    )
    frame[FINAL] = apply_month_calibration(
        frame, frame[ROBUST].to_numpy(dtype="float64"), calibration
    ).astype("float32")

    metrics = {
        BASE: metric_block(frame, BASE),
        ROBUST: metric_block(frame, ROBUST),
        FINAL: metric_block(frame, FINAL),
    }
    target = frame[TARGET].to_numpy(dtype="float64")
    climatology = brier(target, np.full(len(frame), target.mean()))
    for values in metrics.values():
        values["standard_bss"] = 1.0 - float(values["brier"]) / climatology
        values["official_like_score"] = max(
            0.0, 100000.0 * float(values["standard_bss"])
        )
    fold_deltas = {
        str(season): metrics[FINAL]["folds"][str(season)]["brier"]
        - metrics[BASE]["folds"][str(season)]["brier"]
        for season in VALID_SEASONS
    }
    bootstrap = {
        "final_vs_v13": cluster_bootstrap(
            frame, FINAL, BASE, repeats=3000, seed=14100
        ),
        "robust_vs_v13": cluster_bootstrap(
            frame, ROBUST, BASE, repeats=3000, seed=14101
        ),
    }
    nested = nested_diagnostics(frame)
    result = {
        "candidate": FINAL,
        "formula": (
            "V13 + R-only weighted directions from MH 12-model-compatible "
            "profile ensemble and two TabM seeds, followed by fixed "
            "game_type x month affine calibration"
        ),
        "target_brier_1500": TARGET_BRIER_1500,
        "direction_order": [
            "p_mh_raw_blend_minus_v13",
            "p_tabm_seed1234_minus_v13",
            "p_tabm_seed2345_minus_v13",
        ],
        "direction_weights": weights.tolist(),
        "month_calibration": calibration,
        "metrics": metrics,
        "fold_deltas_vs_v13": fold_deltas,
        "bootstrap": bootstrap,
        "nested_diagnostics": nested,
        "completion_audit": {
            "local_1500_reached_exploratory": metrics[FINAL]["brier"]
            <= TARGET_BRIER_1500,
            "all_folds_improve_vs_v13": max(fold_deltas.values()) < 0.0,
            "pitcher_bootstrap_ci_below_zero": bootstrap["final_vs_v13"][
                "ci_high"
            ]
            < 0.0,
            "nested_calibration_improves": nested["pooled_brier"]
            < metrics[BASE]["brier"],
            "official_1500_verified": False,
        },
        "warning": (
            "The local 1500 crossing uses calibration fitted on pooled held-out "
            "predictions. Its next-season nested diagnostic is reported separately "
            "and is weaker; official score is not yet available."
        ),
    }
    frame[[
        "row_id",
        "pitcher_id",
        "season",
        TARGET,
        BASE,
        "p_mh_raw_blend",
        "p_tabm_seed1234",
        "p_tabm_seed2345",
        ROBUST,
        FINAL,
    ]].to_parquet(ARTIFACT_DIR / "v14_oof.parquet", index=False)
    (ARTIFACT_DIR / "v14_policy.json").write_text(
        json.dumps(
            {
                "direction_order": result["direction_order"],
                "direction_weights": result["direction_weights"],
                "month_calibration": calibration,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (ARTIFACT_DIR / "v14_results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
