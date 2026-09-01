from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from challengers.v14_mh_profile_ensemble.temporal_train import (
    ARTIFACT_DIR,
    CACHE_DIR,
    CAT_COLS,
    VALID_SEASONS,
    brier,
    fit_cat,
    fit_lgb,
    load_fold,
    logit,
    sigmoid,
)


PRED_DIR = ARTIFACT_DIR / "regular_predictions"


def pred_path(season: int, model: str, seed: int) -> Path:
    return PRED_DIR / f"{season}_{model}_seed{seed}.npy"


def train(models: list[str], seeds: list[int], force: bool = False) -> None:
    schema = json.loads((CACHE_DIR / "schema.json").read_text(encoding="utf-8"))
    cat_idx = [schema.index(c) for c in CAT_COLS]
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    for season in VALID_SEASONS:
        pending = [
            (model, seed)
            for model in models
            for seed in seeds
            if force or not pred_path(season, model, seed).exists()
        ]
        if not pending:
            continue
        x_train, y_train, weights, x_valid, y_valid, _ = load_fold(season, schema)
        regular_train = x_train["game_type"].eq(0).to_numpy()
        regular_valid = x_valid["game_type"].eq(0).to_numpy()
        print(
            f"regular fold={season} train={regular_train.sum():,} "
            f"valid={regular_valid.sum():,}",
            flush=True,
        )
        for model, seed in pending:
            started = time.monotonic()
            if model == "lgb":
                pred = fit_lgb(
                    seed,
                    x_train.loc[regular_train],
                    y_train[regular_train],
                    weights[regular_train],
                    x_valid,
                    cat_idx,
                )
            else:
                pred = fit_cat(
                    model,
                    seed,
                    x_train.loc[regular_train],
                    y_train[regular_train],
                    weights[regular_train],
                    x_valid,
                    cat_idx,
                )
            np.save(pred_path(season, model, seed), pred.astype("float32"))
            print(
                f"saved regular fold={season} model={model} seed={seed} "
                f"R_brier={brier(y_valid[regular_valid], pred[regular_valid]):.12f} "
                f"seconds={time.monotonic()-started:.1f}",
                flush=True,
            )
        del x_train, y_train, weights, x_valid, y_valid
        gc.collect()


def consolidate(models: list[str], seeds: list[int]) -> dict[str, object]:
    parts = []
    report: dict[str, object] = {"folds": {}}
    for season in VALID_SEASONS:
        meta = pd.read_parquet(
            CACHE_DIR / f"features_{season}.parquet",
            columns=["row_id", "control_success", "game_type"],
        )
        logits = []
        for model in models:
            for seed in seeds:
                pred = np.load(pred_path(season, model, seed))
                meta[f"p_regular_{model}_seed{seed}"] = pred
                logits.append(logit(pred))
        meta["p_regular_blend"] = sigmoid(np.mean(logits, axis=0)).astype("float32")
        regular = meta["game_type"].eq(0).to_numpy()
        report["folds"][str(season)] = {
            "rows": int(regular.sum()),
            "blend_brier": brier(
                meta.loc[regular, "control_success"].to_numpy(),
                meta.loc[regular, "p_regular_blend"].to_numpy(),
            ),
        }
        parts.append(meta)
    oof = pd.concat(parts, ignore_index=True)
    oof.to_parquet(ARTIFACT_DIR / "regular_oof.parquet", index=False)
    regular = oof["game_type"].eq(0).to_numpy()
    report["pooled_regular_brier"] = brier(
        oof.loc[regular, "control_success"].to_numpy(),
        oof.loc[regular, "p_regular_blend"].to_numpy(),
    )
    (ARTIFACT_DIR / "regular_results.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="lgb")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    models = [value for value in args.models.split(",") if value]
    seeds = [int(value) for value in args.seeds.split(",") if value]
    train(models, seeds, args.force)
    print(json.dumps(consolidate(models, seeds), indent=2))


if __name__ == "__main__":
    main()
