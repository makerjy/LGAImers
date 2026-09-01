from __future__ import annotations

import gc
import json
import random
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import tabm
import torch
import torch.nn.functional as F

from challengers.v14_mh_profile_ensemble.temporal_train import (
    ARTIFACT_DIR,
    CACHE_DIR,
    CAT_COLS,
)


TRAIN_SEASONS = (2021, 2022, 2023, 2024)
SEEDS = (1234, 2345)
EPOCHS = 3
BATCH_SIZE = 2048


def train_seed(
    train_num: np.ndarray,
    train_cat: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    cardinalities: list[int],
    seed: int,
) -> tuple[dict[str, torch.Tensor], list[dict[str, float | int]]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = tabm.TabM.make(
        n_num_features=train_num.shape[1],
        cat_cardinalities=cardinalities,
        d_out=1,
        n_blocks=3,
        d_block=256,
        dropout=0.1,
        k=16,
        arch_type="tabm-mini",
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.002, weight_decay=0.0003
    )
    history = []
    for epoch in range(1, EPOCHS + 1):
        started = time.monotonic()
        model.train()
        order = np.random.permutation(len(train_num))
        total = 0.0
        seen = 0
        for start in range(0, len(order), BATCH_SIZE):
            index = order[start : start + BATCH_SIZE]
            num = torch.from_numpy(train_num[index]).to(device)
            cat = torch.from_numpy(train_cat[index]).to(device)
            y = torch.from_numpy(target[index]).to(device)
            weight = torch.from_numpy(weights[index]).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(num, cat).squeeze(-1)
            losses = F.binary_cross_entropy_with_logits(
                logits, y[:, None].expand_as(logits), reduction="none"
            ).mean(dim=1)
            loss = (losses * weight).sum() / weight.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach().cpu()) * len(index)
            seen += len(index)
        row = {
            "epoch": epoch,
            "train_loss": total / seen,
            "seconds": time.monotonic() - started,
        }
        history.append(row)
        print(f"seed={seed} {json.dumps(row)}", flush=True)
    state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    del model, optimizer
    if device.type == "mps":
        torch.mps.empty_cache()
    gc.collect()
    return state, history


def run() -> dict[str, object]:
    schema = json.loads((CACHE_DIR / "schema.json").read_text(encoding="utf-8"))
    parts = []
    weight_parts = []
    for season in TRAIN_SEASONS:
        part = pd.read_parquet(CACHE_DIR / f"features_{season}.parquet")
        regular = part["game_type"].eq(0).to_numpy()
        part = part.loc[regular].reset_index(drop=True)
        parts.append(part)
        weight = 0.25 * (season - 2020)
        weight_parts.append(np.full(len(part), weight, dtype="float32"))
    frame = pd.concat(parts, ignore_index=True)
    weights = np.concatenate(weight_parts)
    target = frame.pop("control_success").to_numpy(dtype="float32")
    frame.pop("row_id")

    cat_columns = list(CAT_COLS)
    num_columns = [column for column in schema if column not in cat_columns]
    num = frame[num_columns].to_numpy(dtype="float32")
    median = np.nanmedian(num, axis=0).astype("float32")
    num = np.where(np.isfinite(num), num, median)
    mean = num.mean(axis=0, dtype="float64").astype("float32")
    scale = num.std(axis=0, dtype="float64").astype("float32")
    scale = np.where(scale > 1e-6, scale, 1.0).astype("float32")
    num = ((num - mean) / scale).astype("float32")

    cat_arrays = []
    category_values: dict[str, list[int | float | str]] = {}
    cardinalities = []
    for column in cat_columns:
        values = sorted(frame[column].dropna().unique().tolist())
        mapping = {value: index for index, value in enumerate(values)}
        unknown = len(values)
        cat_arrays.append(
            frame[column].map(mapping).fillna(unknown).to_numpy(dtype="int64")
        )
        category_values[column] = values
        cardinalities.append(unknown + 1)
    cat = np.column_stack(cat_arrays)
    del frame, parts
    gc.collect()

    output_dir = ARTIFACT_DIR / "tabm/full"
    output_dir.mkdir(parents=True, exist_ok=True)
    histories = {}
    for seed in SEEDS:
        state, history = train_seed(
            num, cat, target, weights, cardinalities, seed
        )
        checkpoint = {
            "state_dict": state,
            "num_columns": num_columns,
            "cat_columns": cat_columns,
            "category_values": category_values,
            "cat_cardinalities": cardinalities,
            "numeric_median": median.tolist(),
            "numeric_mean": mean.tolist(),
            "numeric_scale": scale.tolist(),
            "model": {
                "n_num_features": len(num_columns),
                "cat_cardinalities": cardinalities,
                "d_out": 1,
                "n_blocks": 3,
                "d_block": 256,
                "dropout": 0.1,
                "k": 16,
                "arch_type": "tabm-mini",
            },
            "seed": seed,
            "epochs": EPOCHS,
            "train_seasons": list(TRAIN_SEASONS),
            "train_rows": len(num),
        }
        torch.save(checkpoint, output_dir / f"tabm_seed{seed}.pt")
        histories[str(seed)] = history
    result = {
        "train_rows": len(num),
        "train_seasons": list(TRAIN_SEASONS),
        "all_official_rows_used_for_profiles": True,
        "regular_only_model_rows": True,
        "num_features": len(num_columns),
        "categorical_features": len(cat_columns),
        "seeds": list(SEEDS),
        "epochs": EPOCHS,
        "histories": histories,
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
