from __future__ import annotations

import argparse
import json
import math
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
    brier,
    load_fold,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def encode_categories(
    train: pd.DataFrame, valid: pd.DataFrame, columns: list[str]
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    train_arrays = []
    valid_arrays = []
    cardinalities = []
    for column in columns:
        categories = np.sort(train[column].dropna().unique())
        lookup = {value: index for index, value in enumerate(categories)}
        unknown = len(categories)
        train_arrays.append(
            train[column].map(lookup).fillna(unknown).to_numpy(dtype="int64")
        )
        valid_arrays.append(
            valid[column].map(lookup).fillna(unknown).to_numpy(dtype="int64")
        )
        cardinalities.append(unknown + 1)
    return (
        np.column_stack(train_arrays),
        np.column_stack(valid_arrays),
        cardinalities,
    )


def encode_numeric(
    train: pd.DataFrame, valid: pd.DataFrame, columns: list[str]
) -> tuple[np.ndarray, np.ndarray, dict[str, list[float]]]:
    train_values = train[columns].to_numpy(dtype="float32")
    valid_values = valid[columns].to_numpy(dtype="float32")
    center = np.nanmedian(train_values, axis=0).astype("float32")
    train_values = np.where(np.isfinite(train_values), train_values, center)
    valid_values = np.where(np.isfinite(valid_values), valid_values, center)
    mean = train_values.mean(axis=0, dtype="float64").astype("float32")
    scale = train_values.std(axis=0, dtype="float64").astype("float32")
    scale = np.where(scale > 1e-6, scale, 1.0).astype("float32")
    train_values = ((train_values - mean) / scale).astype("float32")
    valid_values = ((valid_values - mean) / scale).astype("float32")
    return train_values, valid_values, {
        "median": center.tolist(),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
    }


@torch.inference_mode()
def predict(
    model: tabm.TabM,
    x_num: np.ndarray,
    x_cat: np.ndarray,
    device: torch.device,
    batch_size: int = 8192,
) -> np.ndarray:
    model.eval()
    outputs = []
    for start in range(0, len(x_num), batch_size):
        stop = min(start + batch_size, len(x_num))
        num = torch.from_numpy(x_num[start:stop]).to(device)
        cat = torch.from_numpy(x_cat[start:stop]).to(device)
        logits = model(num, cat).squeeze(-1)
        outputs.append(torch.sigmoid(logits).mean(dim=1).cpu().numpy())
    return np.concatenate(outputs)


def run(
    valid_season: int = 2024,
    epochs: int = 8,
    batch_size: int = 2048,
    seed: int = 1234,
    arch_type: str = "tabm-mini",
    d_block: int = 256,
    k: int = 16,
) -> dict[str, object]:
    set_seed(seed)
    schema = json.loads((CACHE_DIR / "schema.json").read_text(encoding="utf-8"))
    x_train, y_train, weights, x_valid, y_valid, meta = load_fold(
        valid_season, schema
    )
    regular_train = x_train["game_type"].eq(0).to_numpy()
    regular_valid = x_valid["game_type"].eq(0).to_numpy()
    x_train = x_train.loc[regular_train].reset_index(drop=True)
    y_train = y_train[regular_train]
    weights = weights[regular_train]

    cat_columns = list(CAT_COLS)
    num_columns = [column for column in schema if column not in cat_columns]
    train_num, valid_num, numeric_state = encode_numeric(
        x_train, x_valid, num_columns
    )
    train_cat, valid_cat, cardinalities = encode_categories(
        x_train, x_valid, cat_columns
    )
    del x_train, x_valid

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = tabm.TabM.make(
        n_num_features=len(num_columns),
        cat_cardinalities=cardinalities,
        d_out=1,
        n_blocks=3,
        d_block=d_block,
        dropout=0.1,
        k=k,
        arch_type=arch_type,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.002, weight_decay=0.0003
    )

    best_brier = math.inf
    best_epoch = 0
    best_state = None
    best_prediction = None
    history = []
    for epoch in range(1, epochs + 1):
        started = time.monotonic()
        model.train()
        order = np.random.permutation(len(train_num))
        running_loss = 0.0
        seen = 0
        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size]
            num = torch.from_numpy(train_num[index]).to(device)
            cat = torch.from_numpy(train_cat[index]).to(device)
            target = torch.from_numpy(y_train[index].astype("float32")).to(device)
            weight = torch.from_numpy(weights[index].astype("float32")).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(num, cat).squeeze(-1)
            losses = F.binary_cross_entropy_with_logits(
                logits, target[:, None].expand_as(logits), reduction="none"
            ).mean(dim=1)
            loss = (losses * weight).sum() / weight.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += float(loss.detach().cpu()) * len(index)
            seen += len(index)

        prediction = predict(model, valid_num, valid_cat, device)
        score = brier(y_valid[regular_valid], prediction[regular_valid])
        row = {
            "epoch": epoch,
            "train_loss": running_loss / seen,
            "regular_brier": score,
            "seconds": time.monotonic() - started,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if score < best_brier:
            best_brier = score
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            best_prediction = prediction.astype("float32")

    if best_state is None or best_prediction is None:
        raise RuntimeError("TabM training did not produce a checkpoint")
    output_dir = ARTIFACT_DIR / "tabm"
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{arch_type.replace('-', '_')}_d{d_block}_k{k}"
    torch.save(
        {
            "state_dict": best_state,
            "num_columns": num_columns,
            "cat_columns": cat_columns,
            "cat_cardinalities": cardinalities,
            "numeric_state": numeric_state,
            "valid_season": valid_season,
            "seed": seed,
            "arch_type": arch_type,
            "d_block": d_block,
            "k": k,
        },
        output_dir / f"{tag}_regular_{valid_season}_seed{seed}.pt",
    )
    np.save(
        output_dir / f"prediction_{tag}_{valid_season}_seed{seed}.npy",
        best_prediction,
    )
    result = {
        "valid_season": valid_season,
        "train_rows": len(train_num),
        "valid_regular_rows": int(regular_valid.sum()),
        "device": str(device),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "arch_type": arch_type,
        "d_block": d_block,
        "k": k,
        "best_epoch": best_epoch,
        "best_regular_brier": best_brier,
        "history": history,
    }
    (output_dir / f"result_{tag}_{valid_season}_seed{seed}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-season", type=int, default=2024)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--arch-type", choices=["tabm", "tabm-mini", "tabm-packed"], default="tabm-mini"
    )
    parser.add_argument("--d-block", type=int, default=256)
    parser.add_argument("--k", type=int, default=16)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.valid_season,
                args.epochs,
                args.batch_size,
                args.seed,
                args.arch_type,
                args.d_block,
                args.k,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
