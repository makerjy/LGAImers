"""Build row-independent entity profiles from seasons strictly before a cutoff."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


TARGET = "control_success"

PITCHER_COLS = [
    "pp_succ_c", "pp_n", "pp_cum_n", "pp_cum_succ", "pp_conf",
    "pp_n_seasons", "pp_league", "pp_vs_L", "pp_vs_R", "pp_ahead",
    "pp_behind", "pp_2strike", "pp_3ball", "pp_highli", "pp_early",
    "pp_late", "pp_r_succ", "pp_f_succ", "pp_last_season",
    "pp_p_futures", "pp_p_inn1", "pp_p_inn8", "pp_mean_inning",
    "pp_inning_std", "pp_mean_li", "pp_p_3ball", "pp_p_2strike",
    "pp_mean_balls", "pp_mean_strikes",
]

BATTER_COLS = [
    "bb_succ_c", "bb_n", "bb_cum_n", "bb_cum_succ", "bb_conf",
    "bb_league", "bb_vs_L", "bb_vs_R", "bb_last_season", "bb_platoon",
    "bb_p_futures", "bb_p_3ball", "bb_p_2strike", "bb_mean_li",
    "bb_mean_inning",
]

SHRINK_RATE_COLS = [
    "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate", "asof_batter_success_rate",
    "asof_batter_middle_rate", "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
]


def compute_shrink_priors(history: pd.DataFrame) -> dict[str, float]:
    return {
        col: float(pd.to_numeric(history[col], errors="coerce").mean())
        for col in SHRINK_RATE_COLS
    }


def _effect_table(
    data: pd.DataFrame,
    entity: str,
    mask: pd.Series | np.ndarray,
    name: str,
    *,
    k: float = 250.0,
) -> pd.Series:
    subset = data.loc[np.asarray(mask, dtype=bool), [entity, TARGET]]
    if subset.empty:
        return pd.Series(dtype="float64", name=name)
    prior = float(subset[TARGET].mean())
    agg = subset.groupby(entity, sort=True)[TARGET].agg(["sum", "count"])
    effect = (agg["sum"] + prior * k) / (agg["count"] + k) - prior
    effect.name = name
    return effect


def _last_season_effect(
    data: pd.DataFrame,
    entity: str,
    target_season: int,
    name: str,
) -> pd.Series:
    mask = data["season"].eq(target_season - 1)
    return _effect_table(data, entity, mask, name, k=150.0)


def _base_entity(
    data: pd.DataFrame,
    entity: str,
    prefix: str,
    league: float,
) -> pd.DataFrame:
    agg = data.groupby(entity, sort=True)[TARGET].agg(["sum", "count"])
    out = pd.DataFrame(index=agg.index)
    out[f"{prefix}_succ_c"] = (
        (agg["sum"] + league * 250.0) / (agg["count"] + 250.0) - league
    )
    out[f"{prefix}_n"] = agg["count"].astype("float64")
    out[f"{prefix}_cum_n"] = agg["count"].astype("float64")
    out[f"{prefix}_cum_succ"] = agg["sum"].astype("float64")
    out[f"{prefix}_conf"] = agg["count"] / (agg["count"] + 1000.0)
    out[f"{prefix}_league"] = league
    return out


def _build_pitcher(data: pd.DataFrame, target_season: int, league: float) -> pd.DataFrame:
    key = "pitcher_id"
    out = _base_entity(data, key, "pp", league)
    out["pp_n_seasons"] = data.groupby(key, sort=True)["season"].nunique()

    splits = {
        "pp_vs_L": data["batter_hand"].eq(1),
        "pp_vs_R": data["batter_hand"].eq(2),
        "pp_ahead": data["strikes_before"].gt(data["balls_before"]),
        "pp_behind": data["balls_before"].gt(data["strikes_before"]),
        "pp_2strike": data["strikes_before"].eq(2),
        "pp_3ball": data["balls_before"].eq(3),
        "pp_highli": data["li"].ge(1.5),
        "pp_early": data["inning"].le(3),
        "pp_late": data["inning"].ge(7),
        "pp_r_succ": data["game_type"].eq("R"),
        "pp_f_succ": data["game_type"].eq("F"),
    }
    for name, mask in splits.items():
        out[name] = _effect_table(data, key, mask, name)
    out["pp_last_season"] = _last_season_effect(data, key, target_season, "pp_last_season")

    grouped = data.groupby(key, sort=True)
    out["pp_p_futures"] = grouped["game_type"].agg(lambda x: x.eq("F").mean())
    out["pp_p_inn1"] = grouped["inning"].agg(lambda x: x.eq(1).mean())
    out["pp_p_inn8"] = grouped["inning"].agg(lambda x: x.ge(8).mean())
    out["pp_mean_inning"] = grouped["inning"].mean()
    out["pp_inning_std"] = grouped["inning"].std()
    out["pp_mean_li"] = grouped["li"].mean()
    out["pp_p_3ball"] = grouped["balls_before"].agg(lambda x: x.eq(3).mean())
    out["pp_p_2strike"] = grouped["strikes_before"].agg(lambda x: x.eq(2).mean())
    out["pp_mean_balls"] = grouped["balls_before"].mean()
    out["pp_mean_strikes"] = grouped["strikes_before"].mean()
    return out[PITCHER_COLS]


def _build_batter(data: pd.DataFrame, target_season: int, league: float) -> pd.DataFrame:
    key = "batter_id"
    out = _base_entity(data, key, "bb", league)
    out["bb_vs_L"] = _effect_table(
        data, key, data["pitcher_hand"].eq(1), "bb_vs_L"
    )
    out["bb_vs_R"] = _effect_table(
        data, key, data["pitcher_hand"].eq(2), "bb_vs_R"
    )
    out["bb_last_season"] = _last_season_effect(data, key, target_season, "bb_last_season")
    out["bb_platoon"] = out["bb_vs_L"] - out["bb_vs_R"]

    grouped = data.groupby(key, sort=True)
    out["bb_p_futures"] = grouped["game_type"].agg(lambda x: x.eq("F").mean())
    out["bb_p_3ball"] = grouped["balls_before"].agg(lambda x: x.eq(3).mean())
    out["bb_p_2strike"] = grouped["strikes_before"].agg(lambda x: x.eq(2).mean())
    out["bb_mean_li"] = grouped["li"].mean()
    out["bb_mean_inning"] = grouped["inning"].mean()
    return out[BATTER_COLS]


def _build_team(data: pd.DataFrame, key: str, name: str, league: float) -> pd.DataFrame:
    agg = data.groupby(key, sort=True)[TARGET].agg(["sum", "count"])
    values = (agg["sum"] + league * 400.0) / (agg["count"] + 400.0) - league
    return values.rename(name).to_frame()


def _numpy_table(frame: pd.DataFrame) -> dict[str, object]:
    frame = frame.sort_index()
    return {
        "ids": frame.index.to_numpy(dtype="int64"),
        "cols": list(frame.columns),
        "values": frame.to_numpy(dtype="float32"),
    }


def build_profiles(
    train: pd.DataFrame,
    target_season: int,
    source_seasons: Iterable[int] | None = None,
) -> dict[str, object]:
    """Build profiles using only rows from seasons before ``target_season``."""
    if source_seasons is None:
        source_seasons = sorted(
            int(s) for s in train.loc[train["season"].lt(target_season), "season"].unique()
        )
    source_seasons = sorted(int(s) for s in source_seasons if int(s) < target_season)
    history = train.loc[train["season"].isin(source_seasons)].copy()
    if history.empty:
        empty = {"ids": np.array([], dtype="int64"), "cols": [], "values": np.empty((0, 0), dtype="float32")}
        return {
            "pitcher": empty,
            "batter": empty.copy(),
            "pteam": empty.copy(),
            "bteam": empty.copy(),
            "league": float(train[TARGET].mean()),
            "source_seasons": source_seasons,
        }

    league = float(history[TARGET].mean())
    return {
        "pitcher": _numpy_table(_build_pitcher(history, target_season, league)),
        "batter": _numpy_table(_build_batter(history, target_season, league)),
        "pteam": _numpy_table(_build_team(history, "pitcher_team_id", "pt_succ_c", league)),
        "bteam": _numpy_table(_build_team(history, "batter_team_id", "bt_succ_c", league)),
        "league": league,
        "source_seasons": source_seasons,
    }


def compare_profile_tables(
    actual: dict[str, object], expected: dict[str, object]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for table_name in ("pitcher", "batter", "pteam", "bteam"):
        left = actual[table_name]
        right = expected[table_name]
        ids_equal = np.array_equal(left["ids"], right["ids"])
        for col in right["cols"]:
            if col not in left["cols"] or not ids_equal:
                rows.append({"table": table_name, "column": col, "max_abs_diff": np.inf})
                continue
            li = left["cols"].index(col)
            ri = right["cols"].index(col)
            a = np.asarray(left["values"])[:, li]
            b = np.asarray(right["values"])[:, ri]
            diff = np.abs(a.astype("float64") - b.astype("float64"))
            both_nan = np.isnan(a) & np.isnan(b)
            mismatch_nan = np.isnan(a) ^ np.isnan(b)
            value = np.inf if mismatch_nan.any() else float(np.nanmax(np.where(both_nan, np.nan, diff), initial=0.0))
            rows.append({"table": table_name, "column": col, "max_abs_diff": value})
    return pd.DataFrame(rows)
