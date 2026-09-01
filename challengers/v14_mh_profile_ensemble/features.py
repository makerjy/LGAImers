"""피처 엔지니어링 — 학습/추론 공용, **행 단위로 완전히 닫혀 있다**.

이 모듈의 모든 연산은 다음만 사용한다:
  1) 예측 대상 행 자신의 입력 변수
  2) 그 행의 값만으로 계산한 파생값
  3) 학습 데이터로 사전 계산되어 인자로 전달된 상수/룩업표 (prof, priors)

행 간 연산(groupby / rolling / lag / 전체평균 / rank)은 존재하지 않는다.
따라서 test.csv에 해당 행 1개만 있든 전체가 있든 결과가 동일하다.
프로파일 생성 등 집계가 필요한 코드는 profiles.py / train_utils.py에 있고
제출 패키지에 포함되지 않는다.
"""
import numpy as np
import pandas as pd

ID_COL = "row_id"
TARGET_COL = "control_success"

CAT_MAPS = {
    "top_bottom": {"T": 0, "B": 1},
    "game_type": {"R": 0, "F": 1},
}
CAT_COLS = ["top_bottom", "game_type", "pitcher_hand", "batter_hand",
            "pitcher_team_id", "batter_team_id"]

# 원본 상황 변수. 의도적 제외:
#   season               2025는 학습에 없어 트리가 외삽 불가
#   pitcher_id/batter_id 신규 선수 비중이 커 cold-start 취약 -> 프로파일로 대체
#   asof_*_n (원시)      연도별 누적 폭증, 2025는 학습 구간 밖 -> saturation으로 대체
#   base_state           runner_on_* + num_runners_on과 중복
#   away_win_expectancy  100 - home_win_expectancy로 완전 중복
#   run_top/bot_before   run_total_before + score_diff로 표현됨
SITUATION = [
    "game_month", "game_dayofweek", "inning",
    "balls_before", "strikes_before", "outs_before",
    "run_total_before", "score_diff_pitcher_team",
    "runner_on_1b", "runner_on_2b", "runner_on_3b", "num_runners_on",
    "home_win_expectancy", "li",
]

# (비율, 표본수) — 학습 prior로 shrinkage
SHRINK_PAIRS = [
    ("asof_pitcher_success_rate", "asof_pitcher_n"),
    ("asof_pitcher_reverse_rate", "asof_pitcher_n"),
    ("asof_pitcher_middle_rate", "asof_pitcher_n"),
    ("asof_pitcher_ball_rate", "asof_pitcher_n"),
    ("asof_pitcher_strike_rate", "asof_pitcher_n"),
    ("asof_batter_success_rate", "asof_batter_n"),
    ("asof_batter_middle_rate", "asof_batter_n"),
    ("asof_pitcher_fastball_rate", "asof_pitcher_pitchmix_n"),
    ("asof_pitcher_breaking_rate", "asof_pitcher_pitchmix_n"),
    ("asof_pitcher_offspeed_rate", "asof_pitcher_pitchmix_n"),
]
SHRINK_NAME = {
    "asof_pitcher_success_rate": "p_succ", "asof_pitcher_reverse_rate": "p_rev",
    "asof_pitcher_middle_rate": "p_mid", "asof_pitcher_ball_rate": "p_ball",
    "asof_pitcher_strike_rate": "p_stk", "asof_batter_success_rate": "b_succ",
    "asof_batter_middle_rate": "b_mid", "asof_pitcher_fastball_rate": "p_fb",
    "asof_pitcher_breaking_rate": "p_br", "asof_pitcher_offspeed_rate": "p_os",
}

PP_LOOKUP_EXTRA = ["pp_cum_n", "pp_cum_succ"]
BB_LOOKUP_EXTRA = ["bb_cum_n", "bb_cum_succ"]

PP_COLS = ["pp_succ_c", "pp_conf", "pp_n_seasons", "pp_league", "pp_vs_L", "pp_vs_R",
           "pp_ahead", "pp_behind", "pp_2strike", "pp_3ball", "pp_highli",
           "pp_early", "pp_late", "pp_r_succ", "pp_f_succ", "pp_last_season",
           "pp_p_futures", "pp_p_inn1", "pp_p_inn8", "pp_mean_inning",
           "pp_inning_std", "pp_mean_li", "pp_p_3ball", "pp_p_2strike",
           "pp_mean_balls", "pp_mean_strikes"]
BB_COLS = ["bb_succ_c", "bb_conf", "bb_league", "bb_vs_L", "bb_vs_R",
           "bb_last_season", "bb_platoon", "bb_p_futures", "bb_p_3ball",
           "bb_p_2strike", "bb_mean_li", "bb_mean_inning"]

DEFAULT_CONFIG = {
    "use_profiles": True,
    "use_shrinkage": True,
    "shrink_k": 150,        # 친구 설계서 실측 최적
    "conf_k": 500.0,        # n/(n+K) saturation
    "keep_season": False,
    "keep_raw_ids": False,
    "shrink_priors": None,  # train_utils.compute_shrink_priors()가 만든 상수
    "use_season_to_date": False,  # 대상 시즌 한정 성적 복원
    "std_min_n": 20,        # 시즌 한정 표본이 이보다 적으면 결측 처리
}


def _num(df, col):
    return pd.to_numeric(df[col], errors="coerce").astype("float64")


def _lookup(df, key_col, table, cols):
    """행별 id -> 학습 프로파일 조회.

    table은 pandas가 아니라 순수 numpy 구조다:
        {"ids": 정렬된 int64 배열, "cols": [컬럼명], "values": float32 2D 배열}
    pandas 버전 간 피클 호환 문제를 피하고, 조회가 행 단위로 닫혀 있음을
    코드상 자명하게 만들기 위함이다. (searchsorted = 각 행의 id만 참조)
    """
    ids = np.asarray(table["ids"], dtype="int64")
    values = np.asarray(table["values"], dtype="float32")
    index = {c: i for i, c in enumerate(table["cols"])}

    k = pd.to_numeric(df[key_col], errors="coerce").fillna(-1).to_numpy(dtype="int64")
    n = len(k)
    out = {}
    if len(ids) == 0:
        return {c: np.full(n, np.nan, dtype="float32") for c in cols}
    pos = np.clip(np.searchsorted(ids, k), 0, len(ids) - 1)
    hit = ids[pos] == k
    for c in cols:
        v = np.full(n, np.nan, dtype="float32")
        j = index.get(c)
        if j is not None:
            v[hit] = values[pos[hit], j]
        out[c] = v
    return out


def build_features(df, config=None, prof=None):
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    col = {}

    # ---------- 상황 ----------
    for c in SITUATION:
        col[c] = _num(df, c).astype("float32")
    if cfg["keep_season"]:
        col["season"] = _num(df, "season").astype("float32")
    if cfg["keep_raw_ids"]:
        for c in ["pitcher_id", "batter_id"]:
            col[c] = _num(df, c).astype("float32")

    b, s = _num(df, "balls_before"), _num(df, "strikes_before")
    col["count_state"] = (b * 3 + s).astype("float32")
    col["count_diff"] = (b - s).astype("float32")
    col["is_3ball"] = (b == 3).astype("int8")
    col["is_2strike"] = (s == 2).astype("int8")
    col["abs_score_diff"] = _num(df, "score_diff_pitcher_team").abs().astype("float32")
    col["is_late_close"] = (
        (_num(df, "inning").to_numpy() >= 7)
        & (np.asarray(col["abs_score_diff"]) <= 2)
    ).astype("int8")
    col["same_hand"] = (_num(df, "pitcher_hand") == _num(df, "batter_hand")).astype("int8")

    # ---------- 범주형 ----------
    for c in CAT_COLS:
        if c in CAT_MAPS:
            col[c] = df[c].astype(str).map(CAT_MAPS[c]).fillna(-1).astype("int16")
        else:
            col[c] = _num(df, c).fillna(-1).astype("int16")

    # ---------- asof_* ----------
    ck = float(cfg["conf_k"])
    for raw, name in [("asof_pitcher_n", "p_conf"), ("asof_batter_n", "b_conf"),
                      ("asof_pitcher_pitchmix_n", "m_conf")]:
        n = _num(df, raw).fillna(0.0)
        col[name] = (n / (n + ck)).astype("float32")

    if cfg["use_shrinkage"]:
        priors = cfg.get("shrink_priors")
        if not priors:
            raise ValueError("use_shrinkage에는 학습에서 만든 shrink_priors가 필요합니다.")
        k = float(cfg["shrink_k"])
        for rate_col, n_col in SHRINK_PAIRS:
            pr = float(priors[rate_col])
            r = _num(df, rate_col).fillna(pr)
            n = _num(df, n_col).fillna(0.0)
            col[SHRINK_NAME[rate_col]] = (((r * n) + pr * k) / (n + k)).astype("float32")
    else:
        for rate_col, _ in SHRINK_PAIRS:
            col[SHRINK_NAME[rate_col]] = _num(df, rate_col).astype("float32")

    for tag in ["success", "middle"]:
        for w in [1, 3, 5]:
            src = f"asof_pitcher_prev{w}_game_{tag}_rate"
            col[f"p_{'succ' if tag == 'success' else 'mid'}_prev{w}"] = _num(df, src).astype("float32")
    col["has_prev"] = _num(df, "asof_pitcher_prev1_game_success_rate").notna().astype("int8")

    car = _num(df, "asof_pitcher_success_rate")
    p1 = _num(df, "asof_pitcher_prev1_game_success_rate")
    p3 = _num(df, "asof_pitcher_prev3_game_success_rate")
    p5 = _num(df, "asof_pitcher_prev5_game_success_rate")
    col["form_1_5"] = (p1 - p5).astype("float32")
    col["form_3_5"] = (p3 - p5).astype("float32")
    col["form_1_car"] = (p1 - car).astype("float32")
    col["form_5_car"] = (p5 - car).astype("float32")
    mcar = _num(df, "asof_pitcher_middle_rate")
    m1 = _num(df, "asof_pitcher_prev1_game_middle_rate")
    m5 = _num(df, "asof_pitcher_prev5_game_middle_rate")
    col["mid_form_1_5"] = (m1 - m5).astype("float32")
    col["mid_form_1_car"] = (m1 - mcar).astype("float32")

    rev = _num(df, "asof_pitcher_reverse_rate")
    col["p_succ_x_rev"] = (car * rev).astype("float32")
    col["p_ball_minus_stk"] = (_num(df, "asof_pitcher_ball_rate")
                               - _num(df, "asof_pitcher_strike_rate")).astype("float32")
    col["p_succ_minus_mid"] = (car - mcar).astype("float32")
    col["matchup"] = (car - _num(df, "asof_batter_success_rate")).astype("float32")
    col["matchup_mid"] = (mcar - _num(df, "asof_batter_middle_rate")).astype("float32")
    col["p_nonfb"] = (_num(df, "asof_pitcher_breaking_rate")
                      + _num(df, "asof_pitcher_offspeed_rate")).astype("float32")

    # ---------- 과거 시즌 프로파일 (학습에서 만든 룩업표 조회) ----------
    if cfg["use_profiles"]:
        if prof is None:
            raise ValueError("use_profiles에는 prof(프로파일 묶음)가 필요합니다.")
        pp = _lookup(df, "pitcher_id", prof["pitcher"], PP_COLS)
        bb = _lookup(df, "batter_id", prof["batter"], BB_COLS)
        for c in PP_COLS:
            col[c] = pp[c]
        for c in BB_COLS:
            col[c] = bb[c]
        col["pt_succ_c"] = _lookup(df, "pitcher_team_id", prof["pteam"], ["pt_succ_c"])["pt_succ_c"]
        col["bt_succ_c"] = _lookup(df, "batter_team_id", prof["bteam"], ["bt_succ_c"])["bt_succ_c"]

        # 상황 매칭 — 지금 상황에 해당하는 스플릿을 직접 골라 건넨다
        bh, ph = _num(df, "batter_hand").to_numpy(), _num(df, "pitcher_hand").to_numpy()
        col["pp_vs_cur"] = np.where(bh == 1, pp["pp_vs_L"], pp["pp_vs_R"]).astype("float32")
        col["bb_vs_cur"] = np.where(ph == 1, bb["bb_vs_L"], bb["bb_vs_R"]).astype("float32")
        col["matchup_cur"] = (np.asarray(col["pp_vs_cur"])
                              - np.asarray(col["bb_vs_cur"])).astype("float32")

        bn, sn = b.to_numpy(), s.to_numpy()
        col["pp_count_cur"] = np.where(
            bn == 3, pp["pp_3ball"],
            np.where(sn == 2, pp["pp_2strike"],
                     np.where(sn > bn, pp["pp_ahead"],
                              np.where(bn > sn, pp["pp_behind"], pp["pp_succ_c"])))
        ).astype("float32")
        gt = df["game_type"].astype(str).to_numpy()
        col["pp_seg_cur"] = np.where(gt == "F", pp["pp_f_succ"], pp["pp_r_succ"]).astype("float32")
        inn = _num(df, "inning").to_numpy()
        col["pp_inn_cur"] = np.where(
            inn >= 7, pp["pp_late"],
            np.where(inn <= 3, pp["pp_early"], pp["pp_succ_c"])
        ).astype("float32")

        # 올해 폼 vs 통산 — 프로파일은 편차 저장이므로 league를 더해 절대수준 복원
        pabs = pp["pp_succ_c"] + pp["pp_league"]
        labs = pp["pp_last_season"] + pp["pp_league"]
        col["form_vs_career"] = (car.to_numpy() - pabs).astype("float32")
        col["prev1_vs_last"] = (p1.to_numpy() - labs).astype("float32")
        col["prev5_vs_last"] = (p5.to_numpy() - labs).astype("float32")
        col["b_form_vs_career"] = (
            _num(df, "asof_batter_success_rate").to_numpy()
            - (bb["bb_succ_c"] + bb["bb_league"])
        ).astype("float32")

    out = pd.DataFrame(col, index=df.index)
    # ---------- 대상 시즌 한정 성적 ----------
    # asof_*는 '통산' 누적이라(시즌 경계에서 리셋되지 않음) 베테랑일수록
    # 올해 성적이 희석된다. 행 자신의 (asof_n, asof_rate)에서 학습 데이터로
    # 계산한 그 선수의 '이전 시즌들 통산 합계'를 빼면 대상 시즌 한정 성적이
    # 나온다. prev1/3/5경기(짧은 창)와 통산(긴 창) 사이의 중간 창에 해당한다.
    #
    # 규칙 관점: 사용하는 것은 (1) 행 자신의 입력 변수, (2) 공식 학습 데이터로
    # 만든 선수별 상수뿐이다. 평가 데이터의 다른 행은 일절 참조하지 않으며,
    # 개별 투구의 정답을 복원하지도 않는다(여러 투구에 대한 평균값).
    if cfg["use_season_to_date"] and cfg["use_profiles"]:
        min_n = float(cfg["std_min_n"])
        for tag, key, table, extra, rate_col, n_col in [
            ("p", "pitcher_id", prof["pitcher"], PP_LOOKUP_EXTRA,
             "asof_pitcher_success_rate", "asof_pitcher_n"),
            ("b", "batter_id", prof["batter"], BB_LOOKUP_EXTRA,
             "asof_batter_success_rate", "asof_batter_n"),
        ]:
            ex = _lookup(df, key, table, extra)
            prior_n = np.nan_to_num(ex[extra[0]], nan=0.0)
            prior_s = np.nan_to_num(ex[extra[1]], nan=0.0)
            n_all = _num(df, n_col).fillna(0.0).to_numpy()
            s_all = n_all * _num(df, rate_col).fillna(0.0).to_numpy()
            sn = n_all - prior_n
            ss = s_all - prior_s
            ok = (sn >= min_n) & (ss >= -0.5) & (ss <= sn + 0.5)
            rate = np.full(len(df), np.nan, dtype="float32")
            rate[ok] = np.clip(ss[ok] / sn[ok], 0.0, 1.0)
            col[f"{tag}_std_rate"] = rate
            col[f"{tag}_std_n"] = np.where(sn >= 0, sn, np.nan).astype("float32")
            col[f"{tag}_std_conf"] = (np.maximum(sn, 0) /
                                      (np.maximum(sn, 0) + 300.0)).astype("float32")
            base = np.asarray(col[SHRINK_NAME[rate_col]], dtype="float32")
            col[f"{tag}_std_vs_career"] = (rate - base).astype("float32")

    return out_frame(col, df)


def out_frame(col, df):
    out = pd.DataFrame(col, index=df.index)
    return out.replace([np.inf, -np.inf], np.nan)


def apply_fixed_shift(pred, shift):
    """학습에서 확정한 상수 shift 적용 — 행별 독립(순위 불변)."""
    p = np.clip(np.asarray(pred, dtype="float64"), 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p)) + float(shift)
    return 1.0 / (1.0 + np.exp(-z))
