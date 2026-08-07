# script.py
import json
import os

import joblib
import numpy as np
import pandas as pd

ID_COL = "row_id"
TARGET_COL = "control_success"

# 과거 이력 rate의 베이지안 shrinkage 강도 (train.py와 반드시 동일해야 함)
SHRINK_K = 300.0
# 표본 수 saturation 계수 — n/(n+K) 로 [0,1] 로 눌러 2025의 큰 n도 학습 구간 안에 들어오게 한다
CONF_K = 500.0

# 원본에서 그대로 쓰는 수치형 컬럼
RAW_NUM = [
    "game_month", "game_dayofweek", "inning",
    "balls_before", "strikes_before", "outs_before",
    "run_total_before", "score_diff_pitcher_team",
    "runner_on_1b", "runner_on_2b", "runner_on_3b", "num_runners_on",
    "home_win_expectancy", "li",
]

# 범주형 컬럼 (feature_meta.json 의 levels 로 Categorical 고정)
CAT_COLS = [
    "top_bottom", "game_type", "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id",
]

# 학습/추론에서 절대 쓰지 않는 컬럼
#   season          : 2025는 학습에 없어 트리가 외삽 불가
#   pitcher/batter_id: 연도별 신규 선수 24~33%, cold-start 취약
#   asof_*_n        : 누적값이라 연도별 폭증(+411%/+650%) — conf 피처로 변환해서만 사용
#   base_state      : runner_on_* + num_runners_on 과 중복, EDA상 기여도 음수
DROP_COLS = [ID_COL, TARGET_COL, "season", "pitcher_id", "batter_id", "base_state"]


# =======================
# 데이터 로드 유틸
# =======================

def load_test(path):
    """평가 데이터(csv) 로드. 한 행이 투구 하나."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    if ID_COL not in df.columns:
        raise ValueError(f"test 데이터에 {ID_COL} 컬럼이 없음: {list(df.columns)[:5]}")
    return df


def load_sample_submission(path):
    """sample_submission.csv 로드 — 제출 파일의 row_id 순서/컬럼 기준."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    if list(df.columns[:2]) != [ID_COL, TARGET_COL]:
        raise ValueError(
            f"sample_submission 컬럼이 ({ID_COL}, {TARGET_COL})이 아님: "
            f"{list(df.columns)}")
    return df


def load_artifacts(model_dir):
    """model/ 안의 학습 산출물 로드. 학습 데이터에는 접근하지 않는다."""
    model = joblib.load(os.path.join(model_dir, "model.pkl"))
    with open(os.path.join(model_dir, "feature_meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    with open(os.path.join(model_dir, "calibration.json"), encoding="utf-8") as f:
        calib = json.load(f)
    return model, meta, calib


# =======================
# 학습 때 사용한 전처리 (train.py 가 이 함수를 그대로 import 해서 씀)
# =======================

def _shrink(rate, n, prior):
    """표본 수 기반 베이지안 shrinkage. n=0 이면 prior 로 수렴한다."""
    r = rate.fillna(prior).to_numpy(dtype="float64")
    nn = n.fillna(0.0).to_numpy(dtype="float64")
    return (r * nn + prior * SHRINK_K) / (nn + SHRINK_K)


def engineer(df, priors):
    """투구 직전 정보만으로 파생 피처를 만든다.

    행 단위로 닫혀 있는 변환만 사용한다. 다른 행의 통계나 전체 분포를 참조하지
    않으므로 test.csv 를 행 단위로 독립 예측한다는 규칙을 위반하지 않는다.
    priors 는 학습 데이터에서 산출해 feature_meta.json 에 저장된 상수다.
    """
    out = pd.DataFrame(index=df.index)

    for c in RAW_NUM:
        out[c] = pd.to_numeric(df[c], errors="coerce")

    pn = pd.to_numeric(df["asof_pitcher_n"], errors="coerce")
    bn = pd.to_numeric(df["asof_batter_n"], errors="coerce")
    mn = pd.to_numeric(df["asof_pitcher_pitchmix_n"], errors="coerce")

    # --- 투수 통산 이력 (shrinkage) ---
    p_succ = _shrink(df["asof_pitcher_success_rate"], pn, priors["p_succ"])
    p_rev = _shrink(df["asof_pitcher_reverse_rate"], pn, priors["p_rev"])
    p_mid = _shrink(df["asof_pitcher_middle_rate"], pn, priors["p_mid"])
    p_ball = _shrink(df["asof_pitcher_ball_rate"], pn, priors["p_ball"])
    p_stk = _shrink(df["asof_pitcher_strike_rate"], pn, priors["p_stk"])
    out["p_succ"] = p_succ
    out["p_rev"] = p_rev
    out["p_mid"] = p_mid
    out["p_ball"] = p_ball
    out["p_stk"] = p_stk

    # --- 타자 이력 ---
    b_succ = _shrink(df["asof_batter_success_rate"], bn, priors["b_succ"])
    b_mid = _shrink(df["asof_batter_middle_rate"], bn, priors["b_mid"])
    out["b_succ"] = b_succ
    out["b_mid"] = b_mid

    # --- 표본 수는 saturation 변환으로만 사용 (raw 누적값은 연도 드리프트가 심함) ---
    out["p_conf"] = (pn.fillna(0.0) / (pn.fillna(0.0) + CONF_K)).to_numpy()
    out["b_conf"] = (bn.fillna(0.0) / (bn.fillna(0.0) + CONF_K)).to_numpy()
    out["m_conf"] = (mn.fillna(0.0) / (mn.fillna(0.0) + CONF_K)).to_numpy()

    # --- 구종 구성 ---
    p_fb = _shrink(df["asof_pitcher_fastball_rate"], mn, priors["p_fb"])
    p_br = _shrink(df["asof_pitcher_breaking_rate"], mn, priors["p_br"])
    p_os = _shrink(df["asof_pitcher_offspeed_rate"], mn, priors["p_os"])
    out["p_fb"] = p_fb
    out["p_br"] = p_br
    out["p_os"] = p_os
    # 2022년 구종 레짐 변화(패스트볼 57.5%->47.7%)의 방향을 직접 담는 축
    out["p_nonfb"] = p_br + p_os

    # --- 최근 폼: 직전 N경기 vs 통산 편차 ---
    prev_cols = {
        1: ("asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev1_game_middle_rate"),
        3: ("asof_pitcher_prev3_game_success_rate", "asof_pitcher_prev3_game_middle_rate"),
        5: ("asof_pitcher_prev5_game_success_rate", "asof_pitcher_prev5_game_middle_rate"),
    }
    prev_s, prev_m = {}, {}
    for k, (sc, mc) in prev_cols.items():
        prev_s[k] = pd.to_numeric(df[sc], errors="coerce").fillna(
            pd.Series(p_succ, index=df.index)).to_numpy(dtype="float64")
        prev_m[k] = pd.to_numeric(df[mc], errors="coerce").fillna(
            pd.Series(p_mid, index=df.index)).to_numpy(dtype="float64")
        out[f"p_succ_prev{k}"] = prev_s[k]
        out[f"p_mid_prev{k}"] = prev_m[k]

    out["has_prev"] = df["asof_pitcher_prev1_game_success_rate"].notna().astype("int8")
    out["form_1_5"] = prev_s[1] - prev_s[5]
    out["form_3_5"] = prev_s[3] - prev_s[5]
    out["form_1_car"] = prev_s[1] - p_succ
    out["form_5_car"] = prev_s[5] - p_succ
    out["mid_form_1_car"] = prev_m[1] - p_mid
    out["mid_form_1_5"] = prev_m[1] - prev_m[5]

    # --- 상호작용 / 매치업 ---
    out["p_succ_x_rev"] = p_succ * p_rev
    out["p_ball_minus_stk"] = p_ball - p_stk
    out["p_succ_minus_mid"] = p_succ - p_mid
    out["matchup"] = p_succ - b_succ
    out["matchup_mid"] = p_mid - b_mid

    # --- 볼카운트 ---
    b = out["balls_before"]
    s = out["strikes_before"]
    out["count_state"] = b * 3 + s
    out["count_diff"] = b - s
    out["is_3ball"] = (b >= 3).astype("int8")
    out["is_2strike"] = (s >= 2).astype("int8")

    # --- 경기 상황 ---
    out["abs_score_diff"] = out["score_diff_pitcher_team"].abs()
    out["is_late_close"] = ((out["inning"] >= 7) & (out["abs_score_diff"] <= 2)).astype("int8")

    for c in CAT_COLS:
        out[c] = df[c].astype("object")

    return out


def build_features(df, meta):
    """engineer() 결과를 학습 때의 컬럼 순서/범주 레벨에 정렬한다."""
    X = engineer(df, meta["priors"])
    for c in CAT_COLS:
        levels = meta["cat_levels"][c]
        s = X[c].astype("str")
        # 2025 신규 팀/코드 등 학습 때 없던 값은 명시적으로 결측 처리한다.
        # (HistGradientBoosting 은 결측을 학습된 방향으로 보내므로 안전하게 동작)
        unseen = ~s.isin(levels)
        if unseen.any():
            print(f"  주의: {c} 미지 범주 {int(unseen.sum())}건 -> 결측 처리")
            s = s.where(~unseen, other=None)
        X[c] = pd.Categorical(s, categories=levels)
    missing = [c for c in meta["features"] if c not in X.columns]
    if missing:
        raise ValueError(f"생성되지 않은 피처: {missing}")
    return X[meta["features"]]


# =======================
# 후처리 (캘리브레이션)
# =======================

def apply_calibration(p, calib):
    """로짓 절편 시프트.

    시프트 값은 학습 데이터의 연도별 성공률 추세(2019~2024)로부터 사전에 산출해
    calibration.json 에 저장한 상수다. 평가 데이터의 예측 분포를 보고 사후에
    맞추는 방식이 아니므로 대회 규칙상 허용된다.
    """
    eps = float(calib.get("clip_eps", 1e-6))
    p = np.clip(np.asarray(p, dtype="float64"), eps, 1.0 - eps)
    z = np.log(p / (1.0 - p)) + float(calib["logit_shift"])
    return np.clip(1.0 / (1.0 + np.exp(-z)), eps, 1.0 - eps)


# =======================
# 제출 파일 생성 유틸
# =======================

def merge_predictions(sub, ids, preds):
    """sample_submission의 row_id 순서에 맞춰 예측 확률 병합.

    예측에 없는 row_id는 sample_submission의 기존 값(placeholder)을 유지한다.
    """
    pred_map = dict(zip(ids, preds))
    values, n_missing = [], 0
    for rid, cur in zip(sub[ID_COL], sub[TARGET_COL]):
        p = pred_map.get(rid)
        if p is None:
            n_missing += 1
            values.append(cur)
        else:
            values.append(p)
    if n_missing:
        print(f" 경고: 예측이 없어 placeholder를 유지한 row_id {n_missing}건")
    sub[TARGET_COL] = values
    return sub


def save_submission(path, sub):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sub.to_csv(path, index=False, encoding="utf-8")


# =======================
# main
# =======================

def main():
    # ---- 경로 변수 (필요에 따라 수정) ----
    TEST_DIR = "./data"            # test.csv, sample_submission.csv 위치
    MODEL_DIR = "./model"          # 학습 산출물 위치
    OUT_DIR = "./output"
    TEST_PATH = os.path.join(TEST_DIR, "test.csv")
    SAMPLE_SUB_PATH = os.path.join(TEST_DIR, "sample_submission.csv")
    OUT_PATH = os.path.join(OUT_DIR, "submission.csv")

    # ---- 모델 로드 ----
    print("Load model...")
    models, meta, calib = load_artifacts(MODEL_DIR)
    print(f" OK. n_models={len(models)}  n_features={len(meta['features'])}"
          f"  logit_shift={calib['logit_shift']:+.5f}")

    # ---- 테스트 데이터 로드 ----
    print("Load test data...")
    test = load_test(TEST_PATH)
    sub = load_sample_submission(SAMPLE_SUB_PATH)
    print(f" test={len(test)}  submission={len(sub)}")

    # ---- 전처리 (학습과 동일) ----
    print("Build features...")
    ids = test[ID_COL].tolist()
    X = build_features(test, meta)
    print(f" features={X.shape[1]}")

    # ---- 예측 (제구 성공 확률) ----
    print("Inference model...")
    if len(X):
        acc = np.zeros(len(X), dtype="float64")
        for i, m in enumerate(models, 1):
            acc += m.predict_proba(X)[:, 1]
            print(f"  model {i}/{len(models)} done")
        preds = acc / len(models)
        preds = apply_calibration(preds, calib)
    else:
        preds = np.array([], dtype="float64")
    print(f" preds={len(preds)}"
          + (f"  mean={preds.mean():.5f}  min={preds.min():.5f}"
             f"  max={preds.max():.5f}" if len(preds) else ""))

    # ---- sample_submission 기반 결과 생성 ----
    print("Build submission...")
    sub = merge_predictions(sub, ids, preds)
    sub[TARGET_COL] = np.clip(
        pd.to_numeric(sub[TARGET_COL], errors="coerce").fillna(0.5).to_numpy(), 0.0, 1.0)
    save_submission(OUT_PATH, sub)
    print(f"✅ Saved: {OUT_PATH} (rows={len(sub)})")


if __name__ == "__main__":
    main()
