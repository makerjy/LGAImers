# LG Aimers 9기 x LG스포츠: 투수 제구 성공 확률 예측

투구 직전까지 알려진 경기 상황과 선수 이력을 이용해 각 투구의
`control_success` 확률을 예측하는 프로젝트입니다. 최종 V14는 시간 순서와
행 독립성을 지키는 피처 파이프라인 위에 CatBoost, LightGBM, TabM을 결합하고,
기존 V13 예측을 보정하는 방향성 앙상블로 구성했습니다.

> **검증 범위:** 아래 점수는 2022-2024 시즌 temporal OOF에서 계산한 로컬
> 결과입니다. `1515.04`는 공식 점수가 아닌 official-like 환산치이며, 대회
> 서버 점수는 아직 검증되지 않았습니다.

## 결과

| 모델 | Temporal OOF Brier | Standard BSS | Official-like score |
|---|---:|---:|---:|
| V13 baseline | 0.246641 | 0.013345 | 1334.49 |
| V14 uncalibrated | 0.246279 | 0.014791 | 1479.12 |
| **V14 final** | **0.246189** | **0.015150** | **1515.04** |

낮은 Brier score가 더 좋습니다. V14 final은 V13 대비 Brier를
`0.000451` 낮췄고, 세 개 검증 시즌에서 모두 개선됐습니다.

| 검증 시즌 | V13 대비 Brier 변화 |
|---|---:|
| 2022 | -0.000510 |
| 2023 | -0.000649 |
| 2024 | -0.000202 |

투수 단위 3,000회 cluster bootstrap에서도 V14와 V13의 Brier 차이 95% CI가
`[-0.000567, -0.000352]`로 0 아래에 위치했습니다.

## 최종 구조

```text
Pitch context and as-of history
        |
        +-- row-independent situation features
        +-- Bayesian-shrunk as-of rates
        +-- strict-past pitcher / batter / team profiles
        |
        v
CatBoost variants + LightGBM profile ensemble
        |
        +-- TabM seed 1234
        +-- TabM seed 2345
        +-- V13 backbone prediction
        |
        v
R-game-only weighted residual directions
        |
        v
Fixed game_type x month affine calibration
        |
        v
control_success probability
```

최종 예측식의 핵심은 V13을 기준점으로 두고, regular-season 행에 대해서만
MH profile ensemble과 두 TabM 모델의 예측 방향을 가중 결합하는 것입니다.
마지막으로 학습 OOF에서 고정한 `game_type x month` 보정값을 적용합니다.

## 피처 엔지니어링

- 투구 카운트, 주자, 이닝, 점수 차, LI, 기대 승률 등의 경기 상황
- 최근 1/3/5경기 폼과 통산 성적의 차이
- 표본 수에 따른 Bayesian shrinkage와 confidence saturation
- 이전 시즌만 사용한 투수, 타자, 투수팀, 타자팀 프로파일
- 좌우 상대, 유불리 카운트, 2스트라이크/3볼, 고레버리지, 이닝 구간 split
- 현재 시즌 성적과 과거 통산 프로파일의 차이
- raw 선수 ID와 season 값에 직접 의존하지 않는 cold-start 대응

`build_features()`는 예측 대상 행과 학습에서 미리 확정한 lookup table만
참조합니다. 테스트 행 전체의 집계, 순위, rolling 연산을 사용하지 않아 단일 행
추론과 batch 추론이 같은 결과를 냅니다.

## 검증 설계

- 검증 시즌: 2022, 2023, 2024
- 학습 데이터: 각 검증 시즌보다 과거인 최대 4개 시즌
- 최근 시즌 가중치를 높이는 recency weighting 적용
- 선수 프로파일은 검증 시즌 이전 데이터로만 생성
- 최종 비교는 pitcher-cluster bootstrap으로 불확실성 확인
- 제출 패키지는 행 순서 shuffle, 미지 선수/결측값, IndexError 회귀 테스트 수행

최종 패키지 검증 결과는 다음과 같습니다.

| 검사 | 결과 |
|---|---|
| ZIP CRC integrity | PASS |
| IndexError regression | PASS |
| Row-order independence | PASS |
| Unknown/missing input | PASS |
| TrackMan file dependency | 없음 |
| 10분 이내 추론 | PASS |

## 프로젝트 구조

```text
challengers/v14_mh_profile_ensemble/
  features.py          # 행 독립 피처 생성
  profiles.py          # strict-past 선수/팀 프로파일
  temporal_train.py    # temporal OOF CatBoost/LightGBM 학습
  regular_train.py     # 정규시즌 특화 실험
  residual_stack.py    # 잔차 스태킹 실험
  tabm_screen.py       # TabM 후보 검증
  tabm_full_refit.py   # TabM 최종 재학습
  select_v14.py        # 앙상블 선택, 보정, bootstrap
final/
  v14_results.json     # 전체 성능과 진단 결과
  v14_policy.json      # 최종 앙상블/보정 정책
  package_verification.json
pyproject.toml
uv.lock
```

대회 원본 CSV, OOF parquet와 모델 가중치는 Git에서 제외합니다. 최종 제출
패키지는 [v14-final Release](https://github.com/taeg2/Konkuk_CS_Aimers/releases/tag/v14-final)에서
다운로드할 수 있습니다. 파일명은 `v14_submit.zip`이며 SHA-256은
`af76cc40cb2313bf2b9bc8cd6deda943be245adf6f9f5d74c1ef6b3561d33e09`입니다.

## 실행 환경

Python 3.11 이상과 [uv](https://docs.astral.sh/uv/)를 사용합니다.

```bash
uv sync --dev
```

공식 데이터를 아래 위치에 둡니다.

```text
data/raw/train.csv
data/raw/test.csv
data/raw/sample_submission.csv
```

Temporal CatBoost/LightGBM OOF 실험은 다음 명령으로 실행합니다.

```bash
uv run python -m challengers.v14_mh_profile_ensemble.temporal_train
```

TabM과 최종 선택 단계는 각각의 모듈로 분리되어 있습니다.

```bash
uv run python -m challengers.v14_mh_profile_ensemble.tabm_screen
uv run python -m challengers.v14_mh_profile_ensemble.tabm_full_refit
uv run python -m challengers.v14_mh_profile_ensemble.select_v14
```

## 한계와 해석

- 로컬 1515 crossing은 pooled held-out prediction으로 적합한 월별 calibration을
  포함하므로 탐색적 결과로 해석해야 합니다.
- next-season nested calibration은 V13보다 약해 calibration의 시간 이동 안정성이
  충분히 확인되지 않았습니다.
- TrackMan history는 공식 데이터와 신뢰할 수 있는 선수 crosswalk를 확보하지
  못했고 negative-control 실험에서 이득이 없어 최종 추론 의존성에서 제외했습니다.
- 최종 성능 판단에는 대회 서버 점수와 추가적인 미래 시즌 검증이 필요합니다.
