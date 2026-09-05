# LG Aimers 9기 × LG스포츠
# 투수 제구 성공 확률 예측

투구가 시작되기 직전까지 알 수 있는 경기 상황과 선수의 과거 기록으로
`control_success`의 확률을 예측했다. 제출 시점은 2025시즌이었고, 학습 데이터는
2019~2024시즌으로 구성되어 있다.

처음에는 CatBoost에 feature를 계속 추가하는 방식으로 시작했다. 이후 validation과
leaderboard의 차이를 확인하면서 선수 기록을 만드는 시점, 최근 시즌의 비중, 확률
보정 방식을 다시 바꿨다. HCCN은 그 과정에서 시도한 residual challenger였고, 최종
후보는 strict-past profile을 입력으로 사용하는 tree ensemble과 TabM blend에 더
가까워졌다.

## 문제와 데이터

한 행은 투구 하나다. 예측값은 0 또는 1의 class가 아니라 제구 성공 확률이다.
확률이 실제 빈도와 얼마나 가까운지가 중요하므로 Brier score와 Brier Skill Score
(BSS)를 주 지표로 사용했다.

| 항목 | 확인된 값 |
|---|---:|
| train | 1,475,092행 × 49열 |
| test | 5행 × 48열 |
| 학습 시즌 | 2019~2024 |
| 예측 시즌 | 2025 |
| target | `control_success` |
| train target mean | 0.523766 |
| 정규시즌(`R`) 행 | 1,314,088 |
| `F` game type 행 | 161,004 |

`balls_before`, `strikes_before`, `outs_before`, 이닝, 점수 차, 주자, `li`,
투수·타자 손잡이, 팀과 선수별 누적 기록이 기본 입력이었다. 모든 원본 컬럼을
그대로 사용하지 않았다. 2025년에는 학습에 없는 값이 생기는 컬럼, 다른 컬럼과
완전히 겹치는 값, 예측 시점 이후 정보가 섞일 수 있는 값부터 제외했다.

## 처음에는 CatBoost로 시작했다

처음에는 경기 상황과 `asof_*` 누적 기록을 조합해 CatBoost 기준선을 만들었다.
CatBoost, LightGBM, TabM 후보를 각각 확인했고, 좋은 validation 점수 하나만으로
모델을 고르지 않으려고 Brier를 기준으로 비교했다.

이때부터 두 가지 문제가 반복해서 나타났다.

1. 모델을 크게 만들거나 feature를 많이 추가해도 점수가 계속 같은 방향으로
   좋아지지는 않았다.
2. random split에서 좋아 보이는 결과가 다음 시즌을 흉내 낸 validation이나
   leaderboard에서 같은 폭으로 재현되지 않았다.

`jw_branch` 문서에는 random 5-fold AUC `0.55931`과 temporal AUC `0.55033`이
각각 기록되어 있다. 두 수치는 서로 다른 validation 방식의 결과라서 현재 V14
Brier와 직접 비교할 수는 없지만, 행을 무작위로 섞었을 때 결과가 낙관적으로 보일
수 있다는 판단을 하게 만든 기록이다.

2025년을 예측하는 문제에서는 검증 행보다 미래에 해당하는 정보가 feature에
들어가면 안 된다. 모델 구조를 바꾸기 전에 시즌 순서와 선수 기록 생성 방식부터
다시 확인하기로 했다.

## Feature engineering

### 경기 상황과 최근 기록

V14의 공개 feature builder는 한 행의 입력과 학습 시 미리 만든 lookup만 사용한다.
`build_features()` 안에는 행 간 `groupby`, `rolling`, `lag`, rank가 없다.

상황 feature는 다음과 같다.

- `game_month`, `game_dayofweek`, 이닝
- 볼·스트라이크·아웃 카운트와 `count_state`, `count_diff`
- 투수 팀 기준 점수 차, 절대 점수 차, 늦은 이닝 접전 여부
- 1/2/3루 주자와 주자 수
- `home_win_expectancy`, `li`
- 투수·타자 손잡이와 `same_hand`

투수의 최근 1/3/5경기 성공률과 중간 결과율도 사용했다. 최근 기록과 통산
기록의 차이(`form_1_5`, `form_3_5`, `form_1_car` 등)를 추가해 현재 폼을 별도
값으로 전달했다. 시즌 한정 기록은 `asof_*` 누적값에서 과거 시즌 누적값을 빼서
복원한다. 표본이 `std_min_n=20`보다 적으면 안정적인 시즌 기록으로 사용하지
않는다.

### strict-past profile

2025시즌의 선수를 예측하면서 2025시즌의 기록을 profile에 섞을 수는 없다.
`profiles.py`는 target season보다 작은 시즌만 골라 투수·타자·투수팀·타자팀
lookup을 만든다.

투수 profile에는 전체 성공률, 표본 수, 시즌 수, 좌·우타자 상대 성적, 앞선/뒤진
카운트, 2스트라이크·3볼, high-LI, 초반·후반 이닝, `R/F` split, 평균 이닝과
평균 LI를 넣었다. 타자 profile에는 좌·우 투수 상대 성적, platoon 차이,
`R/F`와 카운트·LI·이닝 관련 값이 들어간다. 팀 profile은 과거 성공률을 별도
lookup으로 제공한다.

예측 행은 ID로 lookup을 조회할 뿐이다. 현재 test 전체를 다시 집계하지 않으므로
한 행만 넣었을 때와 전체 행을 넣었을 때의 feature가 달라지지 않는다. 최종 패키지
검증에서도 행 순서를 섞었을 때 최대 예측 차이는 `0.0`이었다.

### shrinkage와 reliability

10번 던져 7번 성공한 선수와 1,000번 던져 700번 성공한 선수의 관측 성공률은
둘 다 0.7이다. 두 값을 같은 신뢰도로 볼 수는 없다. 그래서 성공률을 그대로
넣지 않고 학습 history의 prior와 표본 수를 함께 사용했다.

현재 V14 feature code의 비율 보정식은 다음과 같다.

```text
shrunk_rate = (observed_rate × n + prior × k) / (n + k)
confidence  = n / (n + 500)
```

일반 feature의 `shrink_k`는 `150`이다. profile 내부의 entity effect에는 `k=250`,
직전 시즌 effect에는 `k=150`, 팀 profile에는 `k=400`이 사용된다. prior는 각
training history에서 계산한다.

표본이 없거나 새로운 선수인 경우에는 lookup miss와 수치 결측이 남을 수 있다.
CatBoost와 LightGBM은 결측을 그대로 처리하고, TabM은 training fold의 중앙값과
표준화 통계를 사용한다. 범주형 입력은 training vocabulary에 없는 값을 unknown
bucket으로 보낸다. V14에서는 raw `pitcher_id`, `batter_id`, `season`을 기본 입력에
넣지 않았다. 새 선수 ID를 외우거나 2025를 연속형 season 값으로 외삽하는 문제를
줄이기 위해서다.

### 최근 시즌과 recency

2019년 기록과 2024년 기록을 같은 비중으로 사용할지 계속 확인했다. 현재 V14
temporal training은 validation season 직전 최대 4개 시즌을 사용한다.

| validation season | training seasons | season weight |
|---:|---|---|
| 2022 | 2020~2021 | 2020: 0.75, 2021: 1.00 |
| 2023 | 2020~2022 | 2020: 0.50, 2021: 0.75, 2022: 1.00 |
| 2024 | 2020~2023 | 2020: 0.25, 2021: 0.50, 2022: 0.75, 2023: 1.00 |

이 설정은 `temporal_train.py`의 현재 코드 기준이다. 과거 `mh_branch`의 4년
weight와 `jw_branch`의 5년 decay를 참고했지만, 모든 과거 실험이 현재 최종 모델에
그대로 들어간 것은 아니다.

### TrackMan

현재 V14 source와 최종 feature schema에는 TrackMan feature가 없다. 최종 제출
경로에서 TrackMan 데이터를 사용했다고 볼 수 있는 코드나 artifact도 남아 있지
않다. 따라서 README에서 TrackMan 성능을 최종 모델 결과처럼 쓰지 않았다.

## HCCN을 붙여 본 이유

CatBoost가 이미 기준선 역할을 하고 있을 때 neural network가 전체 확률을 처음부터
다시 예측하게 하는 것은 부담이 컸다. 기준 모델의 예측을 버리고 새 분류기를
학습하기보다, CatBoost가 놓치는 부분만 작은 보정값으로 학습하는 방법을 먼저
확인했다.

입력을 하나의 벡터로 합치지 않고 다음 세 그룹으로 나눴다.

- **Entity**: 투수, 타자, 팀, 경기 상태의 categorical embedding
- **History**: as-of 성적, 최근 폼, 표본 수와 결측 관련 수치
- **Context**: count, base, inning, score, LI, pressure

출력은 다음 순서다.

```text
CatBoost OOF probability
          ↓ logit
      base_logit
          +
      HCCN residual
          ↓ sigmoid
   final probability
```

```text
final_logit = base_logit + delta_logit
prediction   = sigmoid(final_logit)
```

이미 맞고 있는 기준선의 확률을 크게 흔들지 않기 위해 residual을 제한했다.
`max_delta × tanh(delta_raw)`로 logit 보정 범위를 묶고, residual L2 penalty를
추가했다. reliability gate는 pitcher, matchup, pressure, prior expert의 비중을
행마다 정하고, V2/V3의 magnitude gate는 보정 자체의 크기를 줄일 수 있도록 했다.

## HCCN 구조와 실제 입력

초기 `ResidualHCCN`의 공유 표현은 다음 순서로 만들어진다.

```text
Entity Tower       History Tower       Context Tower
embeddings         numeric MLP         cat embeddings + numeric MLP
      \                 |                    /
       \                |                   /
        +--------- shared representation --+
                         |
             +-----------+-----------+
             |                       |
      low-rank Cross Network       Deep MLP
       2 layers, rank 32        256 -> 128 -> 64
             +-----------+-----------+
                         |
          4 residual experts, hidden 64
             pitcher / matchup / pressure / prior
                         |
           Reliability Gate -> softmax(4)
                         |
            optional Magnitude Gate (V2/V3)
                         |
       bounded residual -> base logit + residual
```

Cross branch는 low-rank projection과 residual connection을 사용한다.
Deep branch는 LayerNorm, SiLU, dropout이 있는 MLP다. V1 loss는
`BCE + 0.25 × Brier + 0.001 × residual L2`이고, V2/V3는 residual penalty와
anchor loss 설정이 달라졌다.

초기 checkpoint metadata에서 확인한 입력 규모는 다음과 같다.

| 버전 | 확인된 metadata |
|---|---|
| V1 | history raw 52개 → 104개, context raw 27개 → 54개, gate raw 10개 → 20개, entity categorical 11개, 271,461 parameters, final refit 2 epochs |
| V2 | magnitude gate 사용, model-state tower 미사용, 282,278 parameters, final refit 12 epochs |
| V3 | V2에 model-state tower 추가, 308,230 parameters, final refit 15 epochs |

버전별로 구조와 epoch, penalty가 함께 바뀌었다. 그래서 V1/V2/V3 성능 차이를
특정 tower 하나의 효과로 분리할 수 없다. HCCN의 history도 sequence encoder가
아니라 집계된 snapshot vector다. `contextual_history.py`에 여러 contextual
aggregate 정의가 있지만 초기 HCCN training path에서 실제로 호출되는 것은
확인되지 않았다.

## HCCN 결과와 한계

`jy_branch` README와 최종 ZIP policy/manifest에는 다음 기록이 남아 있다. 서로
같은 OOF 생성 방식이라고 확인할 수 없는 값이므로 현재 V14 표와 섞지 않았다.

| 기록 | Brier | 추가 기록 | 확인 방식 |
|---|---:|---|---|
| HCCN v1 | 0.247123 | branch 표기 BSS 1141.43, 2024 Brier 0.248198, AUROC 0.545125 | branch README와 ZIP policy |
| HCCN V4 best preselect | 0.247036 | 최종 모델 전체가 아닌 preselect 후보 | ZIP policy |
| HCCN V11 feature rebuild | 0.246938 | manifest 표기 official-like 1215.46 | ZIP manifest |

HCCN source branch의 `artifacts/`에는 실제 OOF parquet와 metric CSV가 남아 있지
않고 placeholder만 있다. 그래서 HCCN과 기준 CatBoost의 동일 fold·동일 seed 개선폭을
현재 clone에서 다시 계산할 수 없다. 위 숫자는 HCCN이 최종 정답이었다는 근거가
아니라, 당시 challenger를 어디까지 실험했는지 보여주는 기록으로 남겼다.

HCCN 계열을 확장하면서 구조가 복잡해졌지만 leaderboard 성능이 같은 방향으로
계속 좋아지지는 않았다. 그 시점부터 다음 문제를 모델 capacity보다 먼저 다시
확인했다.

- 2019~2024와 2025 사이의 시간 분포 차이
- 시즌별 `R/F` 분포와 성공률 변화
- 선수 기록을 어느 시점까지 사용할지
- 표본이 적은 선수의 극단적인 성공률
- 처음 등장하는 선수의 처리
- 최근 기록과 장기 기록 중 어느 쪽을 더 믿을지
- random split과 시간 순서를 지킨 validation의 차이

예를 들어 정규시즌 성공률은 2022년 `0.503691`, 2023년 `0.503118`, 2024년
`0.489707`이었다. `F`는 2022년 `0.708749`에서 2023년 `0.472904`, 2024년
`0.459280`으로 변했다. 이 변화가 있는데도 network 구조만 계속 키우는 것은
다음 시즌 일반화를 확인하는 방법이 아니었다.

## 다른 branch를 비교한 뒤 바꾼 점

`mh_branch`와 `jw_branch`는 모델 이름보다 feature와 validation을 비교하는 데
사용했다.

| 항목 | `jy_branch` HCCN | `mh_branch` / `jw_branch` 기록 | 현재 V14에 반영한 점 |
|---|---|---|---|
| 선수 기록 | embedding과 집계 history를 network에 입력 | strict-past profile, shrinkage, raw ID 제외 | profile lookup과 표본 신뢰도 사용 |
| 시간 사용 | HCCN config의 2년 window, decay 0.7 | `mh`: 4년 weight, `jw`: 5년 season decay와 월별 recency | 최대 4년 temporal training과 recency weight |
| 모델 | residual expert 하나의 계열 | CatBoost·LightGBM을 여러 seed와 설정으로 결합 | 3 CatBoost + LightGBM 12-model ensemble |
| feature 수 | tower별 history/context/gate 분리 | `mh` 117개, `jw` 문서 112개 | 현재 MH schema 117개 |
| 평가 | temporal fold, Brier, cluster bootstrap 코드 | `mh` BSS와 logit 평균, `jw` 다년 temporal AUC | Brier 중심 temporal OOF와 calibration |
| 기록된 결과 | HCCN v1 Brier 0.247123 | `mh` 문서 LB 1025, `jw` 문서 LB 961·다년 평균 AUC 0.55376 | 서로 다른 지표라 순위 비교는 하지 않음 |

`mh_branch`는 row-independent feature와 12개 모델의 logit 평균을 사용했다.
`jw_branch`는 여러 CatBoost 설정과 LightGBM, seed ensemble, regularization을
사용했다. 두 branch를 따라가며 모델 구조를 더 복잡하게 만드는 것보다 과거 기록을
예측 시점에 맞게 만들고, 서로 다른 tree model의 예측 방향을 비교하는 편이 다음
실험으로 이어지기 쉬웠다.

## 현재 최종 후보

### MH profile ensemble

`temporal_train.py`에는 CatBoost 세 설정과 LightGBM 설정이 있다.

| 모델 | 주요 설정 |
|---|---|
| CatBoost A | 600 iterations, learning rate 0.02, depth 7, L2 25, min leaf 100 |
| CatBoost B | 600 iterations, learning rate 0.035, depth 5, L2 3, min leaf 100 |
| CatBoost C | 1,000 iterations, learning rate 0.02, depth 4, L2 8, min leaf 500 |
| LightGBM | 600 rounds, learning rate 0.01, 31 leaves, min leaf 1,500, feature/bagging fraction 0.7, L2 20 |

최종 MH artifact는 세 seed를 사용해 12개 모델을 저장한다. 각 모델의 확률을 logit으로
바꿔 평균한 뒤 sigmoid로 돌린다. 입력은 117개 feature이고 categorical 입력은
6개다.

### TabM 후보

TabM은 MH feature를 받아 정규시즌(`game_type=R`)만 학습한 보조 모델이다.
현재 코드에서 확인되는 설정은 `n_blocks=3`, `d_block=256`, `k=16`,
`arch_type="tabm-mini"`, dropout `0.1`이다. seed `1234`와 `2345`를 따로
학습해 MH와 다른 예측 방향이 있는지 확인했다.

### V13에서 V14로 결합

V14는 V13의 확률을 기준점으로 두고 MH와 TabM의 절대 확률을 바로 평균내지 않았다.
정규시즌 행에서 기준점과 각 후보의 차이만 결합했다.

```text
p_v14_raw = clip(
    p_v13
    + R × [
        0.290352 × (p_mh - p_v13)
      + 0.268029 × (p_tabm_seed1234 - p_v13)
      + 0.245370 × (p_tabm_seed2345 - p_v13)
    ]
)
```

마지막으로 `game_type × game_month`별 affine calibration을 적용했다. 월별 계수와
방향 가중치는 [`final/v14_policy.json`](final/v14_policy.json)에 저장되어 있다.

## Validation과 OOF

현재 V14는 다음 temporal fold를 사용한다.

| validation season | training seasons | validation rows |
|---:|---|---:|
| 2022 | 2020~2021 | 247,472 |
| 2023 | 2020~2022 | 245,525 |
| 2024 | 2020~2023 | 253,507 |

각 fold에서 profile은 validation season 이전 데이터로만 만든다. feature cache와
OOF prediction은 season별로 저장하고, CatBoost와 LightGBM은 최근 시즌에 더 큰
weight를 받는다.

V14 방향 가중치와 월별 calibration은 세 시즌의 held-out prediction을 모아 적합한
local 비교다. 이 결과를 다음 시즌에 그대로 적용하는 nested 검증도 별도로 돌렸다.

| 검증 | 방향·보정 학습 데이터 | V13 대비 Brier delta |
|---|---|---:|
| 2023 | 2022 | +0.001672 |
| 2024 | 2022~2023 | +0.000432 |

2022는 V13을 유지했다. nested pooled Brier는 `0.247337`로 V13보다 높았다.
그래서 pooled OOF에서 나온 1515 local score를 미래 시즌 성능으로 쓰지 않았다.

평가는 다음 기준으로 나눴다.

- Brier: `mean((y - p)^2)`, 낮을수록 좋다.
- Standard BSS: `1 - Brier / climatology_Brier`.
- Official-like score: `100000 × Standard BSS`의 local 표기. 공식 leaderboard 점수가 아니다.
- AUROC: 초기 branch 기록에 있지만 V14 결과 JSON의 최종 지표에는 없다.

## 실험 결과

아래 표는 [`final/v14_results.json`](final/v14_results.json)의 2022~2024 pooled
temporal OOF 결과다.

| 모델 | Brier ↓ | Standard BSS ↑ | Official-like |
|---|---:|---:|---:|
| V13 baseline | 0.246641 | 0.013345 | 1334.49 |
| V14 uncalibrated | 0.246279 | 0.014791 | 1479.12 |
| **V14 month-calibrated final** | **0.246189** | **0.015150** | **1515.04** |

V14 final의 season별 Brier는 다음과 같다.

| validation season | V13 | V14 final | delta |
|---:|---:|---:|---:|
| 2022 | 0.243444 | 0.242933 | -0.000510 |
| 2023 | 0.248728 | 0.248079 | -0.000649 |
| 2024 | 0.247739 | 0.247537 | -0.000202 |

`pitcher_id` cluster bootstrap 3,000회에서 V14 final − V13 Brier delta는
`-0.000451`, 95% CI는 `[-0.000567, -0.000352]`였다. 같은 투수의 여러 투구가
독립 표본처럼 보이는 문제를 줄이기 위한 진단이다. 공식 서버 점수나 2025년
실제 성능을 의미하지 않는다.

## 실험에서 남기지 않은 것

### HCCN을 계속 키우는 방향

V1, V2, V3는 구조와 학습 epoch, penalty가 함께 바뀌었다. 원본 OOF와 seed별
train/validation log가 현재 Git에 없어서 한 tower나 gate의 효과를 분리할 수
없었다. HCCN은 기준선에 작은 residual을 붙이는 challenger로 남겼고, 최종 V14를
HCCN 단일 모델이라고 부르지 않았다.

### Pooled calibration을 nested 결과처럼 사용하기

세 시즌 held-out prediction에 fit한 calibration은 local 비교에서 개선을 보였다.
하지만 다음 시즌만 사용한 nested 진단은 개선되지 않았다. 최종 설명에서는 두 결과를
분리했다.

### Contextual history aggregate를 실제 feature로 쓰기

초기 HCCN branch에는 pitcher×count, pitcher×hand matchup 등의 aggregate spec이
정의되어 있었지만 HCCN training path에서 호출되는 것을 확인하지 못했다. 설계된
utility를 실제 입력으로 사용했다고 쓰지 않았다.

### TrackMan을 최종 feature로 포함하기

현재 코드와 결과 artifact에서 TrackMan feature가 최종 예측에 들어간 흔적을 찾지
못했다. 그래서 최종 파이프라인에서 제외했다.

## 최종 추론 흐름

현재 저장소에서 설명할 수 있는 V14 경로는 다음과 같다.

```text
train.csv
   │
   ├─ target season 이전 데이터로 pitcher / batter / team profile 생성
   ├─ prior 기반 shrinkage와 recent form 계산
   ├─ season별 feature cache 생성
   │
   ├─ CatBoost A/B/C + LightGBM temporal training
   │       └─ 12-model MH logit ensemble
   │
   ├─ regular-season TabM seed 1234 / 2345
   │
   ├─ V13을 기준으로 R-only residual direction blend
   │
   ├─ game_type × game_month affine calibration
   │
   └─ submission.csv
```

제출 ZIP 전체에는 Track A, HCCN, V11~V13이 앞단에 더 들어간다. 다만 ZIP과 HCCN
학습 artifact는 `.gitignore`로 제외되어 있다. 현재 Git clone만으로 ZIP 전체를
학습부터 다시 실행할 수 있다고 설명하지 않았다.

## 저장소 구조

```text
.
├── challengers/
│   └── v14_mh_profile_ensemble/
│       ├── features.py          # 행 단위 feature 생성
│       ├── profiles.py          # strict-past profile
│       ├── temporal_train.py    # temporal OOF CatBoost/LightGBM
│       ├── regular_train.py     # 정규시즌 실험
│       ├── tabm_screen.py       # TabM 후보 확인
│       ├── tabm_full_refit.py   # TabM seed별 refit
│       ├── residual_stack.py    # residual 비교 실험
│       └── select_v14.py        # direction blend/calibration/bootstrap
├── final/
│   ├── v14_results.json         # V13/V14 결과와 nested 진단
│   ├── v14_policy.json          # direction weight와 calibration
│   ├── package_verification.json # 제출 패키지 검증
│   └── SHA256SUMS.txt           # local ZIP checksum 기록
├── data/raw/                    # 원본 CSV, Git 제외
├── pyproject.toml
├── uv.lock
└── README.md
```

## 실행 방법

Python 3.11 이상과 `uv`를 사용한다.

```bash
uv sync --dev
```

원본 파일은 다음 위치에 둔다.

```text
data/raw/train.csv
data/raw/test.csv
data/raw/sample_submission.csv
```

V14 feature cache와 temporal prediction을 생성한다.

```bash
uv run python -m challengers.v14_mh_profile_ensemble.temporal_train --cache-only
uv run python -m challengers.v14_mh_profile_ensemble.temporal_train \
  --models cat_a,cat_b,cat_c,lgb --seeds 0
```

TabM 후보와 refit은 다음과 같이 실행한다.

```bash
uv run python -m challengers.v14_mh_profile_ensemble.tabm_screen \
  --valid-season 2024 --seed 1234
uv run python -m challengers.v14_mh_profile_ensemble.tabm_full_refit
```

`select_v14`는 MH/TabM output과 과거 V13 OOF
`challengers/v11_feature_rebuild/artifacts/main_screen_oof.parquet`가 필요하다.
이 artifact는 현재 Git에 없으므로 깨끗한 clone에서 마지막 selection 명령이 바로
실행되지는 않는다.

```bash
uv run python -m challengers.v14_mh_profile_ensemble.select_v14
```

## Implementation과 확인 범위

현재 저장소에서 재현 가능한 후속 모델:

- [`features.py`](challengers/v14_mh_profile_ensemble/features.py)
- [`profiles.py`](challengers/v14_mh_profile_ensemble/profiles.py)
- [`temporal_train.py`](challengers/v14_mh_profile_ensemble/temporal_train.py)
- [`select_v14.py`](challengers/v14_mh_profile_ensemble/select_v14.py)
- [`v14_results.json`](final/v14_results.json)

초기 HCCN source는 별도 작업 branch의
[`challengers/residual_hccn/`](https://github.com/taeg2/Konkuk_CS_Aimers/tree/jy_branch/challengers/residual_hccn)에
남아 있다. HCCN branch의 결과와 현재 V14 JSON을 같은 실험표로 합치지 않은 이유는
OOF 원본과 실행 조건이 모두 보존되어 있지 않기 때문이다.

이 저장소에서 마지막으로 확인한 것은 local temporal OOF와 제출 패키지 검증이다.
공식 leaderboard 점수와 실제 2025 정답 기반 성능은 저장소에 없다.
