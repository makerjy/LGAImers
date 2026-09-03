# LG Aimers 9기 × LG스포츠
# 투수 제구 성공 확률 예측

투구 직전까지 관측 가능한 경기 상황과 선수의 과거 이력을 이용해 각 투구의
`control_success` 확률을 예측한 프로젝트입니다. 이 저장소의 핵심은 단순히
정확도가 높은 분류기를 고르는 것이 아니라, 시간 순서를 보존한 검증과 확률
보정, 그리고 새로운 선수·결측·행 순서 변화에 견디는 추론 파이프라인을 함께
설계한 데 있습니다.

최종 로컬 비교에서는 V13 기준선 위에 V14 앙상블을 residual 방향으로 더하고
`game_type × game_month` 보정을 적용했습니다. 2022–2024 temporal OOF에서
V14 최종 후보의 Brier score는 **0.246189**였습니다. 이 값은 공식 대회 점수가
아니며, 아래의 평가 범위와 한계를 함께 읽어야 합니다.

## 프로젝트 개요

### 문제 정의

각 행은 한 번의 투구를 나타내며, 모델은 해당 투구가 `control_success = 1`이
될 확률을 출력합니다. 따라서 최종 목적함수는 임계값 기준의 단순 분류 정확도보다
확률 예측의 품질을 평가하는 Brier score에 가깝습니다.

실제 운영 시점에는 투구 결과를 알 수 없으므로, 다음 원칙을 지켰습니다.

- 검증 시즌의 정답이나 미래 시즌 정보를 피처 생성에 사용하지 않습니다.
- 선수·팀 프로파일은 예측 대상 시즌보다 이전 시즌으로만 만듭니다.
- 추론 시 `test.csv` 전체를 다시 집계하지 않고, 한 행과 고정된 lookup만 참조합니다.
- 표본이 적거나 처음 등장한 선수는 shrinkage, confidence, unknown bucket으로 처리합니다.

### 데이터와 예측 대상

현재 작업공간의 원본 CSV를 직접 확인한 결과는 다음과 같습니다.

| 항목 | 확인된 내용 |
|---|---:|
| 학습 데이터 | 1,475,092행 × 49열 |
| 테스트 데이터 | 5행 × 48열 |
| 학습 시즌 | 2019–2024 |
| 테스트 시즌 | 2025 |
| 타깃 | `control_success` 이진값 |
| 학습 타깃 평균 | 0.523766 |
| 정규시즌(`R`) 행 | 1,314,088 |
| `F` game_type 행 | 161,004 |

주요 입력은 투구 카운트(`balls_before`, `strikes_before`), 아웃·이닝·주자,
점수 차·기대 승률·레버리지 지수(`li`), 좌우 타자/투수, 팀, 선수별 as-of 누적
성적 및 최근 1/3/5경기 성적입니다. 원본의 모든 컬럼을 그대로 모델에 넣지 않고,
중복·외삽 위험·누수 가능성을 검토해 모델별 입력으로 재구성했습니다.

## 연구 질문과 의사결정 흐름

이 프로젝트의 모델 개발은 “더 복잡한 네트워크를 만들면 더 좋아질 것”이라는
가설에서 출발했지만, 최종적으로는 데이터의 시간 구조와 feature reliability를
검증하는 방향으로 이동했습니다.

1. 원본 경기 상황과 as-of 성적을 기반으로 CatBoost·LightGBM·Tabular NN 계열을
   비교했습니다.
2. 서로 다른 의미의 입력을 Entity / History / Context로 나누고, 강한 CatBoost
   기준선의 logit을 보정하는 Residual HCCN을 설계했습니다.
3. bounded residual, reliability gate, residual penalty를 넣어 기준선에서
   지나치게 멀어지는 보정을 제한했습니다.
4. 이후 동료 브랜치의 실험을 교차검토하면서, 모델 표현력보다 strict-past
   profile·shrinkage·recency weighting·이질적 tree ensemble이 더 직접적으로
   시간 이동 문제를 다룬다는 점에 주목했습니다.
5. 최종 V14에서는 V13을 기준점으로 유지하고, MH 앙상블과 TabM이 제시하는
   **방향**만 정규시즌에 제한적으로 결합한 뒤 `game_type × month` affine
   calibration을 적용했습니다.

이 README의 HCCN 분석은 대화에서 세운 설계 의도와 실제 코드·artifact를
분리해서 서술합니다. 프로젝트 문맥에서는 HCCN을 *Hierarchical Contextual
Cross Network*로 부르지만, 소스의 클래스명은 `ResidualHCCN`입니다. 초기
HCCN source-only 구현과 동료 모델의 비교 기준은 다음 branch snapshot입니다.

- [초기 HCCN source (`jy_branch`)](https://github.com/taeg2/Konkuk_CS_Aimers/tree/jy_branch)
  — commit `6fab3c0`
- [동료 tree ensemble (`jw_branch`)](https://github.com/taeg2/Konkuk_CS_Aimers/tree/jw_branch)
  — commit `b88d01e`
- [고성능 BSS 모델 (`mh_branch`)](https://github.com/taeg2/Konkuk_CS_Aimers/tree/mh_branch)
  — commit `19b1e3f`

`jy_branch`에는 HCCN 학습 소스와 테스트는 있지만 OOF parquet/metric CSV는
포함되어 있지 않습니다. 따라서 branch README의 수치는 “branch에 기록된 결과”,
현재 `final/`의 JSON 수치는 “이 저장소에서 직접 확인한 결과”로 구분했습니다.

## 전처리와 feature engineering

### 1. 경기 상황 피처

현재 Git에 공개된 V14 피처 빌더는 다음과 같은 투구 직전 상황을 사용합니다.

- 달·요일·이닝
- 볼·스트라이크·아웃 카운트
- 투수 팀 기준 점수 차와 누적 득점
- 1/2/3루 주자 유무와 주자 수
- 홈 승리 기대값과 `li`
- `count_state`, `count_diff`, `is_3ball`, `is_2strike`
- 절대 점수 차, 늦은 이닝 접전 여부, 투수·타자 same-hand 여부

원본에서 완전히 중복되는 `away_win_expectancy`, `run_top_before`,
`run_bot_before` 등은 V14 입력에서 직접 사용하지 않고 이미 포함된 정보로
표현했습니다. `season`과 raw 선수 ID도 기본 V14 모델 입력에서는 제외했습니다.
2025년을 학습하지 않은 상태에서 시즌을 트리의 연속값으로 넣는 외삽을 피하고,
ID 자체의 암기 대신 과거 프로파일과 unknown 처리를 사용하기 위해서입니다.

### 2. as-of 성적의 신뢰도 조정

누적 성공률을 그대로 사용하면 표본이 적은 선수의 극단적인 비율이 과대 반영될
수 있습니다. V14는 학습 히스토리에서 계산한 prior `p`와 표본 수 `n`을 이용해
다음과 같이 shrinkage합니다.

```text
shrunk_rate = (observed_rate × n + prior × k) / (n + k)
```

V14 기본값은 `k = 150`이며, 표본 신뢰도는 다음과 같이 포화시켰습니다.

```text
confidence = n / (n + 500)
```

투수·타자의 성공률/중간 결과 비율, 볼·스트라이크 비율, 투수 구종 비율에 이
방식을 적용합니다. 최근 1/3/5경기 성공률과 중간 결과율, 최근 폼과 통산 성적의
차이, 투수-타자 matchup 및 구종 조합도 파생합니다.

### 3. strict-past 선수·팀 프로파일

`profiles.py`는 예측 대상 시즌보다 이전 시즌만 골라 다음 lookup을 생성합니다.

- 투수 프로파일: 전체 성적, 시즌 수, 좌우 타자 상대, 유리/불리 카운트,
  2스트라이크·3볼, high-LI, 초반·후반 이닝, `R/F` split,
  이닝·LI·카운트 사용 패턴
- 타자 프로파일: 전체 성적, 좌우 투수 상대, platoon 차이, 최근 시즌,
  `R/F`·카운트·LI·이닝 패턴
- 투수팀·타자팀 프로파일: 과거 성공률의 shrinkage 값

프로파일 내부의 기본 entity shrinkage에는 `k=250`, split effect에는 `k=250`,
직전 시즌 effect에는 `k=150`, 팀 프로파일에는 `k=400`이 사용됩니다. 이렇게
계산된 표를 numpy 기반 ID lookup으로 저장해 pandas 버전이나 행 배치에 따른
차이를 줄였습니다.

### 4. 시즌 한정 폼과 결측 처리

`asof_*` 누적값에서 과거 시즌의 누적 성공 수·시도 수를 빼면 현재 시즌 한정
성적을 복원할 수 있습니다. 다만 표본이 20회 미만이면 안정적인 시즌 통계로
취급하지 않습니다.

결측은 모델별로 다르게 처리합니다.

- CatBoost/LightGBM: 결측을 모델이 처리하도록 유지합니다.
- TabM: 학습 fold의 중앙값으로 수치형 결측을 대체한 뒤 평균·표준편차로
  표준화합니다.
- 범주형: fold 학습 vocabulary에 없는 값은 unknown bucket으로 보냅니다.
- HCCN: 수치형 변환에 결측 indicator를 추가하고, 범주형은 `0=missing`,
  `1=unknown`, `2 이상=학습 vocabulary`로 인코딩합니다.

`build_features()` 자체에는 행 간 `groupby`, `rolling`, `lag`, rank가 없습니다.
필요한 집계는 학습 시점의 strict-past 프로파일로 미리 만들며, 테스트 추론은
각 행과 고정 lookup만 봅니다. 최종 패키지 검증에서 행 순서를 섞었을 때 최대
절대 예측 차이는 `0.0`이었습니다.

## HCCN 구조

### HCCN의 저장 위치와 해석 범위

초기 HCCN의 전체 source는 현재 저장소의 과거 작업을 정리한
[`jy_branch`](https://github.com/taeg2/Konkuk_CS_Aimers/tree/jy_branch)에 보존되어
있습니다. 핵심 구현은
[`challengers/residual_hccn/`](https://github.com/taeg2/Konkuk_CS_Aimers/tree/jy_branch/challengers/residual_hccn),
공통 feature 생성은
[`src/features/`](https://github.com/taeg2/Konkuk_CS_Aimers/tree/jy_branch/src/features),
실행·검증은
[`challengers/residual_hccn/train.py`](https://github.com/taeg2/Konkuk_CS_Aimers/blob/jy_branch/challengers/residual_hccn/train.py)와
[`evaluate.py`](https://github.com/taeg2/Konkuk_CS_Aimers/blob/jy_branch/challengers/residual_hccn/evaluate.py)에
있습니다.

현재 `makerjy/LGAImers`의 최신 V14 source tree에는 HCCN 학습 모듈을 복사해 두지
않았지만, 로컬 최종 제출 파일 `final/v14_submit.zip` 내부에는 다음 구현과
checkpoint가 보존되어 있습니다.

```text
src/residual_hccn/          # V1
src/residual_hccn_v2/       # magnitude-gated V2
src/residual_hccn_v3/       # model-state tower를 추가한 V3
src/residual_hccn_v4_selective/  # 선택적 router/보정 실험
model/track_v1/final_residual_hccn.pt
model/track_v2/final_hccn_v2.pt
model/track_v3/final_hccn_v3.pt
```

따라서 아래 HCCN 설명은 `jy_branch`의 공개 source와 최종 ZIP 내부의 source·
checkpoint·policy를 서로 대조해 정리한 것입니다. ZIP은 모델 가중치 용량 때문에
`.gitignore`로 제외되어 있어, 현재 저장소의 깨끗한 clone만으로 HCCN 학습을
재현할 수 있다고 주장하지 않습니다.

### 목적: 전체 재분류기가 아닌 기준 모델의 residual correction

HCCN은 입력만으로 처음부터 확률을 다시 학습하는 구조가 아니라, 기준 모델의
확률 `p_base`를 logit으로 바꾼 뒤 작은 보정값을 학습합니다.

```text
base_logit = logit(p_base)
final_logit = base_logit + delta_logit
prediction = sigmoid(final_logit)
```

이 설계의 의도는 다음과 같습니다.

- 이미 강한 CatBoost/앙상블 기준선의 안정성을 보존합니다.
- 선수·상황별로 기준선이 놓치는 방향만 학습합니다.
- `tanh`와 residual L2 penalty로 보정 폭을 제한합니다.
- 데이터가 부족한 상황에서는 reliability gate와 magnitude gate가 보정값을
  작게 만들 수 있습니다.

### 도입 배경과 실제 구현 계약

초기 HCCN은 “경기 상황과 선수 이력을 하나의 평면 벡터로 넣는 대신, 의미가 다른
정보를 계층적으로 나누고 기준 모델의 부족한 부분만 보정하면 확률 품질이 좋아질
것”이라는 가설을 검증하기 위한 challenger였습니다. 여기서 계층은 데이터에
관측된 공식 계층이라기보다, `entity`(누가 던지고 치는가), `history`(과거에
어떻게 했는가), `context`(지금 어떤 상황인가)를 모델링 목적에 따라 나눈
inductive bias입니다.

실제 `ResidualHCCN`의 입력·학습 계약은 다음과 같습니다.

| 경로 | 실제 구현에서 확인된 내용 | 설계 의도 |
|---|---|---|
| 기준 예측 | fold-safe CatBoost OOF 확률을 logit으로 변환 | 강한 기준선의 확률 수준을 보존 |
| Entity | 투수·타자·팀·상황 범주형 embedding | 반복되는 entity 조합의 표현 학습 |
| History | as-of 성적, 최근 폼, 표본 수·결측·shrinkage 관련 수치 | 선수 이력의 규모와 신뢰도 함께 전달 |
| Context | count, base, inning, score, LI, pressure 등 수치와 범주 | 동일 선수라도 상황별 잔차를 분리 |
| Gate | count/base, 표본 수·결측·LI 관련 입력 | 어떤 expert의 보정을 신뢰할지 결정 |
| 출력 | 네 expert residual의 gate 가중합을 bounded correction으로 변환 | 전체 재분류보다 작은 logit 보정 |

중요한 구현상의 한계도 있습니다. `jy_branch`의 학습 코드는 `x_all_num` 전체를
직접 사용하지 않고 history/context/entity/gate 배열을 명시적으로 받습니다. 또한
history는 순서가 있는 시계열 encoder가 아니라 집계된 snapshot 벡터입니다. 따라서
이 모델을 “선수 이력을 계층적 시퀀스로 학습한 모델”이라고 설명하는 것은 실제
구현보다 강한 표현이며, 정확히는 **집계 이력·현재 상황·entity embedding을
분리한 residual network**입니다.

### 상세 아키텍처

V2/V3 checkpoint의 구조와 설정은 다음과 같습니다. V1은 같은 큰 흐름의 초기
버전이며 model-state tower와 magnitude gate가 없습니다.

```text
row-level / as-of / model-state features
                    |
        +-----------+------------+
        |           |            |
  Entity Tower  History Tower  Context Tower
  embeddings    Numeric MLP     cat embeddings
  pitcher/      166 -> 128      + numeric MLP
  batter/team   -> 64            64 -> 96 -> 64
        |           |            |
        +-----------+------------+
                    |
         optional Model-State Tower
                 48 -> 64 -> 32 (V3)
                    |
              concatenate shared
                representation
             /                    \
  Low-rank Cross Network          Deep MLP
  2 layers, rank 32             256 -> 128 -> 64
             \                    /
              concatenate features
                    |
        4 residual experts, hidden 64
  pitcher / matchup / count_pressure / prior
                    |
       Reliability Gate: numeric + categorical
              -> hidden 32 -> softmax(4)
                    |
       optional Magnitude Gate: 48 -> 32 -> sigmoid(alpha)
                    |
  weighted delta -> tanh bound -> residual scale
                    |
        base_logit + delta_logit -> sigmoid
```

구현상 핵심은 다음과 같습니다.

- Entity/Context/Gate 입력은 각 범주별 embedding을 연결합니다.
- History와 Context는 별도 tower로 처리해 선수 이력과 경기 상황을 섞기 전
  표현을 분리합니다.
- Cross branch는 다음 형태의 low-rank interaction을 두 번 적용합니다.
  `x_(l+1) = x_0 * W_up(GELU(W_down(x_l))) + x_l`
- Deep branch는 SiLU·LayerNorm·dropout을 포함한 MLP입니다.
- 네 개 expert가 같은 공유 표현에서 서로 다른 residual 방향을 출력합니다.
- Reliability Gate가 pitcher/matchup/count-pressure/prior expert의 가중치를
  행별로 softmax 결정합니다.
- V2/V3의 Magnitude Gate가 최종 보정 크기 `alpha`를 조절합니다. V2는 model
  state를 사용하지 않고, V3는 다른 모델의 예측 상태를 별도 tower로 사용합니다.

보정식은 checkpoint 코드에 다음과 같이 구현되어 있습니다.

```text
delta_bounded = max_delta × tanh(delta_raw)
delta_logit = residual_scale × alpha × delta_bounded
final_logit = base_logit + delta_logit
```

| 버전 | checkpoint 후보 | 확인된 차이 |
|---|---|---|
| HCCN V1 | `residual_hccn_max_delta_05` | `max_delta=0.5`, cross 2층/rank 32, residual scale 초기값 0.1 |
| HCCN V2 | `magnitude_gated_hccn` | `max_delta=0.4`, magnitude gate 사용, `alpha` 초기값 0.65, model-state tower 미사용 |
| HCCN V3 | `v3_b_model_state_magnitude` | V2 구조에 model-state tower 사용, 나머지 핵심 설정은 V2와 동일 |

V1의 loss 설정은 BCE + `0.25 × Brier` + `0.001 × residual L2`이고, V2/V3는
여기에 `0.005 × residual L2`와 `0.01 × anchor loss`를 사용합니다. 즉 HCCN의
학습 목표도 분류 손실만 최소화하는 것이 아니라 기준선에서 필요 이상으로 멀어지지
않는 확률 보정을 지향합니다.

### 초기 버전의 규모와 실험 조건

로컬에 보존된 checkpoint metadata를 기준으로 보면 V1은 history 52개, context
27개, gate 10개의 raw 수치 컬럼에 결측 indicator를 더해 각각 104, 54, 20개로
확장했고, 11개 entity categorical 입력을 사용했습니다. V1 parameter count는
`271,461`, 최종 refit epoch는 `2`였습니다. V2는 magnitude gate와 설정 변경으로
`282,278` parameters·12 epochs, V3는 model-state tower까지 추가해 `308,230`
parameters·15 epochs로 기록되어 있습니다.

이 수치는 버전별 학습 예산과 구조가 동시에 바뀐 metadata입니다. 따라서 V1/V2/V3
차이를 순수하게 “tower 하나를 추가한 효과” 또는 “파라미터가 많아서 생긴 과적합”으로
해석할 수 있는 controlled ablation은 아닙니다. 현재 저장소에는 각 버전의 원본 OOF와
seed별 train/validation 로그가 남아 있지 않으므로, 여기서는 구조와 학습 조건을
재현 가능한 사실로 제시하고 원인을 단정하지 않습니다.

### HCCN이 최종에서 맡은 위치

제출 ZIP의 `script.py`를 기준으로 보면 HCCN은 다음 계층의 일부입니다.

- Track A의 CatBoost·LightGBM·Tabular NN·prior expert가 기준 예측을 만듭니다.
- HCCN V1/V2/V3가 각각 기준 예측에 residual correction을 적용합니다.
- V4/V5 selective router가 기준선과 challenger 중 어느 쪽을 사용할지 선택하고
  보정합니다.
- V11/V12/V13의 lookup·season correction을 거친 결과가 V14의 base가 됩니다.
- 따라서 V14 최종 예측은 HCCN 단일 모델의 출력이라고 부르기보다, HCCN 계열을
  포함한 이전 스택 위에 V14가 추가된 구조라고 설명하는 것이 정확합니다.

### 초기 HCCN을 주력으로 확장하지 않은 이유

HCCN을 최종 주력으로 확장하지 않고 후속 설계로 이동한 이유를 단순히 “신경망이
성능이 낮아서”라고
정리할 수는 없습니다. 확인 가능한 소스와 artifact를 기준으로는 다음과 같은
구조적 가설이 더 타당합니다.

1. **계층은 유용한 가정이지만 데이터의 실제 계층은 아니었습니다.** Entity,
   history, context를 분리한 것은 해석 가능한 설계였지만 `count_state`와
   `base_state`처럼 여러 경로에 반복되는 변수가 있고, 선수 이력은 sequence가
   아닌 집계 snapshot이었습니다. `contextual_history.py`에는 pitcher×count,
   pitcher×hand matchup 등 더 세분화된 aggregate spec이 정의되어 있지만,
   `jy_branch`의 HCCN 학습 경로에서 실제 호출되는 것은 확인되지 않았습니다.
   설계 문서에 존재하는 feature와 모델에 실제 공급된 feature를 구분해야 하는
   이유입니다.
2. **capacity가 병목이었다는 직접 증거는 없습니다.** 원본에는 약 147만 투구,
   투수 792명, 타자 830명이 있고, 투수별 행 수는 극도로 불균형합니다. 로컬
   확인 기준으로 10행 미만 투수는 5명, 30행 미만은 50명, 100행 미만은
   143명입니다. 이는 raw ID embedding과 복잡한 tower가 cold-start에서
   얻을 수 있는 신호를 제한할 수 있음을 시사하지만, HCCN의 train/validation
   gap 로그가 tracked되어 있지 않아 과적합을 사실로 단정하지 않았습니다.
3. **residual의 신호가 작고 불안정할 가능성이 있었습니다.** `max_delta`,
   residual scale, L2/anchor penalty로 보정 폭을 제한한 것은 안전한 설계였지만,
   잔차 자체가 강한 일반화 신호인지 확인하려면 seed별 residual 분포와 fold별
   개선이 필요합니다. 해당 원본 report가 현재 branch에 없으므로 “잔차가
   noise였다”고 결론내리는 대신, residual challenger를 최종 모델이 아닌
   검증 대상으로 남겼습니다.
4. **history reliability 문제는 구조보다 feature 설계의 문제에 가까웠습니다.**
   HCCN도 표본 수·missing flag·smoothed rate를 사용했지만, raw player embedding과
   집계 history가 함께 들어가는 구조였습니다. 후속 V14는 strict-past lookup,
   prior 기반 shrinkage, season-to-date와 recency weighting으로 신뢰도를 더
   명시적으로 제어했습니다.
5. **시간에 따른 regime shift를 별도 설계해야 했습니다.** 원본에서 정규시즌
   성공률은 2022년 `R=0.503691`, 2023년 `R=0.503118`, 2024년 `R=0.489707`로
   움직였고, `F`는 2022년 `0.708749`에서 2023년 `0.472904`, 2024년 `0.459280`로
   크게 달라졌습니다. 초기 HCCN의 2년 training window와 최종 2022–2024
   refit에는 이 변화를 별도로 분리해 다루는 장치가 제한적이었습니다.
6. **검증은 좋았지만 재현성은 제한적이었습니다.** HCCN 코드는 temporal fold,
   validation Brier early stopping, pitcher-cluster bootstrap을 고려했습니다.
   반면 branch에 OOF parquet와 metric CSV가 보존되지 않아, HCCN과 기준선의
   동일 조건 비교 및 seed 안정성을 현재 clone에서 다시 계산할 수 없습니다.

즉 초기 HCCN의 교훈은 “표현력을 더 키우자”가 아니라, **검증 가능한 과거 이력과
시간 이동에 강한 feature를 먼저 만들고, 복잡한 모델은 그 위에서 제한적으로
사용하자**는 방향 전환이었습니다.

## 현재/최종 후보 모델과 설계 의도

### V14 MH profile ensemble

현재 Git에 공개된 `temporal_train.py`는 세 가지 CatBoost 설정과 LightGBM을
지원합니다. 최종 ZIP의 `model/mh/model.pkl`에는 이 네 모델을 세 seed로 구성한
12개 모델이 기록되어 있습니다.

| 모델 | 확인된 설정 | 의도 |
|---|---|---|
| CatBoost A | 600 iterations, lr 0.02, depth 7, L2 25, min leaf 100 | 중간 깊이의 안정적인 범주형 상호작용 |
| CatBoost B | 600 iterations, lr 0.035, depth 5, L2 3, min leaf 100 | 더 얕고 다른 bias를 갖는 diversity 모델 |
| CatBoost C | 1,000 iterations, lr 0.02, depth 4, L2 8, min leaf 500 | 강한 regularization과 보수적 일반화 |
| LightGBM | 600 rounds, lr 0.01, 31 leaves, min leaf 1,500, feature/bagging fraction 0.7, L2 20 | 트리 구조와 학습 방식이 다른 diversity 모델 |

각 모델의 예측은 logit으로 바꾼 뒤 평균내고 sigmoid로 되돌립니다. 최종 패키지의
MH 입력은 117개 feature이며, 그중 6개는 categorical 입력입니다.

### TabM 후보

TabM은 MH feature를 입력으로 받아 정규시즌(`game_type=R`) 행만 학습한 보조
신경망입니다. 확인된 구조는 다음과 같습니다.

- `n_blocks=3`
- `d_block=256`
- `k=16`
- `arch_type="tabm-mini"`
- dropout `0.1`
- AdamW, learning rate `0.002`, weight decay `0.0003`
- 최종 refit seed `1234`, `2345`

각 fold에서 범주형 vocabulary와 unknown bucket을 학습하고, 수치형은 중앙값
대체와 평균·표준편차 표준화를 수행합니다. 최종 refit 코드는 2021–2024
정규시즌 데이터에 최근 시즌일수록 큰 가중치를 주고 seed 두 개를 학습합니다.
여러 seed를 평균내는 목적은 단일 신경망의 분산을 줄이고 MH 트리 앙상블과 다른
오류 방향을 확보하는 것입니다.

### V13 기준선과 V14 residual 방향 결합

V14는 V13을 기준점으로 유지하며, 각 challenger의 절대 예측값을 그대로 평균내지
않습니다. `select_v14.py`에 기록된 식은 다음과 같습니다.

```text
R = 1 if game_type == "R" else 0

p_robust = clip(
    p_v13
    + R × [
        w_mh    × (p_mh    - p_v13)
      + w_tabm1 × (p_tabm1 - p_v13)
      + w_tabm2 × (p_tabm2 - p_v13)
    ]
)
```

최종 정책에 저장된 방향 가중치는 다음과 같습니다.

| 방향 | 가중치 |
|---|---:|
| `p_mh_raw_blend - p_v13` | 0.290352 |
| `p_tabm_seed1234 - p_v13` | 0.268029 |
| `p_tabm_seed2345 - p_v13` | 0.245370 |

이후 `game_type × game_month`별로 학습한 fixed affine calibration을 적용합니다.
보정식의 개별 계수는 [v14_policy.json](final/v14_policy.json)에 그대로 보존되어
있습니다.

### 동료 고성능 모델과의 구조 비교

초기 HCCN의 방향을 평가할 때는 신경망의 복잡도만 보지 않고, 실제 점수 기록이
남은 `jw_branch`와 `mh_branch`의 feature·validation·ensemble 설계를 함께
비교했습니다. 두 branch의 점수는 서로 다른 시점과 지표로 기록되어 있으므로
현재 V14의 Brier 표와 직접 순위를 매기지 않았습니다.

| 항목 | 초기 HCCN (`jy_branch`) | `mh_branch` / `jw_branch` | 후속 설계에 미친 영향 |
|---|---|---|---|
| 기준 모델 | CatBoost OOF logit 위 residual correction | CatBoost·LightGBM 여러 모델의 이질적 앙상블 | 단일 복잡 모델보다 모델군의 오류 다양성을 활용 |
| feature | Entity / History / Context tower, gate 입력 | strict-past profile, shrinkage, row-independent feature | 이력을 sequence가 아닌 검증 가능한 lookup으로 단순화 |
| 선수 표현 | raw pitcher/batter ID embedding과 집계 이력 | `mh`: raw ID 제외, `jw`: profile 중심·ID 제외 | cold-start와 ID 암기 위험을 직접 관리 |
| 시간 가중 | HCCN config의 2년 window, decay `0.7` | `mh`: 4년·`1/.75/.5/.25`, `jw`: 5년·season decay `0.7` 및 월별 recency | temporal shift를 모델 구조보다 학습 표본 설계로 반영 |
| 앙상블 규모 | 네 residual expert와 reliability gate | `mh`: 12개 모델, `jw`: 7 model type × 6 seed = 42개 | expert 내부 혼합과 모델 family 간 diversity를 구분 |
| calibration | Brier 중심 loss와 residual/anchor penalty | `mh`: BSS 중심 평가와 logit 평균·shift 기록 | ranking보다 확률 수준과 calibration을 최종 기준으로 채택 |
| 검증 | 2022–2024 temporal fold, Brier early stopping, cluster bootstrap | `jw`: 다년 temporal AUC 평균 기록, random split의 낙관성도 문서화 | validation protocol을 모델의 일부로 취급 |
| 기록된 성능 | branch README에 Brier `0.247123` 등 기록 | `mh` 문서: LB `1025`, `jw` 문서: LB `961`, 다년 평균 AUC `0.55376` | 서로 다른 프로토콜임을 표시하고 숫자 자체보다 설계를 채택 |

이 비교에서 얻은 핵심은 HCCN의 tower를 그대로 확장하는 것이 아니라, `mh`의
행 독립성·strict-past profile·4년 recency와 `jw`의 강한 regularization·다년
temporal holdout·여러 tree family를 후속 V14에 흡수하는 것이었습니다. 현재 V14의
`MH profile ensemble`은 이 판단을 코드로 옮긴 결과이고, TabM은 동일 feature 위에서
오류 방향의 추가 다양성을 확인하기 위한 보조 후보입니다.

## Validation / OOF 전략

### Temporal OOF

검증 시즌은 2022, 2023, 2024이며 현재 코드의 fold는 다음과 같습니다.

| validation season | training seasons | validation rows |
|---:|---|---:|
| 2022 | 2020–2021 | 247,472 |
| 2023 | 2020–2022 | 245,525 |
| 2024 | 2020–2023 | 253,507 |

각 fold는 validation season 이전의 최대 4개 시즌을 사용하며, 가장 최근 시즌에
큰 recency weight를 줍니다. 프로파일은 각 target season보다 과거 데이터로만
생성합니다. 이 구조는 random split보다 2025 테스트 상황에 가까운 시간 이동을
검사하기 위한 것입니다.

### Calibration과 nested diagnostic의 구분

V14의 로컬 1500-point equivalent crossing은 세 시즌의 held-out prediction을
모아 방향 가중치와 월별 calibration을 적합한 exploratory 결과입니다. 따라서
일반적인 의미의 완전히 nested한 최종 일반화 성능으로 해석하면 안 됩니다.

별도 nested diagnostic은 다음 프로토콜로 실행되었습니다.

- 2023 검증: 2022로 방향과 보정 학습
- 2024 검증: 2022–2023으로 방향과 보정 학습
- 2022는 V13 유지

이 nested 결과는 오히려 V13보다 나빴습니다. 이 사실을 숨기지 않고 모델의
시간 이동 안정성 한계로 기록합니다.

### 평가 지표

- **Brier score**: `mean((y - p)^2)`, 낮을수록 좋습니다.
- **Standard BSS**: `1 - Brier / climatology_Brier`, 높을수록 좋습니다.
- **Official-like score**: 저장소의 로컬 비교용 `max(0, 100000 × Standard BSS)`.
  공식 서버 점수가 아닙니다.
- **AUROC**: 일부 초기 탐색 노트북과 패키지 내부 실험 코드에는 존재하지만,
  V14 최종 결과 JSON에는 저장되어 있지 않습니다. 서로 다른 검증 프로토콜의
  AUROC를 최종 표에 섞지 않았습니다.

## 실험 결과

### 최종 V14 비교

아래 값은 [v14_results.json](final/v14_results.json)의 `metrics`를 그대로
옮긴 2022–2024 pooled temporal OOF 결과입니다.

| 모델 | Brier ↓ | Standard BSS ↑ | Official-like |
|---|---:|---:|---:|
| V13 baseline | 0.246641 | 0.013345 | 1334.49 |
| V14 uncalibrated | 0.246279 | 0.014791 | 1479.12 |
| **V14 month-calibrated final** | **0.246189** | **0.015150** | **1515.04** |

V14 final은 V13보다 Brier가 `0.000451` 낮았습니다. 시즌별 결과도 다음과
같습니다. 음수 delta는 V13 대비 Brier 감소를 의미합니다.

| 검증 시즌 | V13 Brier | V14 final Brier | delta |
|---:|---:|---:|---:|
| 2022 | 0.243444 | 0.242933 | -0.000510 |
| 2023 | 0.248728 | 0.248079 | -0.000649 |
| 2024 | 0.247739 | 0.247537 | -0.000202 |

### HCCN 및 과거 후보에 대한 확인 가능한 기록

다음 값들은 V14의 최종 비교표와 동일한 산출물이라고 단정하지 않고, 제출 ZIP의
policy/manifest에 기록된 과거 후보의 참고값으로 표시합니다.

| 기록 | Brier | Standard BSS | Official-like | 근거 |
|---|---:|---:|---:|---|
| HCCN V1 anchor | 0.247123 | 미기록 | 미기록 | ZIP `model/v4/v4_policy.json` |
| HCCN V4 best preselect | 0.247036 | 미기록 | 미기록 | ZIP `model/v4/v4_policy.json` |
| HCCN V11 feature rebuild | 0.246938 | 미기록 | 1215.46 | ZIP `model/MANIFEST.md` |
| V13 baseline | 0.246641 | 0.013345 | 1334.49 | `final/v14_results.json` |

HCCN V4 policy는 HCCN V1 anchor보다 낮은 Brier의 best preselect 후보를
`CHAMPION_CANDIDATE`로 기록했습니다. 다만 최종 제출의 상위 계층은 V11/V12/V13
및 V14까지 이어지므로, 이를 곧바로 “HCCN V4가 전체 최종 모델”이라고 부르지
않았습니다.

`jy_branch/README.md`에는 HCCN v1에 대해 Brier `0.247123`, branch 표기 BSS `1141.43`,
2024 Brier `0.248198`, AUROC `0.545125`가 기록되어 있습니다. 이 값은 branch의
기록으로서 초기 시도의 방향성을 보여주는 참고값이며, 현재 V14 표와 같은 OOF
생성·calibration·평가 프로토콜이라고 확인할 수 없습니다. 특히 branch의
`artifacts/`에는 실제 OOF parquet나 metric CSV 대신 placeholder만 남아 있어,
기준 CatBoost와 HCCN 사이의 동일 조건 개선폭을 이 clone에서 독립적으로 재계산할
수 없습니다. 대화 중 제시되었던 별도의 기준선 수치도 tracked artifact에서
확인되지 않아 본문 성능표에는 넣지 않았습니다.

### 불확실성 확인

V14 final과 V13의 차이를 `pitcher_id` cluster 단위로 3,000회 bootstrap했습니다.

| 항목 | 값 |
|---|---:|
| 관측 Brier delta (V14 − V13) | -0.000451 |
| 95% CI | [-0.000567, -0.000352] |
| delta < 0 비율 | 1.0 |

이는 같은 투수의 여러 투구가 독립 표본처럼 보이는 문제를 완화하려는 진단입니다.
통계적 유의성의 최종 증명이나 대회 서버 점수를 의미하지는 않습니다.

## 잘 된 점과 실패·한계

### 잘 된 점

- V13 대비 V14 final Brier가 세 validation season 모두 개선되었습니다.
- pitcher cluster bootstrap의 95% CI가 0 아래에 위치했습니다.
- row-order independence 검사에서 최대 절대 차이 `0.0`을 확인했습니다.
- 미지 선수·결측 입력과 `IndexError` 회귀 시나리오를 통과했습니다.
- ZIP CRC integrity와 10분 이내 추론 검사를 통과했습니다.
- TrackMan 외부 파일 없이 제출 패키지가 동작하도록 구성했습니다.

### 실패·한계

- 월별 calibration을 pooled held-out prediction에 적합한 로컬 1515 결과는
  탐색적입니다. 다음 시즌만으로 nested하게 적합한 calibration은 V13보다
  약했습니다.
- 현재 Git 소스에는 HCCN 학습 코드가 없고, HCCN은 최종 제출 ZIP 내부에만
  보존되어 있습니다. HCCN 구조의 재현성과 코드 리뷰 가능성이 낮습니다.
- `select_v14.py`는 생성된 MH/TabM 산출물과 과거 V13 OOF 파일
  `challengers/v11_feature_rebuild/artifacts/main_screen_oof.parquet`를 요구하며,
  해당 V13 artifact는 현재 Git tree에 없습니다. 따라서 깨끗한 clone에서
  선택 단계를 즉시 재현할 수 없습니다.
- 최종 ZIP은 여러 세대의 모델과 가중치를 함께 포함한 제출 패키지이며, 모든
  과거 후보의 동일 조건 OOF 표가 tracked artifact로 남아 있지는 않습니다.
- TrackMan branch는 선수 crosswalk를 확보하지 못했고, permutation/negative
  control 검증에서 최종 채택할 이득을 확인하지 못해 제외했습니다.
- 공식 대회 서버 점수와 미래 시즌 검증 결과는 이 저장소에서 확인되지 않습니다.

## 연구 고찰과 개선 방향

이 프로젝트에서 가장 중요한 결과는 특정 모델 하나의 숫자가 아니라, 어떤 가설을
버리고 어떤 신호를 남겼는지입니다.

- **Residual 접근은 타당한 출발점이었습니다.** 기준 모델의 확률을 보존하면서
  추가 정보를 작은 방향 보정으로 제한했기 때문에, 복잡한 challenger를 안전하게
  비교할 수 있었습니다. 다만 HCCN이 최종 V14의 단독 출력이 된 것은 아니며,
  branch의 raw OOF 부재로 HCCN의 독립적인 일반화 우위를 확정할 수 없습니다.
- **표현력보다 feature reliability를 우선하게 되었습니다.** raw ID와 tower를
  늘리는 대신 strict-past profile, shrinkage, 표본 수, season-to-date, recency를
  명시적으로 설계했습니다. 이 변화는 “새로운 선수와 시간 이동에서도 무엇을
  신뢰할 것인가”라는 질문에 직접 답합니다.
- **expert mixture와 heterogeneous ensemble은 같은 diversity가 아닙니다.**
  HCCN의 네 expert는 하나의 공유 표현 안에서 residual 방향을 나누는 구조인 반면,
  `mh`/`jw`의 CatBoost·LightGBM 조합은 서로 다른 학습 편향과 feature 사용법을
  가진 모델 family를 결합합니다. 후속 V14에서는 후자의 오류 다양성을 더 직접적인
  개선 방향으로 채택했습니다.
- **calibration은 모델 선택의 부속 단계가 아니라 목적함수의 일부입니다.**
  AUROC가 순위만 평가하는 반면 Brier/BSS는 확률의 크기까지 평가합니다. 그래서
  V14는 절대 확률을 무작정 평균내지 않고 V13 대비 residual 방향을 제한된
  `R` 행에만 적용한 뒤 calibration을 별도로 검토했습니다.
- **validation protocol 자체가 모델의 일부입니다.** pooled held-out 결과에서
  개선된 calibration도 nested next-season 진단에서는 약해졌습니다. 따라서 현재
  1515 local 결과를 미래 시즌 성능으로 포장하지 않고, 다음 연구에서는 방향·보정
  정책을 각 검증 시즌 이전에만 fit하는 nested OOF를 기본 절차로 삼아야 합니다.

후속 개선의 우선순위는 (1) HCCN source와 OOF/seed 로그를 함께 보존해 재현성을
복구하고, (2) strict-past profile과 season-to-date의 rule 해석을 명시적으로
검증하고, (3) Brier 기반 nested calibration을 고정한 뒤, (4) compact neural
challenger와 tree ensemble의 residual diversity를 공정하게 비교하는 것입니다.

## 최종 추론 파이프라인

제출 ZIP의 `script.py`가 수행하는 전체 흐름은 다음과 같습니다.

```text
train.csv / test.csv
        |
        +-- Track A: CatBoost + LightGBM + Tabular NN + prior expert
        |       \-- logit stacker + probability calibration
        |
        +-- HCCN V1 / V2 / V3 residual challengers
        |       \-- V4/V5 selective router and calibration
        |
        +-- V11 lookup correction
        +-- V12 diversity + lookup correction
        +-- V13 season correction
        |
        +-- V14 MH: 12-model strict-past profile ensemble
        +-- V14 TabM: regular-season seed 1234 / 2345
        |       \-- R-only residual direction blend
        |
        +-- fixed game_type × month affine calibration
        |
        +-- p_final_v14 -> submission.csv
```

V14만 분리해서 설명하면 `V13 → MH/TabM 방향 결합 → 월별 calibration`이고,
ZIP 전체를 실행하면 그 앞에 Track A와 HCCN/V11–V13 계층이 더해집니다. 이
구분이 현재 소스와 실제 제출물의 차이를 가장 정확하게 설명합니다.

## 저장소 구조

```text
.
├── challengers/
│   └── v14_mh_profile_ensemble/
│       ├── features.py          # 행 독립 V14 피처 생성
│       ├── profiles.py          # strict-past 투수·타자·팀 프로파일
│       ├── temporal_train.py    # temporal OOF CatBoost/LightGBM
│       ├── regular_train.py     # 정규시즌 특화 트리 실험
│       ├── tabm_screen.py       # TabM 후보 screening
│       ├── tabm_full_refit.py   # TabM seed별 full refit
│       ├── residual_stack.py    # nested residual stack 실험
│       └── select_v14.py        # 방향 결합·calibration·bootstrap
├── final/
│   ├── v14_results.json         # V13/V14 성능·nested 진단
│   ├── v14_policy.json          # 방향 가중치·월별 보정 정책
│   ├── package_verification.json # 제출 패키지 검증 결과
│   └── SHA256SUMS.txt           # local ZIP checksum 기록
├── data/raw/                    # 원본 CSV; .gitignore 대상
├── pyproject.toml               # Python 의존성·실행 설정
├── uv.lock
└── README.md
```

현재 작업 트리의 `notebooks/`는 미추적 상태의 탐색용 파일이며 이 README 커밋에
포함하지 않았습니다. 해당 노트북은 이전 `src/` 구조를 참조하는 초기 실험 기록도
포함하므로, 그 안의 AUROC나 Brier를 현재 V14 표에 합치지 않았습니다.

### Git에서 제외된 산출물

원본 CSV, OOF parquet, 모델 가중치, 최종 제출 ZIP은 용량 때문에 Git에서
제외됩니다. 현재 로컬에는 `final/v14_oof.parquet`와
`final/v14_submit.zip`이 있으며, ZIP의 SHA-256은 다음과 같습니다.

```text
af76cc40cb2313bf2b9bc8cd6deda943be245adf6f9f5d74c1ef6b3561d33e09
```

## 실행 방법

### V14 연구 파이프라인

Python 3.11 이상과 `uv`를 사용합니다.

```bash
uv sync --dev
```

원본 파일을 다음 위치에 둡니다.

```text
data/raw/train.csv
data/raw/test.csv
data/raw/sample_submission.csv
```

피처 cache와 temporal OOF를 생성합니다.

```bash
uv run python -m challengers.v14_mh_profile_ensemble.temporal_train --cache-only
uv run python -m challengers.v14_mh_profile_ensemble.temporal_train \
  --models cat_a,cat_b,cat_c,lgb --seeds 0
```

TabM 후보 및 최종 refit을 실행합니다.

```bash
uv run python -m challengers.v14_mh_profile_ensemble.tabm_screen \
  --valid-season 2024 --seed 1234
uv run python -m challengers.v14_mh_profile_ensemble.tabm_full_refit
```

`select_v14`는 위에서 생성한 cache/prediction과 과거 V13 OOF artifact가 모두
있을 때 실행할 수 있습니다.

```bash
uv run python -m challengers.v14_mh_profile_ensemble.select_v14
```

현재 저장소에는 V13 OOF 원본과 HCCN 학습 소스가 없으므로, 위 마지막 명령을
깨끗한 clone에서 바로 실행할 수 있다는 의미는 아닙니다. 이는 결과 파일이
존재한다는 것과 재현 가능한 연구 저장소라는 것을 구분하기 위한 명시적인
제약입니다.

### 제출 패키지 실행

최종 ZIP은 로컬에만 존재합니다. 압축을 푼 뒤 패키지 내부 `script.py`를 실행하면
패키지에 포함된 Track A, HCCN, V11–V14 모델을 순서대로 호출해
`output/submission.csv`를 생성합니다. ZIP 무결성은 다음 명령으로 확인했습니다.

```bash
shasum -a 256 final/v14_submit.zip
unzip -tq final/v14_submit.zip
```

## 포트폴리오 관점의 핵심 기여

1. **시간 순서를 모델링 규칙으로 고정**: 단순 random split 대신 시즌 temporal
   OOF를 설계하고, strict-past profile로 선수 이력 누수를 차단했습니다.
2. **확률 예측 중심의 모델 선택**: Brier/BSS와 calibration을 기준으로 비교하고,
   classification ranking 지표가 최종 의사결정을 대체하지 않도록 했습니다.
3. **Residual HCCN 설계**: 기준 모델을 버리지 않고 entity·history·context를
   분리한 tower, low-rank cross, expert mixture, reliability/magnitude gate로
   작은 보정만 학습했습니다.
4. **모델 다양성을 방향 정보로 활용**: MH와 TabM의 예측값 자체가 아니라 V13
   대비 방향을 결합해 정규시즌에만 적용했습니다.
5. **실험의 실패도 기록**: nested calibration 약화, TrackMan crosswalk 실패,
   HCCN 소스와 artifact의 비공개 상태를 결과와 함께 명시했습니다.
6. **추론 안정성 검증**: unknown/missing 입력, 행 순서 shuffle,
   `IndexError`, ZIP integrity, 실행 시간까지 제출 단위로 점검했습니다.

## 근거 파일

- [V14 결과와 nested 진단](final/v14_results.json)
- [V14 방향 가중치와 월별 calibration 정책](final/v14_policy.json)
- [제출 패키지 검증 결과](final/package_verification.json)
- [V14 feature builder](challengers/v14_mh_profile_ensemble/features.py)
- [strict-past profile builder](challengers/v14_mh_profile_ensemble/profiles.py)
- [temporal training pipeline](challengers/v14_mh_profile_ensemble/temporal_train.py)
- [V14 selection/calibration/bootstrap](challengers/v14_mh_profile_ensemble/select_v14.py)
