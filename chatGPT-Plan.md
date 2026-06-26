# ChatGPT Advisor Report (2026-06-27)

이 답변은 `GUIDE.md`, `results.csv`, `docs/advisor_questions.md`, 전략 45/46 코드와 기존 실패 실험을 기준으로 작성했다. 현재 기준선은 전략 45 `ET EXPAND`이며, Public RMSE 2,094.9로 전략 28 대비 약 2점 개선된 상태다.

## Executive Summary

전략 45는 Final 1로 고정하는 것이 맞다. ExtraTrees는 단독 성능은 약하지만 기존 CB/LGB 계열과 오차 상관이 충분히 낮아 Ridge 스태킹에서 실제로 기여한 첫 추가 모델이다. 다만 개선폭이 작고 전략 46에서 RF/XGB/ET 변형 확장이 이미 악화됐으므로, 남은 실험은 "더 많은 모델 추가"가 아니라 "딱 다른 오차 표면을 만드는 소수 후보"에 제한해야 한다.

핵심 판단은 다음과 같다.

- Final 1: 전략 45 단일, alpha=1.0, seed=42, 현 제출 그대로 유지.
- Final 2: 전략 45의 과감한 변형보다 `45:28` 또는 `45:26` 블렌드가 더 합리적이다.
- 추가 모델 후보는 LightGBM `extra_trees=True`, HistGradientBoosting, KernelRidge/Spline-Ridge 정도만 우선순위가 있다.
- 1~3위와의 차이는 새 알고리즘보다 합성 데이터 생성식, 고가/대형 구간 처리, Public subset 운의 조합일 가능성이 높다.

## Q1. ExtraTrees 원리의 확장

ExtraTrees가 기여한 이유는 "성능 좋은 모델"이라서가 아니라, CB/LGB와 다른 방식으로 공간을 자르기 때문이다. RF는 최적 split을 찾기 때문에 결국 GB 트리와 비슷한 오차를 만들었고, XGB는 구조적으로 LGB와 너무 가까웠다. 따라서 다음 후보도 단독 OOF보다 상관, Ridge 계수, 구간별 잔차 상쇄를 먼저 봐야 한다.

### 1. LightGBM `extra_trees=True`

가장 먼저 시도할 후보는 LGBM의 extra trees 모드다.

추천 설정:

```python
LGBMRegressor(
    objective="regression",
    extra_trees=True,
    max_depth=3,
    num_leaves=31,
    feature_fraction=0.6,
    bagging_fraction=0.7,
    bagging_freq=1,
    min_child_samples=30,
    learning_rate=0.02,
    n_estimators=3000,
)
```

이 모델은 sklearn ExtraTrees와 완전히 같지는 않지만, 기존 LGBM보다 split randomness가 커서 현재 발견한 성공 원리와 가장 가깝다. 단, `num_leaves=110` 같은 기존 튜닝값을 그대로 쓰면 다시 LGB와 비슷해질 수 있으므로 더 얕고 랜덤하게 두는 편이 맞다.

판정 기준:

- 단독 OOF가 3,000~4,000대여도 탈락시키지 않는다.
- 전략 45의 10개 모델 예측과 평균 상관이 0.970 이하이면 후보로 둔다.
- 10모델 + LGB-ET를 Ridge에 넣었을 때 OOF가 최소 1점이라도 내려가거나, OOF 동일이어도 test 예측이 전략 45와 의미 있게 다르면 제출 후보가 될 수 있다.

### 2. HistGradientBoostingRegressor

`HistGradientBoostingRegressor`는 GB 계열이지만 sklearn 구현의 binning, regularization, leaf growth가 LGB와 다르다. XGB보다 기대값이 높은 이유는 강한 boosting 성능이 아니라 "어설프게 다른 예측 표면"을 만들 가능성 때문이다.

추천은 얕고 보수적인 설정이다.

```python
HistGradientBoostingRegressor(
    loss="squared_error",
    learning_rate=0.03,
    max_iter=800,
    max_leaf_nodes=15,
    min_samples_leaf=30,
    l2_regularization=1.0,
    random_state=42,
)
```

이 모델은 범주형 처리와 스케일링 방식에 민감하므로, LabelEncoded `Gu`, `Dong`을 그대로 쓰는 버전과 OneHot 버전을 둘 다 비교할 가치가 있다.

### 3. KernelRidge 또는 Spline-Ridge

KNN/SVR/Ridge-poly가 실패했기 때문에 커널/선형 계열의 기대값은 낮다. 그래도 하나만 더 본다면 `KernelRidge(RBF)`보다 `SplineTransformer + Ridge/BayesianRidge`를 먼저 추천한다. 이유는 샘플 1,969건에서 RBF는 거리 스케일과 범주형 인코딩에 크게 흔들리고, test 외삽에서 예측이 눌릴 수 있기 때문이다.

추천 구성:

- 연속형: `Exclusive_Area`, `Floor`, `Age`, `YearMonth_Seq`, `Distance_to_Subway`
- 범주형: `Gu`, `Dong` OneHot
- 변환: 연속형에 `SplineTransformer(n_knots=5, degree=3)`
- 모델: `Ridge(alpha=100~1000)` 또는 `BayesianRidge`

판정 기준은 동일하다. 단독 성능보다 전략 45와의 낮은 상관과 Ridge에서 양의 계수를 받는지가 중요하다.

### 하지 말아야 할 확장

RF, XGB, ET seed/depth 변형은 이미 전략 46에서 실익이 낮았다. 같은 계열을 더 늘리면 Ridge가 일부 무시하더라도 test 예측에는 잡음이 섞인다. 모델 수를 늘리는 실험은 지금부터 기대값보다 제출 리스크가 크다.

## Q2. 1~3위와의 격차를 줄이는 전략

1~3위가 썼을 가능성이 큰 접근은 세 가지다.

### 1. 합성 생성식에 더 가까운 구조

현재 파이프라인은 트리 모델이 생성식을 근사한다. 상위권은 더 직접적으로 다음 형태를 모델링했을 가능성이 있다.

```text
log(price) = gu_base + dong_base + area_effect + floor_effect + age_effect
             + brand_effect + month_trend + interaction + noise
```

우리는 평당가 축으로 `Target / Area`를 넣어 이 방향의 일부를 이미 잡았다. 추가 여지가 있다면 `Area` 외의 타겟 분해를 무작정 늘리는 것이 아니라, log additive 구조에서 residual을 분석하는 쪽이다.

실험 후보:

- `log(Target)`에 대해 OneHot `Gu/Dong` + spline 연속형 + interaction 소수만 넣은 Ridge.
- 이 모델의 OOF residual이 전략 45 residual과 다른지 확인.
- 단독 점수가 나쁘더라도 Ridge stack에서 +계수를 받는지 확인.

### 2. 고가/대형 구간의 손실 지배 대응

RMSE에서는 고가, 대형, 특정 구간의 소수 샘플이 순위를 크게 흔든다. 상위권은 전체 평균 RMSE보다 상위 오차 구간을 더 직접적으로 관리했을 수 있다.

다만 이미 sample weight, 구별 분리, local weight가 실패했으므로 다시 큰 구조를 만들기보다 전략 45 기준으로 "구간별 Final 2 리스크"를 보는 것이 낫다.

해야 할 분석:

- OOF에서 `Target` 상위 10%, `Exclusive_Area` 상위 10%, Gangnam/Seocho/Yongsan/Seongdong 구간의 45 vs 28 vs 26 RMSE 비교.
- Test 예측에서 같은 구간의 45-28, 45-26 차이 확인.
- ET 추가가 특정 구간에서만 예측을 올리거나 내리는지 확인.

만약 ET가 고가/대형에서만 과하게 낮추거나 올린다면, Final 2는 전략 45 단일이 아니라 45와 28/26 블렌드가 맞다.

### 3. Public subset에 더 잘 맞은 운 또는 더 강한 transductive tuning

현재 45와 28의 Public 차이는 약 2점이다. 이는 "압도적 새 구조"라기보다 Public subset에서 ET의 오차 상쇄가 조금 더 맞은 수준이다. 1~3위와의 격차도 비슷한 크기라면, 그들이 반드시 더 일반화 좋은 모델을 가졌다고 단정하면 안 된다.

Adversarial AUC 0.505는 Train/Test feature 분포가 거의 같다는 의미라 Private 방어에는 좋은 신호다. 반대로 말하면, AV로 test subset을 더 정교하게 갈라서 Public만 맞추는 길은 기대값이 낮다. 이제는 Public 1~3위 추격보다 Private에서 45가 흔들리지 않는지 확인하는 것이 더 중요하다.

## Q3. 전략 45 기반 Final Submission

### Final 1: 전략 45 단일

Final 1은 전략 45 그대로가 맞다.

- Public 최고점 2,094.9.
- OOF도 전략 28보다 소폭 개선.
- ET 추가는 단순 Public 과적합이라고 보기 어렵다.
- Adversarial Validation AUC=0.505라 Train/Test 분포 리스크도 낮다.

여기서 alpha=0.9, multi-seed, ET 변형, RF/XGB 추가로 희석하지 않는 편이 낫다. 이미 alpha sweep과 multi-seed가 Public에서 손해였고, 전략 46도 확장 악화를 보여줬다.

### Final 2: `45:28` 또는 `45:26` 블렌드

Final 2는 "전략 45의 다른 버전"보다 구조적 방어 블렌드가 낫다.

우선순위:

1. `45:28 = 70:30` 또는 `80:20`
2. `45:26 = 70:30` 또는 `80:20`
3. 전략 28 단일

판단 로직:

- 45와 28은 대부분 구조를 공유하지만 ET 유무가 다르다. `45:28`은 ET 기여를 유지하면서 ET가 Public subset에만 맞았을 위험을 줄인다.
- 26은 평당가 축이 빠진 더 보수적인 계열이다. `45:26`은 PL2+평당가+ET가 모두 같은 방향으로 편향됐을 때의 방어선이다.
- 전략 08 또는 PL2 배제 모델은 너무 멀다. Public 손해가 크고, 지금의 AV 결과상 그렇게까지 보수적으로 갈 근거는 약하다.

추천 최종안:

| 역할 | 제출 | 이유 |
|---|---|---|
| Final 1 | 전략 45 단일 | 최고 Public, OOF도 개선, AV 안정 |
| Final 2 | `45:28 = 70:30` | ET 리스크만 줄이는 가장 얕은 방어 |

`45:28`의 test 예측 차이가 너무 작아 방어 의미가 없으면 `45:26 = 80:20`을 2순위로 둔다. 반대로 `45:26`이 고가/대형 예측을 과하게 낮추면 쓰지 않는다.

## 남은 실험 우선순위

1. `45`, `28`, `26` 제출 파일을 같은 ID 순서로 모아 예측 차이 리포트 생성.
2. `45:28` 70:30, 80:20과 `45:26` 70:30, 80:20 생성.
3. 전체/구별/면적 상위 10%/예측가 상위 10%/신축·구축별 평균 변화 확인.
4. 제출권을 쓴다면 `45:28=70:30` 하나만 먼저 확인.
5. 추가 모델은 LGB-ET 하나를 최우선으로 하고, 개선이 없으면 모델 탐색 종료.

## 최종 권고

전략 45는 현재 구조에서 의미 있는 최선이다. 하지만 개선폭이 2점 수준이므로 "ET가 답이다"라고 확장하기보다, ET가 만든 작은 다양성을 보존하면서 Private에서 그 리스크를 줄이는 운영이 더 중요하다.

따라서 현재 기준 최종 제출은 다음 조합을 권한다.

```text
Final 1 = 전략 45 단일
Final 2 = 45:28 = 70:30
```

남은 시간은 새 모델을 많이 붙이는 데 쓰지 말고, `45:28`, `45:26` 블렌드가 어떤 구간의 예측을 얼마나 바꾸는지 수치화하는 데 쓰는 것이 가장 기대값이 높다.
