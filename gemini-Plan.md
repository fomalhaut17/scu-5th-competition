# Gemini Advisor Report (2026-06-27)

## Executive Summary

현재 상황은 **전략 45 (10모델 Ridge 스태킹)**를 통해 **Public RMSE 2,094.9 (4위)**를 달성하며 최상위권 경쟁에 본격적으로 진입한 상태입니다. 

우리가 발견한 핵심 돌파구는 **"성능이 아주 뛰어나진 않더라도, 기존 트리 모델(CatBoost/LGBM)과 상관관계가 낮아 오차를 상쇄해 줄 수 있는 알고리즘 다양성의 확보"**입니다. 단독 OOF 3,561인 ExtraTrees가 앙상블에서 -3점의 실질적 개선을 이끌어낸 것이 이를 증명합니다.

남은 기간 동안 1~3위와의 격차를 좁혀 우승을 노리기 위한 핵심 테마는 **1) 알고리즘 다양성의 한계 확장**, **2) 합성 데이터 생성 규칙(DGP)의 수학적 역공학**, **3) 고오차 타겟 영역(대형/고가) 집중 방어**입니다.

---

## Q1. ExtraTrees 원리의 확장: "다양성"을 극대화할 수 있는 대체 알고리즘

ExtraTrees가 유일하게 성공한 이유는 **"의사결정 나무의 분기점(Split)을 무작위로 선택하여 개별 트리의 편향은 유지하되, 앙상블을 통해 예측치 표면을 고도로 스무딩(Smoothing)하고 오차 패턴의 상관성을 낮췄기 때문"**입니다. 이와 동일한 효과(Smoothness + Low Correlation)를 낼 수 있는 미시도/확장 알고리즘을 제안합니다.

### 1. LightGBM의 ExtraTrees 모드 (`extra_trees=True`)
- **원리**: LightGBM 내부에 Scikit-learn의 ExtraTrees와 동일하게 무작위 분기를 수행하는 하이퍼파라미터가 존재합니다.
- **적용**: 기존 LGBM raw/log 모델 외에, `extra_trees=True`와 `colsample_bytree`, `subsample`을 극단적으로 낮춘(예: 0.5~0.6) **LGB-ET 모델**을 추가합니다. CB/LGBM의 학습 속도와 강력함을 유지하면서도 ExtraTrees 수준의 이질적인 오차 패턴을 생성합니다.

### 2. 데이터 크기(1,969건)를 저격하는 커널 모델 (Kernel Ridge & GPR)
- 일반적인 정형 데이터 대회와 달리, 본 대회는 **샘플 수(Train 1,969건)가 극도로 적습니다.** 이 크기에서는 딥러닝이나 복잡한 모델보다 **커널(Kernel) 기반 보간 모델**이 압도적인 성능과 완전히 독자적인 오차 패턴을 보입니다.
- **Kernel Ridge Regression (KRR)**: RBF 커널을 사용하여 피처 간 비선형 관계를 고차원으로 투사해 학습합니다. SVR보다 하이퍼파라미터 정규화가 안정적이며, 트리 모델의 계단식(Step-like) 예측과 달리 고도로 부드러운 연속적인 예측값을 제공합니다.
- **Gaussian Process Regression (GPR)**: 예측값뿐만 아니라 예측의 불확실성(Variance)까지 함께 계산합니다. GPR의 예측 평균값은 소량 데이터에서 매우 정교한 보간 성능을 보이며, Ridge 스태킹 시 트리 모델의 과적합을 상쇄하는 강력한 파트너가 됩니다.

### 3. Spline Feature + 선형 Regularization (비트리 비선형 보간)
- 단순히 다항식 피처(`Ridge-poly`)를 적용하는 것은 차원의 저주와 이상치 폭발로 인해 실패하기 쉽습니다.
- **대안**: Scikit-learn의 `SplineTransformer`를 사용하여 연속형 변수(`Area`, `YearMonth_Seq` 등)를 구간별 3차 스플라인 곡선 피처로 변환한 뒤, `BayesianRidge`나 `ElasticNet`으로 학습합니다.
- **효과**: 구간별 비선형 트렌드를 트리 모델과 전혀 다른 수학적 방식으로 학습하므로, 앙상블 다양성이 극대화됩니다.

---

## Q2. 1~3위와의 격차를 줄이는 전략 (Top-tier 분석)

현재 4위에서 1~3위로 올라서기 위해 상위 팀들이 썼을 것으로 판단되는 3가지 유력한 접근법입니다.

### 1. 합성 데이터 생성기(DGP)의 수학적 역공학 (가장 유력)
- 이 데이터는 100% 합성 데이터이므로, 특정 수학적 공식과 분포(예: Gaussian, Lognormal)에 의해 생성되었습니다.
- **상위권의 비밀**: 현실 부동산과 달리 합성 데이터는 **Multiplicative(곱셈적)** 관계가 극도로 뚜렷할 확률이 높습니다.
  - 예: $Price = BasePrice(Gu) \times AreaFactor(Area) \times FloorPremium(Floor) \times Trend(YearMonth) + Noise$
  - 만약 생성기가 이런 구조라면, 우리가 피처를 더하는 방식(`Area_x_Floor`, `Brand_x_Area`)으로는 완벽한 모사가 어렵습니다.
  - **해결책**: 모든 피처와 타겟을 Log 변환하여 **가법적(Additive) 구조**로 변환한 뒤 선형 회귀 계수를 역추적하거나, 곱셈 관계를 직접 피처(예: $Area \times Floor \times BrandScore$)로 인코딩하여 트리 모델의 학습 부담을 극도로 덜어주어야 합니다.

### 2. 고오차 타겟 영역(대형/고가)의 "분할 스태킹" 또는 "국소 가중치"
- 에러 분석 결과, 120㎡ 이상 대형(상위 11% 오차), 6~8억 고가, 강남/성동/용산구의 오차가 전체 RMSE를 지배하고 있습니다.
- **상위권의 비밀**: 일반적인 Ridge 스태킹은 전체 영역의 MSE를 고르게 줄이려 하지만, 상위권은 **오차가 큰 고가/대형 영역에 가중치를 주는 특화 학습(Sample Weighted Ridge)**을 적용했거나, 가격대를 예측하는 분류 모델을 Stage 0로 두고 "대형/고가 전용 Ridge"와 "일반 영역 Ridge"를 따로 앙상블하는 **Local Weighting**을 구축했을 것입니다.

### 3. 정교한 Transductive Pseudo-Labeling & Target Matching
- 1~3위는 Test 데이터셋의 피처 분포를 완벽하게 활용하여 Train 모델과의 괴리를 줄였을 것입니다.
- 우리는 PL2에서 신뢰도 50% 필터링을 썼지만, 그들은 Test 예측값의 평균과 분산이 Train의 최신 트렌드(OOT) 분포와 정확히 일치하도록 조율하는 **Distribution Calibration**이나, Adversarial Validation 분류기의 가중치를 Inverse Probability Weighting(IPW)으로 학습에 녹여 넣었을 가능성이 큽니다.

---

## Q3. 전략 45 기반 Final Submission 시나리오

전략 45가 새로운 최선(Public 2,094.9)이 되었지만, Private LB에서 순위 하락(Shake-down)을 막기 위해 2개의 최종 제출은 극명하게 다른 가치를 지향해야 합니다.

### Final Submission 1: 공격형 / 최고 성능 지향 (The Winner)
*   **대상**: **전략 45 기반 미세 보정본 (또는 개선된 10모델+α 스태킹)**
*   **이유**: Public 4위까지 이끌어준 동력이므로, 이를 폐기하는 것은 손해입니다. Public LB의 트렌드가 Private과 상당히 공유된다는 가정하에 가장 공격적인 조합으로 우승을 노립니다.
*   **보완 전략**: 
    - 트렌드 보정 강도 `α=1.0`을 유지하되, 고가 영역(예: 예측가 5억 이상)에서 트리 모델의 과소예측 편향을 미세하게 보정하는 사후 함수(Post-processing Multiplier)를 적용합니다.

### Final Submission 2: 보수형 / 안정적 일반화 지향 (The Safe Bet)
*   **대상**: **[전략 08 (Original 4모델) + ExtraTrees 2모델 + KRR/LGB-ET] 의 Multi-Seed 앙상블 (PL2 배제)**
*   **이유**: Pseudo-Labeling(PL2)은 Public LB 점수를 크게 올리지만, Test 데이터의 가짜 라벨을 재학습하므로 **Private Test 데이터셋의 숨겨진 분포가 다를 경우 오차가 기하급수적으로 폭발(Overfitting to Public)**할 위험이 존재합니다.
*   **구성 방법**:
    - 의도적으로 PL2(의사 라벨링)을 완전히 제거한 모델셋을 구성합니다.
    - Public에서 비록 손해를 보더라도(예: Public 2,100대 수준), **Multi-Seed (4개 시드 이상) 평균**을 적용하여 모델의 분산을 제안하고 일반화 성능을 극대화합니다.
    - 이 방식은 합성 데이터의 무작위 노이즈를 완벽하게 스무딩하여 Private에서 상위권 팀들이 과적합으로 무너질 때 순위가 수직 상승하는 **최후의 방어선** 역할을 수행합니다.

---

## Action Plan (남은 12일 전략)

1.  **DGP 역공학 실험 (1~2일)**:
    - Log-scale 선형 회귀와 의사 결정 나무 분기점 분석을 통해 `Area`, `Floor`, `Gu` 간의 곱셈적 규칙성이 존재하는지 수식화 및 피처화 진행.
2.  **알고리즘 다양성 확장 (3~5일)**:
    - LGBM `extra_trees=True` 및 Scikit-learn `KernelRidge` (RBF 커널) 로컬 검증 및 앙상블 추가 실험.
3.  **Local Weighting / Specialized Ridge (6~8일)**:
    - 대형 평수(Area >= 120) 및 강남/성동/용산구 전용 Sample Weight를 스태킹 Ridge에 반영하는 실험.
4.  **최종 2개 제출 셋팅 및 검증 (9~12일)**:
    - 1번(PL2 포함 최적화)과 2번(PL2 배제, 멀티시드 스무딩 일반화)의 최종 추론 스크립트 완성 및 무결성 확인.
