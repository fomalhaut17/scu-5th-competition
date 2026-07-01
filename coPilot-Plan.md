# coPilot 제언: 시간 외삽과 Residual 비선형성 (2026-07-01)

> **핵심 아이디어**: Gemini는 오차 세그먼트, opencode는 DGP 함수를 본다. coPilot은 **시간 축의 외삽 편향**과 **Residual 구조의 비선형성**을 본다.

---

## 진단: 현재 파이프라인의 맹점

### 문제 1: Time Extrapolation Bias
```
Train: 2024-01-01 ~ 2025-12-31
OOT:   2025-10-01 ~ 2025-12-31  (마지막 3개월, 246건)
Test:  2026-01-01 ~ 2026-03-31  (미래 3개월, 531건)

현황: 
- 56, 63, 69는 모두 2025년 데이터로 학습된 모델
- Test는 2026년 (시간상 6개월 미래)
- OOT(2025 Q4)와 Test(2026 Q1) 사이에 "시간 점프"가 있음
- 각 컴포넌트의 "시간 외삽 강도"는 다를 수 있음
```

**발견**: 현재 50:30:20은 전체 평균 RMSE 기준이지만, **시간 축에 대한 안정성**은 고려하지 않음.

### 문제 2: Residual 결합의 선형성 가정
```
현재: pred_final = 0.5 * pred_56 + 0.3 * pred_63 + 0.2 * pred_69
      = Ridge(pred_56, pred_63, pred_69) 선형 조합

하지만:
- 각 컴포넌트의 오차가 다른 "기하학적 구조"를 가질 수 있음
- 오차 벡터 공간에서 비선형 결합이 더 효율적일 수 있음
- 예: pred_56이 "과대 추정하는 경향"과 "과소 추정하는 경향"이 구간마다 다름
     → 단순 가중치 선형 조합으로는 이를 못 잡음
```

---

## 제언 3가지 (Gemini/opencode와 다른 축)

### **🎯 제언 1: Time Extrapolation Robustness Scoring**

**개념:**
각 컴포넌트가 "시간을 얼마나 잘 외삽하는가"를 점수화 → 그 점수에 따라 가중치 재조정.

**구체적 실행:**

#### Step 1: 시간 단계별 OOT 분할 (시계열 특성 활용)
```python
# OOT (2025-10-01 ~ 2025-12-31, 246건)를 3개월로 분할
OOT_M10 = OOT[date < '2025-11-01']  # 246 * 1/3
OOT_M11 = OOT[(date >= '2025-11-01') & (date < '2025-12-01')]
OOT_M12 = OOT[date >= '2025-12-01']  # 가장 미래(Test와 가장 가까움)

# 각 구간에서 RMSE 계산
rmse_56_m10, rmse_56_m11, rmse_56_m12 = [...]
rmse_63_m10, rmse_63_m11, rmse_63_m12 = [...]
rmse_69_m10, rmse_69_m11, rmse_69_m12 = [...]
```

#### Step 2: 시간 트렌드 분석 (기울기 계산)
```python
# 각 컴포넌트의 "시간에 따른 성능 변화" 측정
# 만약 RMSE가 악화 추세라면 → Test에서도 악화될 가능성 높음

# 선형 회귀: time_axis(0, 1, 2) vs RMSE
slope_56 = polyfit([0, 1, 2], [rmse_56_m10, rmse_56_m11, rmse_56_m12], 1)[0]
slope_63 = polyfit([0, 1, 2], [rmse_63_m10, rmse_63_m11, rmse_63_m12], 1)[0]
slope_69 = polyfit([0, 1, 2], [rmse_69_m10, rmse_69_m11, rmse_69_m12], 1)[0]

# 해석:
# slope < 0 (개선 추세): 좋음, 외삽에 강함 → 가중치 UP
# slope > 0 (악화 추세): 위험, 외삽에 약함 → 가중치 DOWN
# slope ≈ 0 (평탄): 안정적
```

#### Step 3: Robustness 가중치 조정
```python
# Baseline: w_56=0.5, w_63=0.3, w_69=0.2

# Slope 기반 penalizer (0~1 범위)
robustness_56 = 1.0 if slope_56 <= 0 else (1.0 - 0.3 * abs(slope_56))
robustness_63 = 1.0 if slope_63 <= 0 else (1.0 - 0.3 * abs(slope_63))
robustness_69 = 1.0 if slope_69 <= 0 else (1.0 - 0.3 * abs(slope_69))

# 재정규화
w_56_adj = 0.5 * robustness_56 / (0.5 * robustness_56 + 0.3 * robustness_63 + 0.2 * robustness_69)
w_63_adj = 0.3 * robustness_63 / (...)
w_69_adj = 0.2 * robustness_69 / (...)

pred_final = w_56_adj * pred_56 + w_63_adj * pred_63 + w_69_adj * pred_69
```

**기대 효과:** +3~8점
- **왜**: 시간 안정성을 명시적으로 모델링 → 6개월 미래 외삽의 불확실성 감소
- **예시**: 만약 pred_69(One-Hot)가 최근 악화 추세라면 down-weight, pred_56이 안정적이면 up-weight

**검증:**
- OOT_M12 (2025-12월, Test와 가장 유사)에서 새 가중치 성능 확인
- 개선 신호 → Public 제출

---

### **🎯 제언 2: Residual Nonlinear Stacking (오차 공간의 비선형 구조)**

**개념:**
각 컴포넌트의 residuals를 다시 한 번 **비선형 변환** → 결합. 선형 Ridge의 한계 극복.

**구체적 실행:**

#### Step 1: Residual Matrix 구성
```python
# Train에서:
residual_56 = y_train - pred_56_train
residual_63 = y_train - pred_63_train
residual_69 = y_train - pred_69_train

# Shape: (1969, 3) matrix
residual_matrix = np.column_stack([residual_56, residual_63, residual_69])
```

#### Step 2: Nonlinear 변환 - 3가지 방법 병렬 테스트

**방법 A: Kernel Ridge Regression (오차 공간)**
```python
from sklearn.kernel_ridge import KernelRidge

# 입력: 3개 컴포넌트의 예측값
# 출력: 최종 예측값
# 커널: RBF or Poly (오차 공간에서의 비선형 관계 포착)

kr = KernelRidge(kernel='rbf', alpha=10, gamma=0.01)
kr.fit(np.column_stack([pred_56_train, pred_63_train, pred_69_train]), y_train)
pred_final_kr = kr.predict(np.column_stack([pred_56_test, pred_63_test, pred_69_test]))
```

**방법 B: Decision Tree Regressor (오차의 규칙 발견)**
```python
from sklearn.tree import DecisionTreeRegressor

# Ridge 대신 shallow tree 사용
# 각 컴포넌트의 오차 크기에 따라 가중치를 adaptively 조정
tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=10)
tree.fit(np.column_stack([pred_56_train, pred_63_train, pred_69_train]), y_train)
pred_final_tree = tree.predict(...)

# 해석: "pred_56이 크면 이쪽 가중치, pred_63이 크면 저쪽 가중치" 같은 규칙 자동 발견
```

**방법 C: Neural Network (1-2 layer)**
```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(3,)),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(
    np.column_stack([pred_56_train, pred_63_train, pred_69_train]), 
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2
)
```

#### Step 3: OOT 검증 + 최종 선택
```python
# 각 방법의 OOT RMSE 비교
rmse_kr = evaluate_on_oot(kr)      # 예: 2,594
rmse_tree = evaluate_on_oot(tree)  # 예: 2,596
rmse_nn = evaluate_on_oot(nn)      # 예: 2,593

# 가장 낮은 것 선택 (또는 앙상블)
best_meta = [kr, tree, nn][argmin([rmse_kr, rmse_tree, rmse_nn])]
```

**기대 효과:** +2~6점
- **왜**: 선형 Ridge는 컴포넌트 간 복잡한 상호작용을 못 잡음. 비선형 메타 모델은 "이 상황에서는 56, 저 상황에서는 63"이라는 adaptive 가중치 학습 가능
- **예시**: 예측값이 크면 과소추정 보정(pred_63 up), 작으면 과대추정 보정(pred_69 down) 같은 세밀한 조정

**검증:**
- OOT 5-fold에서 cross-validation RMSE 계산 (과적합 방지)
- 개선 신호 → Public 제출

---

### **🎯 제언 3: Synthetic Data의 "생성 과정 가설" 기반 Mixture Modeling**

**개념:**
합성 데이터 = 알려지지 않은 생성 과정(DGP). 가능한 여러 생성 가설을 세우고, 각각에 최적화된 파이프라인 구성 → 사후적 혼합.

**구체적 실행:**

#### 가설 1: Linear + Tree Residuals (가장 그럴듯)
```
Price = LinearBase(features) + TreeResiduals(nonlinear_terms) + Noise

의미:
- 기본 가격은 features의 선형 결합 (면적 × 기본단가 + 구별 프리미엄)
- 비선형성은 tree 모델이 잡음 (interaction, 비선형 trend)
- 목표: Linear + Tree combination 최적화

실행:
pred_hypothesis_1 = 0.6 * pred_linear_skeleton + 0.4 * pred_tree_residual
```

#### 가설 2: Piecewise Linear (구간별 다른 규칙)
```
Price = {
  LinearModel_Gu1 if Gu == 'Gu1'
  LinearModel_Gu2 if Gu == 'Gu2'
  ...
}

의미:
- 각 구마다 가격 책정 논리가 완전히 다름 (강남 ≠ 강북)
- Per-Gu Skeleton의 아이디어 강화

실행:
pred_hypothesis_2 = Per-Gu Ridge (기존 전략63의 pure 버전)
```

#### 가설 3: Multiplicative (기하학적 복합)
```
Price = BasePrice * LocationFactor * TimeFactor * NoiseFactor

의미:
- 가격은 기본값의 곱(multiplicative) 조합
- Log-price 변환이 이를 선형으로 만듦

실행:
pred_hypothesis_3 = exp(Ridge(log_y, log_features))
```

#### Step 1: 각 가설별 파이프라인 구성
```python
pipeline_h1 = 0.6 * skeleton_linear + 0.4 * gbdt_residuals
pipeline_h2 = per_gu_skeleton
pipeline_h3 = log_price_linear_model

oof_h1 = cross_validate(pipeline_h1, cv=5)  # RMSE: 2,650
oof_h2 = cross_validate(pipeline_h2, cv=5)  # RMSE: 2,705
oof_h3 = cross_validate(pipeline_h3, cv=5)  # RMSE: 2,720
```

#### Step 2: Mixture Weight (사후 확률)
```python
# OOF RMSE 기반 가중치 (더 나은 가설에 더 높은 가중치)
likelihood_h1 = exp(-oof_h1^2 / sigma^2)  # Gaussian likelihood
likelihood_h2 = exp(-oof_h2^2 / sigma^2)
likelihood_h3 = exp(-oof_h3^2 / sigma^2)

# 정규화
w_h1 = likelihood_h1 / (likelihood_h1 + likelihood_h2 + likelihood_h3)
w_h2 = likelihood_h2 / (...)
w_h3 = likelihood_h3 / (...)

# 최종
pred_final = w_h1 * pipeline_h1 + w_h2 * pipeline_h2 + w_h3 * pipeline_h3
```

**기대 효과:** +2~5점
- **왜**: 하나의 가설(선형, Per-Gu, 비선형)에 집중하는 것보다 여러 가설을 확률적으로 혼합하면 robust
- **예시**: 만약 실제 DGP가 "구마다 다른 선형 규칙"(가설2)이면 w_h2가 높아져서 자동으로 강조됨

**검증:**
- OOT에서 mixture weight 검증
- 개선 신호 → Public 제출

---

## 실행 우선순위 & 일정

| 단계 | 작업 | 소요시간 | 병렬 가능 | 제출 |
|------|------|--------|---------|------|
| **A (우선)** | 제언 1: Time Robustness Score 계산 & 가중치 재조정 | 1~2시간 | - | 회당 1회 |
| **B (병렬)** | 제언 2: Residual Nonlinear (KR/Tree/NN 3종 테스트) | 2~3시간 | A와 병렬 | 회당 1~2회 |
| **C (병렬)** | 제언 3: Hypothesis Mixture 구성 | 2시간 | A/B와 병렬 | 회당 1회 |
| **D (종합)** | A+B+C 중 최고 결과 선택 또는 앙상블 | 1시간 | - | 최종 1~2회 |

**타이밍:**
```
2026-07-01 (오늘):   제언 1 계산 시작 (OOT 시간별 분석)
2026-07-02~03:       제언 2 (RKR/Tree/NN 학습)
2026-07-03~04:       제언 3 (Mixture 구성)
2026-07-04~07:       결과 통합 & 최종 제출
```

**제출 전략:**
- 하루 5회 × 7일 = 35회 (충분)
- 각 아이디어: 로컬 OOT 검증 후 유의미하면 즉시 제출
- Fallback: 제언 1 (Time Robustness)만 성공해도 +3점 기대

---

## 예상 개선폭

### 낙관 시나리오
```
현재: 2,028.35
+ 제언 1 (Time Robustness): +3~5점 → 2,023~2,025
+ 제언 2 (Nonlinear Stacking): +2~4점 → 2,019~2,023
+ 제언 3 (Mixture): +1~3점 → 2,016~2,022
→ 최종: ~2,018 (2위 범위)
```

### 기대값 시나리오
```
현재: 2,028.35
+ 제언 1 성공 (Time): +3점 → 2,025.35
+ 제언 2 부분 성공 (Nonlinear): +1~2점 → 2,023~2,024
+ 제언 3 미미 → 0점
→ 최종: ~2,024 (4위 근처)
```

### 보수 시나리오
```
현재: 2,028.35
+ 제언 1 노이즈만 발견 → +0~1점
+ 제언 2 과적합 → 0점
+ 제언 3 개선 없음 → 0점
→ 최종: ~2,027 (현 위치 유지)
```

---

## 이것이 다른 제안과 다른 이유

| 관점 | Gemini | opencode | **coPilot** |
|------|--------|----------|-----------|
| **초점** | 오차 세그먼트 (고가/저가) | DGP 함수 발견 | **시간 축 + 오차 공간 구조** |
| **축** | 피처 공간 (면적×가격) | 함수 공간 (식) | **시간축 + 기하학적 구조** |
| **기대점** | 세그먼트별 오차 최소화 | 생성함수 역공학 | **외삽 안정성 + 비선형 조합** |
| **리스크** | 세그먼트 정의 자의성 | DGP 명확화 어려움 | **메타모델 과적합** |

---

## 최종 메시지

**핵심 통찰:**
1. **OOT(246건) vs Test(531건)** 사이의 "시간 갭"을 무시하고 있음
2. **Ridge 선형 결합**은 각 컴포넌트의 오차 구조(오버쉈팅 패턴, 세그먼트별 편향)를 못 잡음
3. **합성 데이터의 생성 가설**이 여러 개 있을 수 있는데, 하나만 사용 중

**우리의 차별성:**
- 시간을 **명시적으로** 모델링 (trend, extrapolation robustness)
- Residual을 **비선형 공간**에서 재조합 (메타 모델의 능력 확장)
- 생성 가설을 **확률적 혼합** (robustness, diversity)

7일 안에 "구조적 돌파구"를 원한다면, **이미 있는 컴포넌트를 더 영리하게 조합하는 것**만으로도 +3~5점은 가능합니다.

---

**작성**: coPilot (Copilot CLI) | **날짜**: 2026-07-01 | **버전**: v2.0 (Creative Approach)
