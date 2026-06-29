# opencode Advisor Report (2026-06-28)

> Context: Public 6위 (2,086.6), 3위가 2,073→2,027로 46점 점프
> 전일 대비: 성능-다양성 딜레마 확인 (GBDT+Ridge 12모델 = 최적점), 모든 비트리/PL/보정 시도 실패

---

## 진단: 왜 46점 점프가 가능했는가?

3위(2,027)와 우리(2,087)의 격차 60점. 이 차이는 **모델 다양성 소진 상태**에서 발생한 점프.

핵심 방증:
- OOF 2,191 vs Public 2,087 → **CV가 pessimistic (100점 차이)**
- GBDT+Ridge 12모델 스택은 **성능-다양성 딜레마로 최적점 도달**
- 기존 시도는 모두 **모델/피처/PL의 변형**에 집중 → 수확 체감

**3위가 한 46점 점프의 가장 유력한 설명**: 시간 축(Time Axis)을 직접 모델링하거나, OOF-Public 갭을 활용한 Test 분포 매칭.

---

## 제안 1: Time-Aware Decomposition (가장 유망) ⭐⭐⭐

### 문제
현재 GBDT 모델은 YearMonth_Seq를 피처로 받지만, **트리 기반 모델은 시간 외삽(extrapolation)이 근본적으로 약함**. YearMonth_Seq=25(2026년 1월)은 Train의 최대값(2025년 12월, seq=23)을 벗어난 범위 → 트리는 이 구간을 학습한 적이 없음.

### 해결: Linear Trend Decomposition
Train 데이터에서 시간 trend를 먼저 추출하고, 잔차(residual)를 GBDT로 학습:

```
Step 1: Log(Target) ~ Log(피처) + YearMonth_Seq (Ridge)
  → 선형 모델이 시간 외삽을 담당
Step 2: 잔차(Target - Skeleton) ~ GBDD (CB/LGB)
  → 비선형 패턴 학습
Step 3: Skeleton + Residual = Final Prediction
```

### 전략 63(Hybrid Skeleton)과의 차별점
전략 63은 Ridge(alpha=1.0)를 모든 수치형에 대해 log 변환 후 사용. 너무 단순.
개선:
- **alpha search**: Ridge alpha를 0.1~1000 범위에서 CV로 최적화
- **Skeleton 모델 다양화**: Ridge + Lasso + ElasticNet을 각각 skeleton으로 사용 → 앙상블
- **YearMonth_Seq 강조**: Skeleton에서 YearMonth_Seq에 높은 가중치 부여 (또는 유일한 피처로 사용)
- **Gu별 Skeleton**: 각 Gu별로 별도 선형 trend 추정 (GTR의 일반화된 버전)

### 기대 효과
- 시간 외삽을 선형 모델이 전담 → GBDT는 비선형 패턴에 집중
- 3위의 46점 점프를 따라잡을 유일한 방법
- **기대**: 10~30점 (시간 외삽이 Test의 많은 부분을 설명한다면)

### 구현 (30분)
```python
# Gu별 선형 Trend Skeleton
for gu in gu_list:
    gu_mask = train['Gu'] == gu
    X_gu = train.loc[gu_mask, 'YearMonth_Seq'].values.reshape(-1, 1)
    y_gu = np.log1p(train.loc[gu_mask, 'Target'].values)
    model = Ridge(alpha=1.0).fit(X_gu, y_gu)
    train_gu_pred = np.expm1(model.predict(X_gu))
    
    test_gu_mask = test['Gu'] == gu
    X_test_gu = test.loc[test_gu_mask, 'YearMonth_Seq'].values.reshape(-1, 1)
    test_gu_pred = np.expm1(model.predict(X_test_gu))

# 잔차 = 실제값 - Skeleton 예측
residual = train['Target'] - train_skeleton_pred

# GBDT로 잔차 학습 (YearMonth_Seq 제외)
gbdt.fit(X_train.drop('YearMonth_Seq'), residual)

# 최종 = Skeleton + GBDT Residual
final = skeleton_test_pred + gbdt_test_pred
```

---

## 제안 2: Public LB 기반 Blend Optimization (실용적) ⭐⭐

### 문제
현재 블렌드 비율(53:47=80:20)은 **단 1회 제출로 결정**. 시드 가중치도 모두 동일(0.25).

### 해결
하루 5회 제출 중 1~2회를 blend 최적화에 할당. Bayesian Optimization으로 LB 피드백 기반 탐색:

**최적화 변수** (6개):
1. w_53 (53 비중, 0.5~1.0)
2. w_seed_42 (시드42 가중치, 0.0~1.0)
3. w_seed_123 (시드123 가중치)
4. w_seed_456 (시드456 가중치)
5. w_seed_789 (시드789 가중치)
6. gtr_alpha (GTR 강도, 0.0~2.0)

### 리스크 관리
- 531개 Test 샘플에서 6개 변수 최적화는 **과적합 위험 낮음**
- Public/Private 상관관계가 높은 대회에서는 유효한 전략
- **안전장치**: 첫 5회는 넓은 범위 탐색, 이후 5회는 최적점 근처 집중

### 간단 버전 (지금 바로 가능)
```python
# 61a(70:30), 61b(90:10) 제출 → 최적 비율 확인
# 최적 비율 근처에서 75:25, 85:15 추가 제출
# 이후 시드 가중치 탐색
```

### 기대 효과
- **기대**: 5~15점 (현재 블렌드가 멀리 떨어져있다면)
- **리스크**: 낮음 (간단한 실험)

---

## 제안 3: OOF-Public 갭 활용 Test Distribution Matching ⭐⭐

### 문제
OOF 2,191 vs Public 2,087 = 104점 차이. 이는 **Test가 Train보다 예측하기 쉬운 구조**임을 의미.

### 해결: 왜 Test가 더 쉬운가?

오차 분석(06-24)에 따르면:
- 대형(120㎡↑): 오차 상위권 과대표 (전체 2.6%지만 오차 11%)
- 강남구: 오차 비율 10.7%로 압도적
- 성동구/용산구: RMSE/평균가 비율 8.1%/6.7%로 최악

→ **Test에 대형/강남구/성동구 샘플이 적다면** OOF보다 Public이 좋은 것.

### 검증 및 활용
1. Train 샘플의 오차 크기와 Gu/Area의 관계 분석
2. 오차가 큰 패턴 식별 (예: 강남구 + Area > 120)
3. Train에서 이 패턴의 모델 가중치를 낮춤 (또는 Test에서 동일 패턴 보정)
4. Adversarial Validation으로 Test-like Train 샘플 식별 → 가중 학습

### 간단 버전
```python
# 오차 상위 10% 샘플 특성 분석
train_orig['oof_error'] = abs(cv_pred - train_orig['Target'])
high_error = train_orig.nlargest(int(len(train_orig)*0.1), 'oof_error')
print(high_error.groupby('Gu').size().sort_values())
print(high_error.groupby(pd.cut(high_error['Exclusive_Area'], bins=[0,60,85,120,200])).size())
```

### 기대 효과
- **기대**: 5~20점 (Test 분포 차이가 크다면)
- **리스크**: 중간 (Train에서 가중치 조정 시 CV 신뢰도 하락)

---

## 제안 4: Rank-Based Ensemble (Robustness) ⭐

### 문제
RMSE는 이상치(outlier)에 극도로 민감. 모델 간 예측 분포가 다를 때 단순 평균은 비효율적.

### 해결
각 모델의 예측을 순위(rank)로 변환 → 평균 → 원래 분포로 역매핑:
```python
from scipy.stats import rankdata

rank_preds = np.array([rankdata(p) for p in model_predictions])
avg_rank = rank_preds.mean(axis=0)
# Train 분포 매핑
final = np.percentile(train_orig['Target'], (avg_rank / len(avg_rank)) * 100)
```

### 기대 효과
- **기대**: 2~5점 (이상치 완화)
- **리스크**: 낮음 (기존 예측과 블렌딩 가능)

---

## 제안 5: Test-Time Pseudo Label with Temporal Adaptation ⭐

기존 PL2는 신뢰도 상위 50%를 Train에 추가. 개선점:
1. **PL2 threshold를 Gu별로 다르게**: 각 Gu의 예측 일치도가 다를 수 있음
2. **PL2 + 시간 보정**: PL2로 Test 예측 후, 시간 trend로 추가 보정
3. **Iterative PL with decay**: 첫 PL2로 학습 → 다시 Test 예측 → 일치도 높은 샘플 추가

### 기대 효과
- **기대**: 3~8점 (PL2 개선 여지)
- **리스크**: 중간 (transductive bias 증가)

---

## 실행 우선순위 (잔여 10일)

| 우선순위 | 실행 | 기대효과 | 시간 | 설명 |
|:--------:|:-----|:--------:|:----:|:-----|
| **★★★** | 61a(70:30) + 61b(90:10) 제출 | 5~15점 | 10분 | 즉시 가능, blend 최적화 1차 |
| **★★★** | Time-Aware Skeleton (Gu별 선형 Trend) | 10~30점 | 30분 | 가장 큰 개선 가능성 |
| **★★☆** | 오차 패턴 분석 + Test 분포 매칭 | 5~20점 | 20분 | 왜 OOF가 Public보다 나쁜지 이해 |
| **★★☆** | Bayesian Blend Optimization (시드+GTR) | 5~15점 | 연속 | LB 피드백 활용 |
| **★☆☆** | Rank-Based Ensemble | 2~5점 | 15분 | 안전한 추가 |

### 제출 계획
```
06-28: 61a(70:30) + 61b(90:10) 제출 + Time-Aware Skeleton 로컬 실험
06-29: Time-Aware Skeleton 결과 확인 → 제출 if 개선
        blend 최적 비율 확인 → 75:25 or 85:15 제출
06-30: 오차 패턴 분석 기반 Test 분포 매칭
        시드 가중치 최적화 (4개 제출)
07-01: 최종 전략 확정 (1차)
07-02~07-07: Public LB 관찰 (주 2회, 방어 제출)
07-08: Final 1 (최선 전략) + Final 2 (방어 전략) 제출
```

### 최종 제출 전략
- **Final 1**: Time-Aware Skeleton + 53:47 Blend (최선 성능)
- **Final 2**: 기존 전략 56 (Blend 53:47 = 80:20) — 방어용
- **조건**: Final 1이 전략 56보다 Public에서 5점 이상 개선 시 채택, 미만 시 전략 56 유지

---

**작성**: opencode (2026-06-28)
**특이사항**: 성능-다양성 딜레마로 기존 모델 구조는 최적점. 46점 점프는 **시간 외삽(time extrapolation) 또는 Test 분포 매칭**으로만 설명 가능. Time-Aware Decomposition이 가장 유망한 돌파구. 당장 61a/61b 제출로 blend 최적화 시작할 것.
