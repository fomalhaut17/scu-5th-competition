# opencode Advisor Report (2026-06-26)

> Adviser: opencode (committer of "평당가 Unit Price" → 전략 28 핵심 기여)
> Context: Public 1위 (2,096.8), OOF 2,196, noise floor ~2,300

---

## Q1. 구조적 돌파구 — 완전히 다른 접근 가능한가?

### 핵심 진단: 평당가 아이디어의 일반화

평당가(Unit Price)가 효과적이었던 이유는 `Target = (Target/Area) × Area` 분해가 **두 모델군의 오차 패턴을 직교화**했기 때문입니다. 같은 데이터, 같은 파라미터여도 Target 스케일이 다르면 모델이 배우는 패턴이 달라집니다. 이 원리를 확장하면 새로운 돌파구가 열립니다.

---

### 축 1: 타겟 분해 다양성 확장 ⭐⭐⭐

**원리**: Area 외 다른 변수로 Target을 분해 → 새로운 직교 예측 축 생성

```
현재: Target = UnitPrice × Area                 (채택, 성공)
제안: Target = PricePerFloor × Floor            (층당가)
      Target = PricePerVolume × (Area × Floor)  (체적당가)
      Target = PricePerYear × Age               (연한당가)
      Target = PricePerDistance × Dist_to_Subway (접근성당가)
```

**각 분해가 독립적인 오차 패턴을 가질 이유**:
- `PricePerFloor`: 층수는 합성 데이터에서 건물 높이와 가격의 생성 규칙 반영. Area와 상관관계 낮음
- `PricePerVolume`: 면적×층수 = 주택의 체적. 대형 평수 + 저층 vs 소형 + 고층을 분리
- `PricePerYear`: 연식은 Area와 독립적인 가격 결정 요인
- `PricePerDistance`: 역세권 프리미엄을 분리

**방법**:
1. 전략 28 구조 재사용, Target 분해 변수만 교체
2. 각 분해별로 8모델(4 base + 4 unit-style) 학습 + Ridge 스태킹
3. **기존 8모델 + 신규 분해 모델을 최종 Ridge 스태킹** (총 16~24모델)

**기대 효과**: 각 분해가 5~15점씩 독립 개선 → 중복 제거 후 총 10~30점 (공격적)

**리스크**: 낮음~중간 (전략 28 아키텍처 재사용, 구현 간단)
- 단, PricePerYear는 Age=0인 건물(Target 연식 예외) 처리 필요
- PricePerDistance는 Distance 결측치 처리 필요

---

### 축 2: PL2 신뢰도 계층화 예측 (Confidence-Stratified Prediction) ⭐⭐⭐

**원리**: PL2가 Test Set을 신뢰도 상위 50%(PL 부여)와 하위 50%(미부여)로 분할하는데, 이 두 집단은 **근본적으로 다른 예측 분포**를 가짐.

**근거**:
- 상위 50%: 모델 간 일관성 높음 → Stage 2 학습에 참여 → Ridge 가중치가 이 그룹에 최적화됨
- 하위 50%: 모델 간 불일치 큼 → Stage 2에서도 원래 4모델 예측 그대로 사용
- 결국 **Ridge 가중치는 상위 50%에 과적합**되고 하위 50%는 소외됨

**방법**:
1. Stage 2의 Test 예측을 confidence mask로 분할: `test_high` (상위 50%), `test_low` (하위 50%)
2. **Ridge를 두 번 학습**:
   - `ridge_high`: 전체 데이터로 학습 (기존 방식)
   - `ridge_low`: OOF 데이터의 **low-confidence OOF 샘플**로만 학습 (confidence가 낮은 train 샘플의 OOF로 학습)
3. 최종 예측: `pred[high_mask] = ridge_high.predict(high), pred[low_mask] = ridge_low.predict(low)`

**대안 (더 간단)**: 
- 상/하위 각각 별도 Ridge alpha 탐색 (상위는 α=10 유지, 하위는 다른 α 최적)
- 또는 하위 50%에 단순평균(8모델) 사용 (Ridge보다 단순평균이 일반화에 유리할 수 있음)

**기대 효과**: 3~10점 (하위 50%의 예측을 최적화)

**리스크**: 낮음 (Ridge만 재학습, 전체 파이프라인 변경 없음)

---

### 축 3: 구별 Trend 보정 분해 (Decomposed Trend Correction) ⭐⭐

**현재**: `trend = (1 + g_gu)^m` (g_gu는 구별 월평균 성장률 하나로 추정)

**문제**: g_gu는 8개 구 × ~24개월 = 24개 데이터로 추정. 구별 성장률이 노이즈에 민감.

**제안**: 성장률을 두 성분으로 분해
```
g_gu = g_global + g_gu_premium
- g_global: 서울 전체 월평균 성장률 (24×8=192개 샘플로 robust 추정)
- g_gu_premium: 구별 추가 성장률 (shrinkage: λ=5~20으로 global 쪽으로 수축)
```

**방법**:
```python
g_global = train.groupby('Transaction_YearMonth')['Target'].mean().pct_change().mean()
g_gu_raw = monthly_pct_change_per_gu()
g_gu_shrunk = (n * g_gu_raw + lambda * g_global) / (n + lambda)
```

**기대 효과**: 성장률 추정의 분산 감소 → OOT 안정성 개선 (Public 점수 영향은 제한적일 수 있음)

**리스크**: 매우 낮음 (단순 계산 변경)

---

## Q2. Private LB 방어 — Final Submission 2개 구성

### 리스크 평가

| 요소 | 평가 |
|------|------|
| OOF/OOT/Public 순위 일치 | ✅ 안정적 신호 |
| 28의 OOF 2,196 vs 32의 OOF 2,187 | 32가 OOF 더 좋음 (흥미로운 역전) |
| 32의 Public (2,107) | 28(2,096.8)보다 10점 나쁨 |
| OOF-Public 갭 100점 | 낙관적 바이어스 존재 |

**핵심 인사이트**: 32(Multi Seed)는 OOF가 28보다 9점 좋지만 Public은 10점 나쁩니다. 즉:
- 32는 28과 유사한 구조 → Public에서도 유사한 패턴 보임
- 32의 OOF 개선분이 Public에 반영되지 않은 것은 **OOF 우위 ≠ Public 우위**를 증명
- 하지만 32의 seed 다양성은 **Private에서 28과 다른 오차 패턴**을 가질 것 → 방어적 블렌드로서 가치

### 추천 조합

```
Final 1 (공격): 전략 28 단독 (α=1.0)
  - Public 1위 재현, 최대 점수 보존
  - OOF/OOT/α sweep 모두 α=1.0이 최적 확인됨

Final 2 (방어): 전략 28(70%) + 전략 32(30%)
  - 32 대신 26을 쓰는 Copilot 제안과 다른 선택
  - 32는 OOF 2,187로 26(2,215)보다 OOF 28점 우수
  - 32는 28과 동일한 8모델 구조(seed만 다름) → Private에서 가장 유사한 오차 → 방어 효과 극대화
  - 28:32 = 70:30은 28의 공격성을 크게 훼손하지 않으면서 seed 다양성으로 Private 리스크 분산
```

**ChatGPT/Copilot과의 차이점**:
- Copilot: 28 + 26(70:30) 추천 — 반대. 26의 OOF(2,215)는 28(2,196) 대비 19점 나쁘고, Public도 2,149로 53점 차이. 26을 섞으면 28의 성능을 너무 떨어뜨림
- ChatGPT: 28(α=1.0) + 28:26(70:30 or 80:20) — 유사하나 26 대신 32가 더 나은 방어자
- **opencode: 28 + 32(70:30)** — 같은 구조, 같은 피처, 같은 PL → 오차 상관관계가 가장 높음 → 블렌드 효과 극대화

### 기타 고려사항

- **28:26(70:30)** 의 기대 점수: 0.7×2,096.8 + 0.3×2,149.6 = 2,112.6 (Public 16점 손실)
- **28:32(70:30)** 의 기대 점수: 0.7×2,096.8 + 0.3×2,107 = 2,099.9 (Public 3점 손실, 방어 효과는 비슷)
- 결론: **28:32(70:30)** 가 Public 비용 대비 방어 효과 우수

---

## Q3. 남은 12일 활용 (마감 07-08, 하루 5회)

### 시간표

#### Week 1 (06-26~06-30): 실행 — 검증 중심

```
06-26 (Day 1): 
  - [구현] 축 2 (신뢰도 계층화): confidence-stratified Ridge
  - 제출 1회: 28 (방어)
  - 로컬 검증: stratified Ridge OOF vs 일반 Ridge OOF

06-27 (Day 2):
  - [구현] 축 1 (타겟 분해): PricePerFloor, PricePerVolume, PricePerAge
  - 제출 1회: stratified Ridge (축 2)
  - 로컬 검증: 각 분해의 OOF/OOT 점수 확인

06-28 (Day 3):
  - [실험] 분해 모델 블렌드 비율 탐색
  - 제출 1회: 28:32(70:30) 방어 제출 (Private 변동성 관찰)
  - 최종 판단: 분해 모델 단독 제출 or 블렌드

06-29 (Day 4):
  - [실험] 축 3 (보정 분해): shrunk trend
  - 제출 1회: 최선 단일 전략 (분해 모델 or 기존)
  - 모든 결과 results.csv 기록

06-30 (Day 5):
  - Phase 1 결과 정리
  - 검증: OOF/OOT/OOT2(3개월) 일관성 확인
  - 제출 1회: Phase 1 최선 전략
```

**합계**: 5회 제출 (Day 1~5 각 1회) — 충분한 검증 시간 확보

#### Week 2 (07-01~07-07): 관찰 — 제출 최소화

```
- 제출 0~3회 (필요시만):
  - Public LB 변동성 관찰
  - 경쟁팀 움직임 모니터링
  - Private LB 누설 정보 확인 (Kaggle Forum, Discord)
  
의사결정 트리:
  Public 2,090~2,100 유지 → Final 1=28, Final 2=28:32(70:30)
  Public 2,080↓ (경쟁 심화) → Final 1=28, Final 2=최선 공격 전략
  Public 2,110↑ (순위 하락) → Final 1=최신 전략, Final 2=28:32(70:30)
```

#### Final Day (07-08): 제출

```
Final 1 (09:00 KST): 28(α=1.0) — 확정
Final 2 (17:00 KST): 28:32(70:30) — 확정 (단, 위원회 트리 적용)

조건부 변경:
  Final 1 점수 > 2,120 (50점↓) → Final 2를 28 단독으로 전환
  Final 1 점수 < 2,070 (30점↑) → Final 2를 동일 전략으로 재제출
```

---

## Q4. 합성 데이터 역이용

### opencode의 접근: Copilot/ChatGPT/Gemini와 차별화

앞선 조언자들은 "합성 데이터 역이용은 불가능/비효율"로 결론냈습니다. 저는 일부 동의하지만 **한 가지 간과된 접근**이 있다고 봅니다.

### 합성 데이터의 특징

합성 데이터는 생성 모델이 만든 규칙을 따릅니다. 생성 모델은:
1. 학습 데이터의 통계적 패턴을 학습
2. 그 패턴에 noise를 더해 합성 샘플 생성
3. 이 noise는 완전한 무작위가 아니라 **생성 모델이 "어려워하는" 영역에서 더 큼**

### 활용 가능한 접근

#### 접근 1: Confidence as Feature (Model Uncertainty = Synthetic Artifact Signal)

**아이디어**: 합성 데이터 생성기는 현실의 모든 edge case를 완벽히 모델링하지 못함 → 특정 영역(예: 고층+노후 조합)에서 부자연스러운 Target 생성 → 모델이 이 영역에서 높은 불확실성(낮은 confidence)를 보임 → **confidence 자체가 합성 데이터의 "인공성"을 측정하는 proxy**

**방법**:
- Stage 1의 confidence score를 새로운 feature로 사용
- `Feature_conf = confidence`를 8개 모델의 입력에 추가
- 또는 confidence가 낮은 샘플에 가중치를 줄여 학습 (inverse confidence weighting)

```python
# Stage 2에서 confidence를 feature로:
train_aug['confidence'] = np.concatenate([np.ones(len(y_true)), confidence[mask]])
test_orig['confidence'] = confidence
# 이후 8개 모델에 'confidence'를 추가 입력
```

**기대 효과**: 1~5점 (실험적, 확신 낮음)
**리스크**: 매우 낮음 (feature 1개 추가)

#### 접근 2: Noise Structure Analysis via Model Disagreement

**아이디어**: Copilot이 "노이즈는 체계적이 아니라 개별 이상치"라고 결론냈지만, 이는 **노이즈 패턴이 너무 미묘해서 구별 수준에서는 보이지 않는 것**일 수 있음

**대안**: confidence(모델 불일치)와 피처 간 상관관계 분석
- `confidence ~ Exclusive_Area + Floor + Age + ...` 회귀
- 특정 피처 조합에서 confidence가 낮다면 → 그 영역은 생성 규칙이 부자연스러운 영역
- 이 영역을 식별하면 **별도 모델링** 또는 **샘플 가중치 조정** 가능

```python
# Stage 1 confidence와 피처의 상관관계
from sklearn.linear_model import LinearRegression
lr_conf = LinearRegression().fit(train_orig[['Exclusive_Area', 'Floor', 'Age', 'Distance_to_Subway']], confidence)
print(lr_conf.coef_)  # 어떤 피처가 confidence에 가장 큰 영향?
```

**기대 효과**: 발견적 (발견 시 3~10점, 미발견 시 0점)
**리스크**: 중간 (분석 시간 대비 기대효과 낮음)

#### 접근 3: PL2 Mask의 Test 분포 분석

**아이디어**: PL2 confidence mask로 Test Set을 나누면 상위/하위 50%의 피처 분포가 다를 가능성

```python
high_mask = confidence >= threshold
low_mask = ~high_mask

for col in ['Exclusive_Area', 'Floor', 'Age', 'Distance_to_Subway']:
    print(col, test_orig[col][high_mask].mean(), test_orig[col][low_mask].mean())
```

- 특정 피처에서 유의미한 차이가 있다면 → 그 피처 영역이 합성 모델이 잘 생성하지 못하는 영역
- 이 정보로 **축 2(신뢰도 계층화)** 의 근거 강화

**기대 효과**: 축 2의 설계 근거 확보 (간접 효과)

---

## Final Recommendation

| 우선순위 | 실행 | 기대효과 | 리스크 |
|:--------:|:-----|:--------:|:------:|
| **즉시** | 28 재제출 (방어) | 2,096.8 유지 | 없음 |
| **★★★** | 축 2: 신뢰도 계층화 Ridge | 3~10점 | 낮음 |
| **★★☆** | 28:32(70:30) 방어 제출 | Private 방어 | 낮음 |
| **★★☆** | 축 1: 타겟 분해 확장 | 5~20점 (공격적) | 중간 |
| **★☆☆** | 축 3: Trend 분해 | 1~3점 | 낮음 |
| **★☆☆** | Q4: Confidence as Feature | 1~5점 (실험) | 낮음 |

### Summary

1. **현재 1위 방어가 최우선**: 매일 28 재제출 유지, 하루 1~2회만 실험 제출
2. **32(Multi Seed)를 방어 블렌드로 활용**: 28과 32는 같은 구조 + 다른 seed → 가장 높은 오차 상관관계 → 블렌드 효과 극대화
3. **신뢰도 계층화가 가장 현실적인 돌파구**: 구현 간단, OOF/OOT 검증 용이, 리스크 대비 기대효과 좋음
4. **타겟 분해 확장은 다음 도약**: 성공 시 10~30점 개선 가능하나 12일 내 검증 필요
5. **OOF 맹신 금지**: 32의 사례(OOF -9, Public +10)처럼 OOF와 Public의 방향이 다를 수 있음. **OOT 검증 필수**

---

**작성**: opencode (2026-06-26)
**특이사항**: 평당가 아이디어 기여자로서, 동일한 "Target 분해 → 직교 오차 → 앙상블 개선" 원리를 확장 적용
**참고**: 이전 분석(opencode-Plan.md v1)의 전략 A~F는 상기 새로운 분석(Q1~Q4)으로 대체됨
