# Copilot 조언 보고서 - 2026-06-27

## 📍 상황 분석

**현황**: Public 4위, RMSE 2,094.9 (전략 45)  
**전 단계**: 전략 28(2,097)에서 **-2점 개선**  
**핵심 발견**: ExtraTrees가 상관 0.963으로 다른 오차 패턴 생성 (단독 OOF 3,561이지만 다양성 가치 있음)

---

## 🎯 Q1 답변: ExtraTrees 원리 확장

### 진단: 왜 ExtraTrees만 성공했나?

**ExtraTrees의 성공 원리**:
- 분할점 선택: **무작위** (vs GB의 최적 분할점)
- 결과: CB/LGB와 **완전히 다른 오차 패턴**
- 상관 0.963 (매우 낮음) → 앙상블에서 높은 가치

---

### 같은 원리의 확장: 2가지 축

#### **축 1: LGBM `extra_trees=True` 옵션** ✅ **우선순위 1** (3시간)

ExtraTrees의 "무작위 분할" 원리를 native LGBM에 적용:

```python
# 변형 1: 약한 다양성
LGBM(
    extra_trees=True,      # 분할점 무작위 선택
    max_depth=8,
    num_leaves=64,
    feature_fraction=0.8,
    seed=42
)

# 변형 2: 강한 다양성
LGBM(
    extra_trees=True,
    max_depth=10,
    num_leaves=128,
    feature_fraction=0.7,
    seed=42
)
```

**기대효과**: +1~3점  
**검증**: OOF 상관 < 0.97이면 기대 높음

---

#### **축 2: KernelRidge (선형 모델의 비선형화)** ✅ **우선순위 2** (2시간)

선형 Ridge를 고차원 feature space로 변환 → 트리 기반과 완전히 다른 구조:

```python
from sklearn.kernel_ridge import KernelRidge

# RBF 커널
kr = KernelRidge(kernel='rbf', alpha=1.0, gamma=0.1)

# 또는 Poly 커널
kr = KernelRidge(kernel='poly', degree=3, alpha=1.0)
```

**기대효과**: +0~2점  
**검증**: 상관만 < 0.97이면 OK (RMSE 형편없어도)

---

### 실행 순서 (병렬 가능)

```
Step 1: LGBM extra_trees 2~3종 CV (3시간)
Step 2: KernelRidge 3종 CV (2시간) [병렬]
Step 3: 상관 < 0.97 모델 필터링
Step 4: 앙상블 재구성 (전략 46)
Step 5: 검증 & 제출 (6/28~6/29)
```

---

## 🎯 Q2 답변: 1~3위 격차 분석

### 격차 추정 (2~5점)

상위권이 우리가 놓친 부분:

#### **A. 더 많은 모델 다양성**
우리: CB, LGB, ExtraTrees 3종류  
상위권 추측: CB, LGB, XGB, ET, KernelRidge, SVR, Ridge-poly 등 5~6종

#### **B. 구간별 가중치 최적화** ⭐ (우리가 놓친 부분)
- 고가(>50 percentile) vs 저가: 다른 가중치
- 대형(>100m²) vs 소형: 다른 가중치
- **구별 Ridge 가중치 독립 계산**
- **기대효과**: +2~5점 (가장 큰 개선폭)

#### **C. 합성 데이터 특성 역이용**
- Price 분포 패턴
- Correlation 구조 최적화
- Heteroscedastic noise 모델링
- **기대효과**: +1~2점

---

### 우리 놓친 부분: 구간별 오차 분석 ✅

**즉시 실행 (1시간)**:
```python
# Q: ET 2개가 정말 유효한가? 특정 구간에만?

# 1) 고가 구간(Price > 50 percentile) 분석
rmse_high_et = RMSE(pred_et[price > p50], y[price > p50])
rmse_high_cb = RMSE(pred_cb[price > p50], y[price > p50])

# 2) 대형 구간(Area > 100m²) 분석  
rmse_large_et = RMSE(pred_et[area > 100], y[area > 100])

# 3) 구별 성능 비교
for gu in gus:
    print(f"{gu}: RMSE비율 {rmse_et[gu]/rmse_cb[gu]:.3f}")
```

**발견 가능한 패턴**:
- "고가만 효과" → 가중치 재설계
- "전체 균등" → 현 전략 정상

---

## 🎯 Q3 답변: Final Submission 구성

### 최종 권장안

#### **Final 1 (공격형): 전략 45 그대로** ✅
```python
pred_final1 = pred_strategy_45  # 2,094.9
```

#### **Final 2 (방어형): 45:28 = 70:30 블렌딩** ✅
```python
pred_final2 = 0.7 * pred_45 + 0.3 * pred_28

# 이유:
# - 45의 공격성 유지 (70%)
# - 28의 보수성으로 리스크 흡수 (30%)
# - Private 변동성 완화
# - Gemini & ChatGPT 권장과 일치
```

---

## 📅 실행 계획

### **Week 1 (6/27~6/29)**

```
6/27-28:
  - LGBM extra_trees 2종 검증
  - KernelRidge 검증
  - 상관 확인

6/29: 제출 1
  - 46 또는 45 제출 (검증 결과에 따라)
```

### **Week 2 (6/30~7/2)**

```
6/30:
  - 구간별 오차 분석
  - 가중치 조정 (선택)

7/1-7/2: 제출 2
  - 45:28=70:30 또는 신규 전략
```

### **Week 3+ (7/3~7/8)**

```
7/3~7/7: 관찰 기간
  - Public LB 모니터링
  - 최종 의사결정 준비

7/8: 최종 제출 (2개)
  - 09:00: Final 1 (45)
  - 17:00: Final 2 (45:28 또는 신규)
```

---

## 핵심 체크리스트

- [ ] LGBM extra_trees 2~3종 CV (3시간)
- [ ] KernelRidge CV (2시간)
- [ ] 상관 < 0.97 필터링
- [ ] 구간별 오차 분석 (1시간)
- [ ] 앙상블 재구성 (전략 46)
- [ ] OOF/OOT 검증
- [ ] 6/29 제출 1, 7/1~2 제출 2

---

## 📊 기대 개선폭

| 항목 | 기대 | 우선순위 | 시간 |
|------|------|--------|------|
| LGBM extra_trees | +1~3점 | 1 | 3h |
| KernelRidge | +0~2점 | 2 | 2h |
| 구간별 분석 | 근거 수집 | 2 | 1h |
| **합계** | **+1~5점** | | 6h |

**최종 목표**: 2,094.9 → 2,090 이상 도달

---

## 최종 메시지

**4위(2,094.9) → 상위권 진입의 경로**:

1. **ExtraTrees 원리 확장** (LGBM-ET + KernelRidge)
   - 검증: 5시간 내 완료
   - 기대: +1~3점
   - 확률: 중상

2. **구간별 오차 분석** (고가/대형 특화)
   - 기대: +0~3점 추가
   - 확률: 중

3. **Final 전략** (공격+방어 조합)
   - Final 1: 45 (공격)
   - Final 2: 45:28=70:30 (방어)

---

**작성**: Copilot (2026-06-27)  
**신뢰도**: 중상 (ExtraTrees 성공 재현 가능성)  
**주의**: Private LB 변동성은 예측 불가. 최악의 경우 45 유지로 방어 필요

