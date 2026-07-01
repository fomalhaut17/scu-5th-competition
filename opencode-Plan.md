# opencode Plan — 2위 추월을 위한 구조적 돌파구

**작성일**: 2026-07-01 | **목표**: 2위 RMSE 2,000.87 추월 (현재 2,028.35, 갭 27.48점, 7일)

---

## 진단: 지금까지의 실패 패턴

| 실패 유형 | 원인 | 사례 |
|-----------|------|------|
| **검증셋 과적합** | OOT 246건으로 신규 컴포넌트 가치 판단 불가 | 전략76 (4-way OOT 개선 but Public 악화) |
| **OOF 낙관성** | in-sample OOF가 Public과 다른 방향 | 전략73 (log_Area), 전략74 (PL2 증강) |
| **성능-다양성 딜레마** | 약한 모델은 다양하지만 노이즈, 강한 모델은 GBDT와 수렴 | 전략44~60 전반 |

**핵심**: 기존 파이프라인(GBDT 12모델 + Ridge + GTR)은 이미 지역 최적점. 점진적 튜닝으로는 2,000.87 도달 불가.

---

## 제안 1: DGP 수식 역공학 (Symbolic Regression) ⭐ 최우선

**근거**: 데이터는 100% 합성. 피처 10개로 2,000개 샘플 생성 → **결정론적 함수 + 노이즈** 구조일 가능성 90%+

| 정황 | 해석 |
|------|------|
| 피처 10개만으로 OOF 2,196 달성 | DGP가 비교적 단순한 함수일 가능성 |
| Linear R²=0.939, RMSE 2,301 | 선형으로도 93.9% 설명 가능 |
| Tree 모델 OOF 2,191 vs Linear OOF 5,765 | **강한 비선형성 존재** (트리만 포착) |
| Ridge Skeleton OOF 2,705 (log-price) | 로그-선형 스켈레톤이 상당 부분 설명 |

### 실행 계획

```python
# 1단계: 현재 피처로 Symbolic Regression (PySR 사용)
# DGP 후보: price = f(Gu, Dong, Area, Floor, Year_Built, YearMonth) + ε
# f는 곱셈/가산 구조일 가능성 높음 (전략51에서 곱셈 구조 확인됨)

# 2단계: 발견된 수식으로 직접 예측
# → Symbolic 예측을 추가 컴포넌트로 블렌딩

# 3단계: 발견된 수식의 잔차를 기존 GBDT로 재학습
```

**단계별 접근**:
1. **PySR 설치** (`pip install pysr`) → 로컬에서 신속 실행
2. 피처 10개 + 목표변수(log price)로 Symbolic Regression
3. 단순도-정확도 파레토 프론티어에서 가장 단순한 수식 선택
4. 발견된 수식 예측값을 pred_78로 등록 → 56+63+69+78 4-way 블렌딩

**기대 효과**: DGP의 핵심 구조를 포착하면 **한 번에 15~30점** 점프 가능

---

## 제안 2: Skeleton 다양화 앙상블 (10+ Skeletons)

현재는 Skeleton 2종(Per-Gu, One-Hot)만 사용. Skeleton은 단순해서 과적합 위험이 낮으므로 **대량 생산**이 가능.

### Skeleton 후보 (Ridge 기반, log-price)

| # | Skeleton 종류 | 설명 |
|---|---------------|------|
| 1 | **Per-Gu Ridge** (기존 63) | 구별 독립 Ridge |
| 2 | **One-Hot Ridge** (기존 69) | 전체 One-Hot |
| 3 | **Per-Dong Ridge** | 동별 독립 Ridge (27개 동) — 더 세분화 |
| 4 | **Per-Gu Lasso** | 구별 Lasso (α sweep) — 다른 정규화 |
| 5 | **Per-Gu ElasticNet** | 구별 ElasticNet — 또 다른 정규화 |
| 6 | **Gu×Time Ridge** | price ~ Gu * YearMonth_Seq — 구별 시간 추세 |
| 7 | **Area-binned Ridge** | 면적 구간별 (소/중/대/초대형) 독립 Ridge |
| 8 | **Floor-binned Ridge** | 층수 구간별 (저/중/고층) 독립 Ridge |
| 9 | **RBF Kernel Ridge** | 비선형 Skeleton |
| 10 | **Polynomial Ridge** (degree=2) | 다항 상호작용 Skeleton |
| 11 | **Robust Ridge** (Huber) | 이상치 덜 민감한 Skeleton |

### 실행 계획
1. 11개 Skeleton 각각 학습 → 각각 GBDT 잔차 학습 (6모델×4시드)
2. 11개 × 6×4 = 264개 예측 → Ridge로 스택 or 단순 블렌딩
3. OOT 검증으로 상위 Skeleton 선별 → 최종 블렌딩

**기대 효과**: Skeleton 다양성 확보 → GBDT 잔차의 출발점 다양화 → **5~10점** 개선

---

## 제안 3: 시간 외삽 재구조화 (Time Decomposition)

Train(2024~2025) → Test(2026 Q1)의 시간 갭이 가장 큰 도전과제.

### 문제 진단
- 구별 평균 보정(α=1.0)이 최선으로 확인됨 — 더 이상 개선 불가능
- 하지만 이는 "모든 아파트가 동일한 비율로 상승"한다는 가정 → **너무 단순함**

### 새로운 접근: 다차원 시간 트렌드

```python
# 1. 피처별 시간 추세 추정
for feature in ['Gu', 'Dong', 'Area_bin', 'Floor_bin', 'Brand']:
    # 각 그룹의 Train 기간 월별 평균가 추세 → 외삽
    # Test 월(202601~03) 예측값 = Train 최종값 × (1 + trend_rate)^months

# 2. 구×면적 세그먼트별 시간 트렌드
# → 강남구 대형은 많이 오르고, 노원구 소형은 적게 오르는 패턴
```

### 실행 계획
1. `Gu × Area_bin` 세그먼트별 월별 평균가 계산
2. 각 세그먼트의 시간 트렌드(기울기) 추정
3. Test 기간 예측 = Skeleton 예측 × 세그먼트별 트렌드 보정계수
4. 기존 GTR(α=1.0)을 세그먼트별 α로 대체

**기대 효과**: 단순 구별 평균보다 정교한 시간 보정 → **3~8점** 개선

---

## 제안 4: Test Set Feature Structure 활용 (Post-Processing)

Test 531건의 피처 분포를 Train과 비교하여 **체계적인 편향 보정**.

### 실행 계획
1. Test 피처로 Train 샘플의 가중치 재조정 (Kernel Density Matching)
2. Test 분포에 맞게 Train 예측 보정 (Distribution Alignment)
3. 특히 고오차 세그먼트(120㎡ 이상, 강남구, 성동구)에 **별도 보정계수** 적용

```python
# Test 집단별 분포 확인
test_groups = test.groupby(['Gu', 'Area_bin']).size()
train_groups = train.groupby(['Gu', 'Area_bin']).size()
ratio = test_groups / train_groups  # 과대/과소 대표 세그먼트 식별

# 과소 대표 세그먼트 → 해당 그룹 예측에 페널티 부여
# 과대 대표 세그먼트 → 해당 그룹 예측 가중치 상향
```

---

## 실험 우선순위 (7일 기준)

| 일차 | 실험 | 비용 | 기대효과 | 설명 |
|------|------|------|---------|------|
| **1일차** | Symbolic Regression (PySR) | 로컬 1시간 | **15~30점** | 최고 리스크-리턴, 바로 시작 |
| **1~2일차** | Skeleton 다양화 11종 | 로컬 2시간 | 5~10점 | 로컬에서 모두 실험 후 Public 확인 |
| **3일차** | Time Decomposition | 로컬 1시간 | 3~8점 | Skeleton과 병행 가능 |
| **4일차** | 최적 Skeleton 선별 + 블렌딩 | 제출 3회 | 5~15점 | 상위 실험 결과 취합 |
| **5~6일차** | Distribution Alignment | 로컬 1시간 | 2~5점 | 마무리용 |
| **7일차** | 최종 블렌딩 가중치 결정 | 제출 5회 | 0~3점 | 기존 OOT 검증 활용 |

---

## 검증 전략

OOT(246건)과 naive OOF 모두 한계를 보였으므로:

1. **Public LB를 최종 검증 기준**: 구조적 변경(제안 1, 2, 3)은 하루 1건씩 Public 확인
2. **내부 검증**:
   - **Time-series CV**: 3-fold 시간순 분할 (Train 내에서)
   - **Bootstrap OOT**: OOT 246건에서 Bootstrap 100회 → 분포 추정
   - **Adversarial Validation**: Train/Test 구분 모델의 AUC 변화 추적 (이미 0.505로 낮음)
3. **최종 선택**: 4개 제안 중 Public이 가장 개선된 조합 선택

---

## 리스크 관리

| 리스크 | 대응 |
|--------|------|
| Symbolic Regression이 유효한 수식을 못 찾음 | → Skeleton 다양화로 전환, 시간 낭비 최소화 (1일 제한) |
| Skeleton 다양화가 OOT는 개선 but Public 악화 (76번 반복) | → 각 Skeleton 단독 Public 확인 → 통과한 것만 블렌딩 |
| 2위도 계속 개선 중 | → 우리의 개선폭이 더 커야 함. 구조적 돌파구(제안1)가 유일한 해법 |
| 제출 횟수 부족 | → 로컬 검증 강화 (Bootstrap OOT), Public은 최종 확인용 |

---

## 결론

**가장 추천**: **제안 1 (Symbolic Regression DGP 역공학)** → 데이터가 100% 합성이므로 반드시 DGP 수식이 존재합니다. 27점 차이는 점진적 개선이 아닌 구조적 발견으로만 극복 가능합니다.

**차선**: **제안 2 (Skeleton 다양화)** → 과적합 위험이 낮은 Skeleton 계층에서 다양성을 확보하는 안전한 접근.

**두 제안은 병행 가능**: 1일차에 PySR 실행 (로컬, 1시간) + Skeleton 11종 동시 학습 → 이후 결과 취합.
