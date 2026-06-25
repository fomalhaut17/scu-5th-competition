# 제5회 인공지능 모델링 경진대회 - Opencode 전략 제안

> GUIDE.md, chatGPT-Plan.md, gemini-Plan.md, results.csv, kaggle_notebooks/, strategies/ 전체 검토 후 작성

---

## 1. 현황 요약

| 항목 | 값 |
|------|-----|
| 최선 Public | **2,155** (전략 08: Ridge stacking + GTR) |
| 최신 OOF | **2,215** (전략 26: PL2 + Scale Blend, 미제출) |
| 잔여 기간 | **13일** (∼2026-07-08) |
| 일 제출 한도 | 5회 |

### 리스크 인지

ChatGPT/Gemini 모두 지적했듯이, 전략 26의 OOF 2,215는 **transductive leakage**로 인해 40~50점 낙관적일 가능성이 높다. Public에서 08을 크게 이길 것이라고 기대하지 말 것.

---

## 2. 차별화 전략 (ChatGPT/Gemini 미커버)

### 전략 A: 구별(Gu) 가중치 최적화 ⭐

**현재 문제**: 모든 구에 동일한 앙상블 가중치 사용 중.

| 구 | OOF RMSE | 비중(건수) | 특성 |
|------|---------:|----------:|------|
| Eunpyeong | 1,542 | ~? | 저가, 예측 잘됨 |
| Gangnam | 2,299 | ~? | 고가, 과소예측 |
| Seongdong | 2,880 | ~? | 최악, 변동성 큼 |

RMSE 차이가 2배에 가까운데도 `cb_log: 0.2, cb_raw: 0.3, lgb_log: 0.2, lgb_raw: 0.3`의 **전역 가중치**를 모든 구에 동일하게 적용 중.

**제안**: 4개 모델 OOF 예측값을 구별로 나누어 Ridge 회귀 또는 Optuna로 **구별 최적 가중치** 탐색.

```
방법:
1. 각 모델의 OOF 예측과 Test 예측을 Gu별로 그루핑
2. Gu별로 최적 앙상블 가중치를 별도 최적화
3. Gu별로 다른 가중치로 Test 예측 결합
```

**기대 효과**: 과소예측이 심한 Gangnam/Seongdong은 raw 모델 비중을 높이고, 안정적인 Eunpyeong/Songpa는 가중치를 다르게 주어 정밀도 향상.

**구현 난이도**: ★☆☆☆☆ (단순, 수십 줄 수정)

---

### 전략 B: 평당가 예측 → 면적 환산 (Unit Price Modeling) ⭐⭐

**현재 문제**: 고가/대형 아파트 과소예측이 전체 RMSE를 악화. 120㎡ 이상이 오차 상위 11% 차지.

**제안**: `Target` 대신 `Target / Exclusive_Area` (평당가)를 예측 목표로 삼은 후 면적을 곱해 최종 가격 산출.

```
Train: unit_price = Target / Exclusive_Area
모델 학습: Y = unit_price (log 변환 or raw)
Test 예측: unit_price_pred * Exclusive_Area → 최종 가격
```

이 방식의 장점:
- 면적 효과를 자연스럽게 분리하여 대형/소형 간 스케일 차이 제거
- 고가 과소예측 완화 (모델이 면적 크기에 덜 의존)
- 구/동별 평당가 패턴이 가격 패턴보다 안정적

**주의**: 면적-가격 관계가 완전 선형이 아니라서 보정 계수(`Area^0.9` 등) 실험 필요.

**구현 난이도**: ★★☆☆☆ (전처리 및 마지막 환산만 수정)

---

### 전략 C: PL 신뢰도 임계값 + 순차적 확장 ⭐⭐

**현재 문제**: PL2의 50% 임계값이 임의적. PL1(19)은 Public 2,173으로 실패.

**제안**: 단일 임계값 대신 **3단계 순차적 PL 확장** (Curriculum Self-Training):

```
Round 1: 상위 30% PL 추가 → 모델 재학습
Round 2: 상위 50% PL 추가 (Round 1 모델로 재예측 후) → 재학습
Round 3: 상위 70% PL 추가 → 최종 학습
```

각 라운드마다 신뢰도를 **재측정**하여 점진적으로 데이터를 늘리는 방식. 한 번에 50%를 추가하는 현재 방식보다 오차 누적 위험 감소.

**혹은 단순 버전**: 10%~90%까지 10% 단위로 임계값을 바꿔 OOF 민감도 체크 후 최적 선택.

**구현 난이도**: ★★★☆☆ (루프 감싸기)

---

### 전략 D: TimeSeriesSplit OOF 앙상블 (시드 앙상블 대체) ⭐⭐⭐

**현재 문제**: Random KFold 1개만 사용 중. 분산보다 편향이 주요 문제지만, OOF-불일치가 존재 (OOF 개선이 Public 개선을 보장하지 않음).

**제안**: KFold 시드만 바꾼 5개 모델 앙상블 대신, **Expanding Window + Rolling Window** 시계열 CV 2종을 추가로 구성해 3개 CV 전략의 예측 평균:

| CV 유형 | 용도 |
|---------|------|
| Random 5-Fold (seed 42) | 기존 유지 |
| Expanding Window (5 splits) | 시간 순서 보존 |
| Blocked TimeSeries (3 splits) | 월 단위 블록 |

각 CV에서 얻은 Test 예측을 단순 평균하여 **분산 감소 + OOF 신뢰도 향상**.

**구현 난이도**: ★★☆☆☆ (KFold 대체만 하면 됨)

---

### 전략 E: 고가 구간 사후 보정 (Post-hoc Calibration) ⭐

**현재 문제**: 고가 과소예측이 체계적 편향. 오차 상위 16건 중 62.5%가 6~8억 구간.

**제안**: OOF 오차를 분석해 **가격 구간별 보정 계수** 산출 후 Test 예측에 적용.

```
1. OOF 예측값을 가격 구간(bin)으로 나눔 (e.g., 2억 단위)
2. 각 구간의 평균 오차(=실제 - 예측) 계산
3. 과소예측 구간(6억~10억)에 양의 보정 추가
4. Test 예측에도 동일 구간 보정 적용
```

**단순화 버전**: `final_pred + error_bin_mean[bin]` 형태.

**구현 난이도**: ★☆☆☆☆

---

### 전략 F: 외부 실거래 데이터 활용 재시도 (핵심 재정의) ⭐⭐⭐

**현재 문제**: 외부 데이터(12,246건)는 트렌드 보정용으로는 실패했지만, **학습 데이터 증강** 용도로는 시도되지 않음.

**제안**: 외부 데이터를 트렌드 보정이 아닌 **사전학습(pre-training) 데이터**로 활용:

```
1. 외부 데이터 (docs/data/external_transactions.csv) 로드
2. 대회 데이터와 동일한 피처 엔지니어링 적용
3. 외부 데이터로 CatBoost/LightGBM 사전학습 (낮은 iteration)
4. 대회 데이터로 fine-tuning (전이 학습)
```

또는 **가중치 초기값(source)**으로 사용 후 대회 데이터에서 추가 학습.

**리스크**: 외부 데이터의 분포 차이 (모든 아파트 vs 샘플). DA(domain adaptation) 가중치가 이미 시도되었으나(failed), 사전학습 접근은 다름.

**구현 난이도**: ★★★★☆

---

## 3. 제출 우선순위 추천 (ChatGPT와 부분적 차이)

ChatGPT의 우선순위(26C → 25 → 24 → 26B → 블렌드)는 합리적이나, 아래 조정 제안:

| 순위 | 전략 | 사유 |
|:---:|:-----|:-----|
| 1 | **26C 재현** | 08 대비 평균 차이 -30으로 가장 안정적, PL효과 테스트 필요 |
| 2 | **전략 08 재제출** | 현재 1위이므로 방어 제출이 우선. 08 100%를 한 번 더 제출해 순위 변동 확인 |
| 3 | **A단순 + 5시드 앙상블** | 전략 08의 base CB+LGB를 5개 시드로 앙상블 → 구현 0, 리스크 0 |
| 4 | **A(구별 가중치)** | OOF 개선 확인 후 제출. 가장 안전한 신규 방향 |
| 5 | **E(구간 보정)** | 구현 즉시. 08 예측에 덧붙여 제출 가능 |

PL 계열(26)은 리스크가 크므로 **제출 5회 중 1~2회만 할당**할 것.

---

## 4. 즉시 실험 가능한 코드 (A안: 구별 가중치)

전략 25에 다음 코드 블록을 추가하는 것으로 구현 가능:

```python
# Gu별 가중치 최적화 (Optuna)
import optuna

gu_list = train_p['Gu'].unique()
gu_weights = {}

for gu in gu_list:
    gu_mask = (train_p['Gu'] == gu).values
    gu_oof = {k: oof[k][gu_mask] for k in ['cb_log','cb_raw','lgb_log','lgb_raw']}
    gu_y = y_true[gu_mask]

    def objective(trial):
        w = [trial.suggest_float(f'w_{k}', 0, 1) for k in ['cb_log','cb_raw','lgb_log','lgb_raw']]
        w = np.array(w) / sum(w)
        pred = sum(w[i] * gu_oof[k] for i, k in enumerate(['cb_log','cb_raw','lgb_log','lgb_raw']))
        return np.sqrt(np.mean((pred - gu_y) ** 2))

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)
    best_w = [study.best_params[f'w_{k}'] for k in ['cb_log','cb_raw','lgb_log','lgb_raw']]
    best_w = np.array(best_w) / sum(best_w)
    gu_weights[gu] = best_w

# Gu별 Test 예측 결합
final_pred = np.zeros(len(test))
for gu in gu_list:
    gu_test_mask = (test_p['Gu'] == gu).values
    w = gu_weights[gu]
    for i, k in enumerate(['cb_log','cb_raw','lgb_log','lgb_raw']):
        final_pred[gu_test_mask] += w[i] * test_pred[k][gu_test_mask]
```

---

## 5. 실패 패턴 요약 (시도 금지)

| 방향 | 실패 사유 |
|:-----|:----------|
| 동별/EWM/외부 트렌드 보정 | 08 단순 GTR을 이기지 못함 |
| XGBoost/MLP 추가 | 과적합 or 폭발 |
| 복잡한 메타 모델 (CatBoost meta) | 소규모 데이터 과적합 |
| Huber/Sample Weight | CB 폭발, 효과 없음 |
| 지리 피처 | Dong 범주형과 중복 |
| Pseudo Label v1 (19) | Public 2,173 확인 완료 |

---

## 6. 핵심 조언

1. **OOF를 맹신하지 말 것** — OOF 2,215 vs 08의 2,234는 19점 차이지만, Public에서는 5~10점 차이로 좁혀질 가능성 높음.
2. **제출 슬롯을 방어와 공격으로 나눌 것** — 하루 5회 중 2회는 08 계열 방어, 3회는 신규 전략 테스트.
3. **가장 단순한 변경부터 시도할 것** — 복잡한 파이프라인 변경보다 구별 가중치(전략 A)나 구간 보정(전략 E)이 구현 시간 대비 기대 효용 높음.
4. **리더보드가 최종 판단** — OOF 2,215인 26이 Public에서 2,160 이하면 PL 계열 중단.
5. **잔여 13일의 현실적 목표** — Public 2,140~2,145 달성 시 실질적 1위 가능.
