# AI 조언자 질문 (2026-06-28)

## 답변 작성 방법

조언자에게 이 파일을 보여주고 아래 규칙대로 작성을 요청하세요.

1. **이 파일 맨 아래 → 본인 이름 답변란**에 요약을 작성합니다.
   - 1~3줄 요약 (핵심 제안만)
   - 상세 답변이 있으면 `조언자이름-Plan.md` 파일 링크 추가
2. **상세 답변**은 리포 루트에 `조언자이름-Plan.md` 파일로 작성합니다.
   - 예: `gemini-Plan.md`, `chatGPT-Plan.md`, `coPilot-Plan.md`, `opencode-Plan.md`
   - 기존 파일이 있으면 **내용을 교체** (날짜별 최신 답변으로 유지)
3. **답변 참고 자료**: `GUIDE.md` (전체 현황), `results.csv` (실험 이력), 이 파일의 질문 내용

답변 예시:
```
#### Gemini
- [상세 답변은 gemini-Plan.md를 참조하십시오.](../gemini-Plan.md)
- **요약**: (1~3줄 핵심 제안)
```

---

## 상황

서울시 아파트 실거래가 예측 대회 (회귀, RMSE) 참가 중입니다.
- 데이터: Train 1,969건 (2024~2025년) / Test 531건 (2026년 1~3월), 피처 10개
- 데이터는 100% 합성(synthetic)이며, 실거래 데이터와 매칭 불가
- 현재 6위 (Public RMSE 2,086.6), 마감 07-08

## 리더보드 (방금 갱신)

| 순위 | 점수 | 제출수 | 비고 |
|------|------|--------|------|
| 1 | 0.0 | 2 | 정답 유출 (출제자 추정) |
| 2 | 2,024 | 17 | |
| **3** | **2,027** | **35** | **오늘 2,073→2,027로 46점 대폭 개선!** |
| 4 | 2,057 | 32 | 오늘 2,074→2,057로 17점 개선 |
| 5 | 2,073 | 35 | |
| **6(나)** | **2,087** | **29** | |

**3위가 오늘 46점을 한번에 개선** → 큰 개선이 가능한 방법이 존재한다는 증거

## 현재 파이프라인

```
[피처] Exclusive_Area, Floor, Distance_to_Subway, Brand_Apartment, Nearby_Parks,
       Gu, Dong, Transaction_YearMonth
     + FE: Age, YearMonth_Seq, Area×Floor, Floor/Area, Brand×Area

[12모델] CB_log, CB_raw, LGB_log, LGB_raw         (RMSE, 가격 직접)
         CB_UP_log, CB_UP_raw, LGB_UP_log, LGB_UP_raw  (RMSE, 평당가=가격/면적)
         ET_log, ET_raw                              (ExtraTrees, 랜덤분기)
         LGBET_log, LGBET_raw                        (LGB extra_trees, GB+랜덤분기)

[메타] Ridge 스태킹 (alpha=100, 5-Fold CV)

[PL2] 4모델 Stage1 → 신뢰도 상위 50% 테스트 데이터를 pseudo label → 재학습

[최종] 전략53(PL2없음, 4시드 평균) × 80% + 전략47(PL2) × 20% → 구별 트렌드 보정(GTR)
```

## 오늘 시도하고 실패한 것들 (전략 57~60)

### 핵심 발견: 성능-다양성 딜레마

**스태킹에 기여하는 모델 조건**: 단독 OOF < 4,000 AND GBDT 대비 상관 < 0.96.
현재 유일하게 이 조건을 만족하는 비트리 모델은 ExtraTrees뿐.

| 시도 | 단독 OOF | GBDT 상관 | 스택 효과 | 실패 원인 |
|------|----------|----------|----------|----------|
| MAE loss (CB/LGB) | 2,278 | 0.998 | +0.2 | 같은 트리, loss만 달라도 같은 예측 |
| ET/LGBET unit_price | 3,800+ | — | +1.3 | 너무 약함 |
| Weighted PL2 (confidence weight) | — | — | +19 | confidence 변별력 없음 |
| DART booster (LGB) | 28,043/2,643 | 0.21/0.995 | +7.2 | log에서 폭발, raw에서 동일 |
| BoxCox 타겟 (λ=0.84) | 2,253 | 0.999 | +1.8 | raw와 사실상 동일 |
| 기본 MLP (128-64-32) | 5,720 | 0.81 | +2.1 | 다양하지만 너무 약함 |
| **Entity Embedding MLP** | **2,723** | **0.98** | **+2.3** | **정확하면 GBDT와 수렴** |

**결론**: 같은 피처에서 모델이 잘 맞출수록 GBDT와 같은 예측에 수렴. 약한 모델은 다양하지만 노이즈.

## 이전에 소진된 방향 (전략 01~51)

- 트렌드 보정: 구별 단순평균이 최적, α=1.0 확정 (동별/EWM/외부/α sweep 전부 실패)
- 외부 데이터: 합성 데이터라 실거래 매칭 불가
- 메타 모델: Ridge가 최적, CatBoost/LGB/ElasticNet 메타 전부 동일 or 악화
- 타겟 변환: log/raw/unit_price만 유효, sqrt/층당가/면적층가 실패
- 피처: 상대피처/TE/지리/동별가격/confidence 전부 트리가 이미 학습 중
- 모델: RF/XGB/SVR/KNN/Ridge-poly/MLP 추가 → 다양성 부족
- 2단 스태킹, 반복PL, 구별 분리, 멀티시드(PL2 있을 때) → 전부 악화
- Residual modeling, sample weight, kNN correction, domain adaptation → 전부 실패

## 질문

**3위가 2,073→2,027로 46점을 단번에 개선했습니다. 우리도 2,087에서 비슷한 점프를 하려면 어떤 방향이 가능할까요?**

우리가 아직 시도하지 않았거나, 시도했지만 다른 방식으로 접근할 수 있는 방법을 제안해주세요.

참고:
- 데이터가 합성이므로 실거래 외부 데이터는 무의미
- Train→Test 시간 갭이 핵심 (2024~2025 → 2026)
- 하루 5회 제출 제한, 마감까지 10일
- 모델 다양성 추가는 거의 소진 (성능-다양성 딜레마)
- OOF와 Public의 갭이 큼 (OOF 2,191 → Public 2,087, ~100점 차이)

---

## 답변

#### Gemini
- [상세 답변은 gemini-Plan.md를 참조하십시오](../gemini-Plan.md)
- **요약**: 모델 다양성 추가의 한계를 인정하고 **'구조적 돌파구'**가 필요합니다. **1) 고오차 영역(대형/고가)에 가중치를 둔 Weighted Ridge 스태킹**, **2) 로그-선형 모델(Skeleton) + GBDT 잔차 모델의 하이브리드 구조**, **3) 모델 간 일치도가 극도로 높은 샘플만 사용하는 Consensus PL**을 제안합니다.

#### ChatGPT
- [상세 답변은 chatGPT-Plan.md를 참조하십시오](../chatGPT-Plan.md)
- **요약**: 46점 점프의 남은 가능성은 새 모델이 아니라 **Train→Test 시간 외삽 보정**에 있습니다. `전략56`을 고정 기준으로 두고 **OOT 기반 temporal adapter**, **log/unit-price/robust/recent GTR family**, **53:47 저차원 blend 탐색**을 우선 제안합니다.

#### opencode
- [상세 답변은 opencode-Plan.md를 참조하십시오](../opencode-Plan.md)
- **요약**: 성능-다양성 딜레마로 모델 구조는 최적점에 도달. 46점 점프의 유일한 설명은 **시간 외삽(Time Extrapolation) 또는 Test 분포 매칭**. **1) Gu별 선형 Trend Skeleton + GBDT 잔차의 Time-Aware Decomposition**, **2) 61a(70:30)/61b(90:10) 즉시 제출로 blend 최적화**, **3) OOF-Public 100점 갭 원인 분석 → Test 분포 매칭**을 제안.
