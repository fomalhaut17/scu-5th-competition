# coPilot-Plan (2026-06-25 UPDATE)

## 📍 현황 (전략 28 기준)

- **기준점**: 전략 28 `PL2+평당가 8모델` (OOF 2,196 / **Public 2,096.8, 1위**)
- **달성**: Copilot 제안 이후 1위 탈환 성공
- **핵심 성과**:
  - PL2 신뢰도 필터(50%) + 데이터 증강: 21점 개선 (OOF)
  - 평당가 크로스 블렌딩: 53점 개선 (Public)
  - Ridge 스태킹: 8모델 최적 조합

---

## 🎯 다음 단계 전략 (우선순위 순서)

### **우선순위 1: 고가 구간 사후 보정 (Post-hoc Price-Level Calibration)**

**배경**: 
- Error Analysis: 6~8억 고가 구간의 오차 비율 62.5% (초과대표)
- 강남구, 성동구, 용산구의 RMSE/평균가 > 6.7%
- 로그 스케일 모델의 고가 과소예측 경향

**방법**:
1. OOF 잔차 분석 → Target 4분위수별(저가/중가/고가/초고가) 보정 계수 계산
2. 구별 × 가격 구간 세밀한 보정 행렬(8구 × 4가격등급 = 32개 파라미터) 구축
3. Ridge 예측 후처리: `pred_final = pred_ridge * (1 + calibration[gu][price_level])`
4. Validation set에서 최종 강도 결정(0.5~1.5 범위 제약)

**장점**:
- 추가 모델 불필요 → leakage 최소화
- 기존 pipeline 영향 없음 → 안정적
- 구별 약점 구(성동/용산) 집중 개선 가능

**기대 효과**: 1~3점 개선 (보수적), 5~10점 (낙관적)

---

### **우선순위 2: 구별 가중치 최적화 (Per-District Ridge Weights)**

**배경**:
- 전역 Ridge 가중치: 모든 구에 동일 적용
- 구별 성능 편차: 은평구 1,542 vs 성동구 2,880 (1,338점 차이)
- 일부 구는 특정 모델(예: CB raw)이 더 우수할 가능성

**방법**:
1. OOF를 8개 구별로 분할
2. **각 구별로 8모델의 Ridge 가중치 독립 계산**
3. Test 예측: 구 정보로 해당 가중치 적용 (예: `pred[gu==강남] = Ridge_강남.predict(...)`)
4. 구별 Ridge vs 전역 Ridge: 5-fold 교차검증으로 안정성 비교

**장점**:
- 성동구/용산구 같은 약점 구 집중 개선
- Ridge 계산 비용 미미 (8배, 계산량 무시할 수준)

**기대 효과**: 3~5점 개선 (보수적), 5~15점 (낙관적, 약점 구 집중 개선 가능)

---

### **우선순위 3: 시드 다양성 스택 (Seed Diversity Ensembling)**

**배경**:
- 기존 8모델: 모두 같은 구조, 같은 seed(42)
- 편향 누적 → 분산 감소 효과 미미
- 트리 모델 다양성 추가(XGB)는 실패했지만, 같은 모델의 시드 변형은 미시도

**방법**:
1. 기존 8모델 구조 유지 (CB log/raw, LGB log/raw, 평당가×4)
2. **다른 random_seed로 재학습**: seed 128, 256, 512 → 총 32모델
3. 상관계수 > 0.98인 모델 제거(중복 필터)
4. 최종 ~24모델 → Ridge 재스태킹

**장점**:
- 검증된 다양성 축 (이미 log/raw 분화 성공)
- 편향-분산 트레이드오프 개선 가능

**기대 효과**: 
- 보수적: 2~5점 (하지만 OOF 개선 ≠ Public 개선)
- 낙관적: 5~10점 (분산 감소)
- **위험**: OOF와 Public 갭이 100점인 상황에서 과신 금지

---

### **우선순위 4 (선택): PL2 임계값 미세 조정 (Threshold Sweep)**

**배경**:
- 현재 신뢰도 상위 50% 고정
- 최적값이 40%, 45%, 55%, 60%일 가능성

**방법**:
1. 각 임계값 시도 (40~60%, 5% 단위)
2. OOF + OOT(시간 holdout) 검증
3. 보수적 선택: OOT RMSE 최소값

**기대 효과**: 1~3점 (하지만 이미 충분히 탐색했을 가능성)

**제약**: 시간 효율성 고려하여 1순위~3순위 모두 효과 없으면 시도

---

### **우선순위 5 (실험적, 신중): 혼합 정밀도 피처링 (Mixed-Precision Scaling)**

**아이디어** (재검토):
- 기존: log / raw 이분화
- 제안: sqrt(Target), Target^0.3 중간 스케일 추가
- 고가에서는 약한 스케일이 과소예측 개선 가능

**제약**: 전략 29에서 sqrt 실패 → 매우 신중할 것, 초기에는 시도 금지

---

## ⚠️ 금지 방향 (반복 금지)

- ❌ 트렌드 보정 변형 (동별, EWM, 중앙값 등)
- ❌ 메타 모델 복잡화 (CatBoost/MLP)
- ❌ Residual Modeling, 피처 변형, 반복 PL
- ❌ 외부 실거래 데이터, XGBoost/MLP 추가
- ❌ 추가 타겟 변환 (층당가, sqrt, 면적층가 — 전략 29~31에서 모두 실패)

---

## 📋 구현 로드맵

### **Phase 1 (Week 1): 우선순위 1~2**
```
Task 1: 고가 보정(Post-hoc) 구현 & OOF/OOT 검증
- kaggle_notebooks/32_price_level_calibration.py 작성
- 구간별 보정 계수 저장 & 적용 로직
- 기대: 1~3점 개선 재현

Task 2: 구별 Ridge 가중치 구현 & 교차검증
- kaggle_notebooks/33_per_gu_ridge.py 작성
- 8개 구별 가중치 독립 계산 & test 적용
- 기대: 3~5점 개선 재현

Phase 1 제출 전략 (5회):
  - 32 (고가보정 단독)
  - 32+33 블렌드 (70:30)
  - 32+33 블렌드 (50:50)
  - 28 (방어, baseline 유지)
  - 32+33+28 삼중 블렌드 (선택)
```

### **Phase 2 (Week 2): 우선순위 3**
```
Task 3: 시드 다양성 스택 구현
- kaggle_notebooks/34_seed_diversity.py 작성
- seed 128, 256, 512로 8모델 재학습 → 상관계수 필터
- 기대: 2~5점 개선 (낙관적: 5~10점)

Phase 2 제출 전략 (5회):
  - 34 (시드 다양성 단독, 심사숙고 후 제출)
  - 32+34 블렌드
  - 33+34 블렌드
  - 28 (방어)
  - 기타 블렌드
```

---

## 🔍 검증 강화

**모든 새 전략에 필수 적용**:
1. OOF RMSE (5-fold) + OOT RMSE (2/3/4개월 holdout 비교)
2. Gu별 RMSE 분석 → 약점 구 개선 확인
3. 고가/저가 구간별 성능 비교
4. Train/Test 통계 확인 (mean/std 이상 편차 < 0.5%)
5. **제출 전 sanity gate**: 
   - OOF 개선 AND OOT RMSE 개선 동시 확인
   - 한쪽만 개선된 경우 신중한 판단 필요

---

## 운영 체크리스트

- [ ] Phase 1: 고가보정 + 구별가중치 구현 (이번 주)
- [ ] 각 전략 로컬 실행 → 로그 기록
- [ ] results.csv 업데이트 (제출 후 public_rmse 기입)
- [ ] 주간 요약: 각 우선순위별 기대효과 vs 실제 효과 비교
- [ ] OOF 낙관성 재확인 (100점 갭 주의)

---

## 📊 성공 기준

- **최소 목표**: 28(2,096.8) 유지 또는 소폭 개선 (2,090 이상)
- **목표**: 2,080~2,090 (1~2위 유지)
- **공격 목표**: 2,070 이하 (상위권 확보)

---

**작성자**: Copilot (전략 파트너)
**마지막 업데이트**: 2026-06-25
**참고**: GUIDE.md (현황 기록), 전략 1~34 실험 결과

---

Implementation suggestions for kaggle_notebooks/26_pl2_scale_blend.py
- Leakage guard: implement outer-5fold PL protocol (outer train → generate PL only from outer-train models → attach PL to outer-train only) and compute conservative OOF from outer validations.
- PL selection rule: top 50% by ensemble agreement (low model variance) AND low fold-wise std; record PL count.
- Targets & meta: train stage models and meta (Ridge) on original target scale; only use np.log1p models when required and revert with np.expm1 before stacking.
- Reproducibility: set seeds, save model artifacts, OOF/test preds (CSV), and random_state in CV.
- Stability checks: log train mean/std vs test pred mean/std, per-Gu mean shifts, and OOF vs OOT (2/3/4 months).
- Runtime safety: use early_stopping_rounds, moderate iterations, and verbose; avoid huge num_boost_round on Kaggle.
- Categorical handling: CatBoost use native; for LightGBM pass pd.Categorical or fallback to one-hot if exploding.
- Sanity gates before submit: require conservative outer-fold OOF improvement AND no large mean shift (>0.5% relative) vs 08 baseline; otherwise prefer defensive 08/26 blend.

These are concise checks Claude should apply while implementing the script.
