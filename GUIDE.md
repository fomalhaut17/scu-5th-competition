# SCU 5th AI Competition - 작업 가이드

## 대회 개요

- **주제**: 서울시 아파트 실거래가 예측 (회귀)
- **평가**: RMSE (낮을수록 좋음)
- **기간**: 2026-06-22 ~ 2026-07-08, 하루 5회 제출
- **플랫폼**: Kaggle (scu-5th-ai-competition)
- **데이터**: Train 1,969건 / Test 531건
- **핵심 특성**: Train은 2024~2025년, Test는 2026년 1~3월 → 시간 갭 존재

## 현재 최선 전략

**전략 08: STACK GU TREND** (Public RMSE **2,155**, 1위)

```
피처 엔지니어링 → CatBoost + LightGBM (5-Fold CV)
→ Ridge 스태킹 → 구별 트렌드 보정
```

- Kaggle 노트북: `kaggle_notebooks/08_stack_gu_trend.py`
- OOF RMSE: 2,234

## 프로젝트 구조

```
├── main.py                  # 원본 baseline (수정 금지)
├── utils.py                 # 공통 유틸 (전처리, CV, 평가, OOT 분할)
├── results.csv              # 전략 결과 + Kaggle 제출 이력
├── dashboard.py             # python dashboard.py 로 현황 조회
│
├── strategies/              # 레이어별 전략 (로컬 실험용)
│   ├── L1_features/         # 피처 확정 (모델 고정: LightGBM)
│   ├── L2_models/           # 모델 테스트 (피처 고정: L1 확정)
│   ├── L3_tuning/           # Optuna 하이퍼파라미터 튜닝
│   ├── L4_blending/         # 블렌딩/스태킹
│   └── _exceptions/         # 레이어 순서 무시 예외 테스트
│       └── oot_validation_test.py  # OOT 검증 스크립트
│
├── kaggle_notebooks/        # Kaggle 제출용 통합 스크립트
│   ├── 01_bl.py ~ 10_stack_gu_real_trend.py
│   └── logs/                # 로컬 실행 로그
│
├── docs/
│   ├── scu-5th-competition.xlsx  # 대회 안내 문서
│   └── data/                     # 외부 실거래 데이터 (공공데이터포털)
│       ├── *.xls                 # 8개 구 원본
│       └── external_transactions.csv  # 파싱 결과 (12,246건)
│
└── submissions.csv          # 최종 제출용 (Kaggle 업로드 파일)
```

## 레이어 파이프라인

각 레이어에서 최선을 확정하고 다음 레이어로 넘어가는 방식.
조합 폭발을 방지하고, 실험 추적이 쉬움.

### L1: 피처 확정 (모델: LightGBM 고정)

| 전략 | OOF RMSE | 결과 |
|------|----------|------|
| FE (피처 엔지니어링) | 2,403 | **확정** |
| TE (타겟 인코딩) | 2,445 | 탈락 |
| FE+TE | 2,446 | 탈락 |

확정 피처: YearMonth_Seq, Area_x_Floor, Floor_per_Area, Brand_x_Area

### L2: 모델 테스트 (피처: FE 고정)

| 전략 | OOF RMSE | 결과 |
|------|----------|------|
| CatBoost | 2,310 | **선발** |
| LightGBM | 2,403 | **선발** |
| Ridge | 5,765 | 선발 (다양성) |
| XGBoost | 2,644 | 탈락 |

### L3: Optuna 튜닝

| 전략 | OOF RMSE | 결과 |
|------|----------|------|
| CatBoost 튜닝 | 2,267 | **확정** |
| LightGBM 튜닝 | 2,298 | **확정** |
| Ridge 튜닝 | 5,765 | 탈락 |

### L4: 블렌딩/스태킹

| 전략 | OOF RMSE | 결과 |
|------|----------|------|
| CB+LGB 가중평균 (50:50) | 2,242 | 확정 |
| Ridge 스태킹 | 2,234 | **최종 확정** |

## Kaggle 제출 이력

### 2026-06-22 (5/5회)

| # | 전략 | OOF | Public | 교훈 |
|---|------|-----|--------|------|
| 01 | BL (보정 없음) | 2,247 | 2,537 | 시간 갭 확인, 보정 필요 |
| 02 | BL + 전체 트렌드 | 2,247 | 2,320 | 보정 효과 확인 |
| 03 | BL + 구별 트렌드 | 2,247 | 2,212 | 구별이 전체보다 나음 |
| 04 | BL + 동별 트렌드 | 2,247 | 2,383 | 과보정 |
| 05 | BL + 실제 상승률 | 2,247 | 2,335 | 과보정 |

### 2026-06-23 (5/5회)

| # | 전략 | OOF | Public | 교훈 |
|---|------|-----|--------|------|
| 06 | BL + 재튜닝 | 2,239 | 2,187 | 튜닝 효과 확인 |
| 07 | 3모델 블렌딩 | 2,237 | 2,192 | XGB 다양성 효과 없음 |
| 08 | **Ridge 스태킹** | **2,234** | **2,155** | **현재 최선** |
| 09 | EWM 트렌드 | 2,234 | 3,154 | 노이즈 증폭, 과보정 |
| 10 | 외부 데이터 보정 | 2,234 | 3,021 | 모집단 불일치 |

## 검증된 교훈

### 트렌드 보정 (실험 완료, 더 이상 시도 불필요)

- **구별 단순평균이 최적** — 동별, EWM, 중앙값, Cap, 외부 데이터 모두 악화
- 보정 강도에 매우 민감: 조금만 과하면 RMSE 1,000 이상 악화
- 부동산 트렌드 자체가 예측하기 어려움 (노이즈 >> 시그널)

### 모델 구조

- 트리 모델 다양성(XGB 추가)은 효과 없음
- Ridge 스태킹은 단순 가중평균 대비 확실히 효과 있음
- 5-Fold가 최적 (10-Fold, 시드 앙상블 효과 없음)
- 분산보다 편향이 주요 문제

### 외부 데이터 (실험 완료)

- 공공데이터포털에서 8개 구 실거래가 수집 (2025.10~2026.03)
- 외부 데이터의 구별 트렌드와 대회 데이터의 트렌드가 다름
- 원인: 외부 데이터는 모든 아파트, 대회 데이터는 특정 샘플
- 결론: 외부 트렌드를 직접 보정에 사용 불가

### OOT 검증

- `utils.py`의 `oot_split()`: 마지막 3개월을 holdout하여 시간 기반 검증
- OOT에서는 보정 없음이 최선이었지만, Public에서는 보정이 효과 있음
- OOT와 실제 리더보드의 패턴이 다를 수 있음에 유의

## 작업 흐름

### 새 전략 추가

1. `strategies/` 해당 레이어에 파일 생성
2. 끝에 `record_result()` 호출 → results.csv 자동 기록
3. 로컬 실행으로 OOF RMSE 확인
4. 유망하면 `kaggle_notebooks/`에 통합 스크립트 생성

### Kaggle 제출

1. `kaggle_notebooks/`에서 로컬 실행 → submission.csv 생성 확인
2. Kaggle 노트북 코드 셀을 해당 파일 내용으로 교체
3. 버전명에 전략명 기입 (예: `08 STACK GU TREND`)
4. Save & Run → 제출
5. results.csv에 date, public_rmse 기입

### 현황 조회

```bash
python dashboard.py
```

## 다음 작업 방향

트렌드 보정 실험은 소진됨. 모델 예측력 자체를 개선하는 방향:

1. **메타 피처 스태킹**: Ridge에 OOF 예측값 + 원본 피처(Gu, Exclusive_Area 등) 함께 입력
2. **이상치 처리**: Target 극단값 클리핑 후 재학습
3. **검증 전략 개선**: 시계열 분할 CV 도입

## 환경

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
