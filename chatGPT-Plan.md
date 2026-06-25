# 제5회 인공지능 모델링 경진대회 - ChatGPT 전략 제안

이 문서는 `GUIDE.md`, `gemini-Plan.md`, `results.csv`, `kaggle_notebooks/`, `strategies/`를 검토한 뒤 Claude가 바로 실행할 수 있도록 정리한 다음 전략이다.

## 1. 현재 판단

현재 확실한 기준점은 `08 STACK GU TREND`다.

- 구조: FE -> CatBoost(log) + LightGBM(log) -> Ridge stacking -> 구별 트렌드 보정
- OOF: 2,234
- Public: 2,155
- 제출 이력상 트렌드 보정, 지리 피처, pseudo label 단독, 외부 실거래 보정, 복잡한 메타 모델은 대부분 Public에서 악화됐다.

Gemini 제안의 핵심인 "원본 타겟 스케일 학습"과 "Scale Blending"은 이미 구현되어 있다.

- `strategies/_exceptions/25_scale_blending.py`
- `strategies/_exceptions/26_pl2_scale_blend.py`
- `kaggle_notebooks/26_pl2_scale_blend.py`
- `submission_l4_25_scale.csv`
- `submission_l4_26_a.csv`
- `submission_l4_26_b.csv`
- `submission_l4_26_c.csv`
- `submission_l4_26_pl2_scale.csv`

결과 기록상 신규 최고 OOF는 `26 PL2+SCALE`이다.

| 전략 | 내용 | OOF | Public |
|---|---|---:|---:|
| 08 | 기존 최선 STK+GTR | 2,234 | 2,155 |
| 24 | PL2 only | 2,226 | 미제출 |
| 25 | Scale Blending | 2,226 | 미제출 |
| 26 | PL2 + Scale Blending | 2,215 | 미제출 |

## 2. 중요한 리스크

`26`의 OOF 2,215는 그대로 믿으면 안 된다. pseudo label을 붙인 뒤 augmented train 전체에 KFold를 섞고, 원본 train 영역만 잘라 OOF를 계산한다. 이 방식은 원본 train 검증 fold를 예측할 때 같은 fold의 pseudo-labeled test 샘플들이 학습 fold에 들어갈 수 있다. 직접적인 Target 누수는 아니지만, test 분포와 자기 예측값을 이용한 transductive regularization이라 OOF가 낙관적으로 보일 수 있다.

예측 분포도 08과 매우 비슷하다.

| 제출 파일 | 평균 | 표준편차 | 08 대비 corr | 08 대비 RMSE 차이 | 평균 차이 |
|---|---:|---:|---:|---:|---:|
| 08 | 40,416 | 9,884 | 1.00000 | 0 | 0 |
| 24 | 40,523 | 9,949 | 0.99981 | 231 | +106 |
| 25 | 40,278 | 9,681 | 0.99972 | 339 | -138 |
| 26A | 40,298 | 9,687 | 0.99965 | 347 | -119 |
| 26B/final | 40,305 | 9,669 | 0.99958 | 374 | -111 |
| 26C | 40,386 | 9,787 | 0.99954 | 316 | -30 |

따라서 26은 OOF 개선 폭은 크지만, Public에서 08을 크게 이길 가능성과 약간 악화될 가능성이 같이 있다. 하루 제출 5회 제한이 있으므로 공격/방어 제출을 나눠야 한다.

## 3. 다음 제출 우선순위

### 1순위: `26C` 제출

제출 파일: `submission_l4_26_c.csv`

이유:
- `26` 계열 중 08 대비 평균 차이가 가장 작다. (-30)
- 08 대비 RMSE 차이도 `26final/B`보다 작다. (316 vs 374)
- pseudo label + scale 효과는 반영하면서, 08의 Public 최적점에서 덜 벗어난다.
- Public shake-up 리스크가 가장 낮은 26 변형이다.

Kaggle 노트북으로 제출할 때는 `kaggle_notebooks/26_pl2_scale_blend.py`가 현재 final을 방법 B로 저장한다는 점에 주의한다. `submission_l4_26_c.csv`를 재현하려면 방법 C(Ridge stacking)를 최종 선택하도록 고정하거나, 로컬 생성 파일과 동일한 코드를 Kaggle 셀에 반영해야 한다.

### 2순위: `25 Scale Blending` 제출

제출 파일: `submission_l4_25_scale.csv`

이유:
- pseudo label 없이 raw/log target scale만 결합하므로 26보다 구조가 단순하다.
- OOF 2,226으로 08보다 8 낮고, transductive 리스크가 없다.
- 단점은 08 대비 평균이 -138, 고가 구간에서 낮아지는 경향이 있어 고가 과소예측 문제가 실제 Public에서 완화되는지 확인이 필요하다.

### 3순위: `24 PL2 only` 제출

제출 파일: `submission_l4_24_pl2.csv`

이유:
- 25와 같은 OOF 2,226이지만 평균은 08보다 +106으로 올라간다.
- 기존 19 pseudo label은 Public이 2,173으로 악화됐지만, 24는 신뢰도 필터가 더 정교하다.
- 25가 평균을 낮추는 방향, 24는 평균을 높이는 방향이라 리더보드 반응 비교 가치가 있다.

### 4순위: `26B/final` 제출

제출 파일: `submission_l4_26_pl2_scale.csv` 또는 `submission_l4_26_b.csv`

이유:
- 기록상 `26`의 대표 산출물이며 OOF 2,215다.
- 다만 08 대비 평균 -111, 고가 구에서 낮추는 폭이 크다. 특히 Gangnam -353, Seocho -159, Yongsan -234 수준이라 Public에서 고가 과소예측이 문제였다면 불리할 수 있다.

### 5순위: 방어용 08 재제출 또는 26C와 08의 미세 블렌드

08이 이미 Public 2,155라 재제출 자체는 순위 개선 목적은 약하다. 남은 제출권이 있다면 `08 70% + 26C 30%` 또는 `08 50% + 26C 50%`를 추천한다.

이유:
- 18 계열에서 08+16 블렌드는 08 비중이 높을수록 안정적이었다.
- 26C는 08과 다르지만 과격하지 않다.
- 단일 26C가 흔들릴 경우 중간 제출이 방어선이 된다.

## 4. Claude 작업 지시

### A. 제출 전 스크립트 정리

`kaggle_notebooks/26_pl2_scale_blend.py`를 복사해서 다음 파일을 만들 것을 권장한다.

- `kaggle_notebooks/27_pl2_scale_ridge.py`

수정 내용:
- PL 상위 50% 고정
- 최종 방법을 C(Ridge stacking)로 고정
- 출력 파일은 Kaggle 표준 `submission.csv`
- 로그에 다음 값 출력
  - Stage 1 4모델 평균 OOF
  - PL 선택 건수
  - Stage 2 A/B/C OOF
  - 최종 선택 방법
  - 08 대비 예측 평균 차이, 표준편차 차이

### B. 리더보드 결과 기록

제출 후 `results.csv`에 반드시 public_rmse를 기록한다.

추천 기록명:

- `L4,27,PL2+SCALE+C,PL2 + Scale Blending + Ridge final + GTR`
- `L4,25,SCALE+STK`
- `L4,24,PL2+STK`

### C. OOF 신뢰도 보강

시간이 있으면 새 전략 구현보다 검증 보강을 먼저 한다.

필수 검증:
- `08`, `24`, `25`, `26A/B/C`를 같은 OOT split에서 비교
- holdout_months를 2, 3, 4로 바꿔 반복
- 보정 없음/GTR 적용 후를 분리해서 비교

목표:
- 26 계열이 random OOF에서만 좋은지, 시간 holdout에서도 08보다 덜 나쁜지 확인한다.
- OOT에서 계속 08보다 나쁘면 Public 제출 우선순위를 낮춘다.

### D. 26 OOF 계산 방식 개선

현재 26의 OOF는 낙관 가능성이 있다. 더 보수적인 측정 방식을 별도 스크립트로 만든다.

권장 방식:
1. 원본 train을 outer 5-fold로 나눈다.
2. 각 outer fold에서 train 부분만 사용해 Stage 1 pseudo label을 만든다.
3. 그 pseudo label을 outer train에만 붙여 Stage 2 모델을 학습한다.
4. outer validation 원본 train을 예측한다.
5. 5개 outer validation 예측을 합쳐 OOF를 계산한다.

이 OOF가 08보다 유지되면 26은 실제 개선 가능성이 높다. 여기서 무너지면 Public 개선은 운에 가깝다.

## 5. 새 모델보다 우선할 것

현재까지 실패한 방향은 반복하지 않는다.

- 동별/외부 트렌드 보정 재시도 금지
- 지리 피처 추가 재시도 금지
- XGB/MLP 추가 금지
- 복잡한 메타 모델 금지
- residual model 재시도 금지
- Huber/sample weight 대형 실험은 낮은 우선순위

남은 개선 여지는 모델 복잡도가 아니라 제출 선택과 검증 신뢰도에 있다.

## 6. 최종 추천

오늘 제출권이 5개라면 다음 순서로 쓴다.

1. `26C` 또는 이를 재현하는 `27_pl2_scale_ridge.py`
2. `25 Scale Blending`
3. `24 PL2 only`
4. `08 50% + 26C 50%`
5. `08 70% + 26C 30%`

리더보드에서 26C가 08보다 좋아지면, 다음 날은 26C 주변의 blend 비율만 좁게 탐색한다. 예를 들어 `08:26C = 30:70`, `20:80`, `10:90`. 26C가 나빠지면 pseudo label 계열은 중단하고 25와 08 사이의 scale-only 방어 블렌드만 확인한다.
