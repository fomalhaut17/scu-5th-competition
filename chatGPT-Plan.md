# ChatGPT Advisor Report (2026-06-28)

기준 문서: `GUIDE.md`, `results.csv`, `docs/advisor_question_0628.md`. 현재 기준선은 전략 56 `BLEND 53:47`, Public RMSE 2,086.6이다.

## 결론

46점 점프를 다시 만들 가능성이 가장 큰 방향은 새 모델 추가가 아니다. 이미 CatBoost, LightGBM, ExtraTrees, LGBM extra_trees, 타겟 변환, MLP, PL, 메타 모델이 대부분 같은 예측으로 수렴했다. 남은 큰 축은 `Train 2024~2025 -> Test 2026Q1` 시간 갭을 더 정교하게 맞추는 것이다.

우선순위는 다음과 같다.

1. **OOF가 아니라 OOT로 시간 보정기를 학습**한다.
2. **현재 GTR을 대체/보완하는 여러 trend family**를 만든다.
3. **최종 예측 파일끼리 저차원 블렌드/스케일을 제출 피드백으로 탐색**한다.
4. 모델 추가는 중단하고, 예측 분포와 구간별 변화량을 관리한다.

## 왜 모델 다양성보다 시간 보정인가

현재 실패 로그가 말하는 것은 명확하다. 좋은 모델은 GBDT와 상관 0.98~0.999로 수렴하고, 상관이 낮은 모델은 단독 OOF가 너무 약하다. 이 상태에서 46점 점프를 만들려면 13번째 모델이 아니라 test 시점의 평균 위치를 맞추는 보정이 필요하다.

특히 Public이 OOF보다 약 100점 좋다. 이는 모델이 일반화가 나쁜 것이 아니라, train OOF에 섞인 합성 노이즈와 2026 test의 분포가 다르게 작동한다는 신호다. 따라서 OOF 1~2점 개선을 쫓기보다, 2026Q1의 평균 레벨과 구/월별 레벨을 맞추는 쪽이 더 기대값이 높다.

## 제안 1: OOT 기반 Temporal Adapter

현재 GTR은 train 전체 월별 평균 상승률을 구별로 계산해 test months ahead에 곱한다. 이 방식은 단순하고 효과가 있었지만, 보정식 자체를 OOT로 학습하지는 않는다.

추천 실험은 다음 구조다.

1. 2024~2025 데이터를 시간 기준으로 나눈다.
   - 학습: 2024-01~2025-09
   - OOT 검증: 2025-10~2025-12
2. 전략 56과 같은 base pipeline을 학습하되, OOT 검증월을 test처럼 예측한다.
3. OOT 잔차를 `month_ahead`, `Gu`, `Exclusive_Area`, `pred_price`, `unit_price_pred` 기준으로 분석한다.
4. 잔차 보정 모델은 매우 작게 둔다.
   - 후보 A: `log(y / pred) = Gu + month_ahead`
   - 후보 B: `log(y / pred) = Gu + month_ahead + area_bin`
   - 후보 C: `y - pred = Gu + month_ahead + pred_bin`
5. 이 보정기를 full train 기준 2026Q1에 적용한다.

핵심은 복잡한 residual model이 아니다. 이미 residual modeling은 실패했다. 여기서의 목적은 개별 샘플 잔차를 맞추는 것이 아니라, 시간 외삽으로 생기는 구/월 단위 레벨 차이만 보정하는 것이다.

제출 후보:

- `56 + OOT log-ratio adapter`
- `56 + OOT additive adapter`
- `56 70% + adapter 30%`

성공 기준:

- OOT 2025Q4에서 전략 56 대비 RMSE가 내려가야 한다.
- 구별 평균 오차의 절대값이 줄어야 한다.
- 보정 배율이 과하지 않아야 한다. 대략 -3%~+5% 범위를 넘으면 과적합 가능성이 크다.

## 제안 2: GTR Family를 다시 넓히기

기존 트렌드 보정은 `구별 단순평균, alpha=1.0`이 최적이었다. 다만 이것은 같은 family 안의 alpha sweep에 가깝다. 아직 다른 trend family는 충분히 분리해서 본 것이 아니다.

시도할 만한 family:

1. **Log trend**
   - 현재: `(1 + pct_growth) ** months`
   - 대안: `exp(mean(diff(log(monthly_mean))) * months)`
   - 합성 DGP가 곱셈 구조라면 log trend가 더 안정적일 수 있다.

2. **Unit-price trend**
   - `Target` 평균 대신 `Target / Exclusive_Area`의 구별 월성장률 사용.
   - 평당가 축이 큰 개선을 만든 전례가 있으므로, 시간 보정도 price보다 unit price에서 더 깨끗할 수 있다.

3. **Robust median trend**
   - 월별 평균 대신 월별 median 또는 trimmed mean 사용.
   - 성동구 OOT처럼 1건 이상치가 MSE를 크게 흔드는 데이터에서는 평균 trend가 불안정할 수 있다.

4. **Recent slope trend**
   - 전체 2024~2025 pct_change 평균 대신 최근 6개월 또는 최근 9개월 log slope 사용.
   - test가 2026Q1이므로 오래된 2024 패턴보다 2025 하반기 기울기가 더 중요할 수 있다.

5. **Global+Gu shrink trend**
   - `trend = w * gu_trend + (1-w) * global_trend`
   - 구별 샘플이 적으므로 Gu trend 100%가 우연히 Public에 맞았더라도 Private에서 흔들릴 수 있다.
   - `w`는 0.5, 0.7, 0.9 정도만 본다.

이 실험은 모델 재학습 없이 최종 예측에 곱하는 방식으로 빠르게 만들 수 있다. 하루 제출 5회 제한이 있으므로 로컬 OOT에서 2개만 남기고 제출한다.

## 제안 3: Public 피드백을 이용한 저차원 운영

3위의 46점 개선은 새로운 알고리즘 하나보다 제출 파일 간 blend/scale을 잘 찾았을 가능성이 높다. 우리도 이미 53(no PL2)와 47(PL2)을 80:20으로 섞어 개선했다. 이 방향을 더 체계화해야 한다.

후보 축:

- `strategy53 no-PL2`
- `strategy47 PL2`
- `strategy56 current 80:20`
- `OOT adapter`
- `GTR family` 1~2개

탐색 방식:

1. 먼저 로컬 OOT에서 후보 예측들의 상관과 구간별 차이를 본다.
2. Public 제출은 한 번에 한 축만 움직인다.
3. 제출 후보는 다음처럼 저차원으로 제한한다.
   - `53:47 = 90:10`
   - `53:47 = 70:30`
   - `56 * global_scale`, scale은 0.995 또는 1.005
   - `56 + OOT adapter 30%`
   - `56 + unit-price GTR`

주의할 점은 Public 과적합이다. 제출 피드백으로 5차원 이상을 맞추면 Private 리스크가 커진다. 하지만 1차원 blend 또는 1차원 scale은 현재처럼 OOF/Public 갭이 큰 상황에서 현실적인 운영 수단이다.

## 제안 4: Test Prediction Distribution Matching

정답 없이도 할 수 있는 점검이 있다. train 2025Q4와 test 2026Q1의 예측 분포를 비교해, 보정 후 분포가 비정상적으로 튀는지 확인한다.

체크 항목:

- 구별 예측 평균 상승률
- 월별 예측 평균 상승률: 2026-01, 2026-02, 2026-03
- area bin별 예측 상승률
- predicted price decile별 상승률
- `Target / Area` 예측 분포

목표는 train의 실제 상승률과 test의 예측 상승률이 일관되는지 보는 것이다. 예를 들어 GTR 후 특정 구의 2026Q1 예측 unit price가 2025Q4보다 10% 이상 튄다면, Public이 좋아도 Private 리스크가 있다.

이 분석은 점수를 직접 올리는 모델은 아니지만, 제출 후보를 줄이는 데 매우 중요하다. 제출 5회 제한에서는 "안 좋은 후보를 제출하지 않는 것"도 점수 개선이다.

## 제안 5: PL은 confidence가 아니라 disagreement regime으로 제한

PL2 threshold와 weighted PL2가 실패한 이유는 confidence 수치 자체가 변별력이 낮기 때문이다. 그래도 PL을 완전히 버리기보다, 모델 합의가 높은 샘플의 특성을 먼저 봐야 한다.

다시 접근한다면 다음 방식만 추천한다.

- confidence 상위 n%가 아니라, `PL2 유/무`, `ET/LGBET`, `log/raw`, `price/unit_price`가 모두 같은 방향으로 움직이는 샘플만 사용한다.
- PL target은 평균이 아니라 median 또는 trimmed mean을 쓴다.
- pseudo sample weight는 작게 고정한다. 예: 원본 1.0, pseudo 0.2.
- 적용 모델은 전체 12모델이 아니라 GBDT 4모델 Stage1에만 제한한다.

다만 우선순위는 낮다. PL 계열은 이미 여러 번 OOF 낙관성과 Public 악화를 보였고, 현재 최선도 no-PL2 80% 쪽이다.

## 하루 5회 제출 운영안

다음 2일은 모델 추가를 멈추고, 보정 후보만 제출하는 편이 좋다.

### Day 1

1. `53:47 = 90:10`
2. `53:47 = 70:30`
3. `56 + log trend GTR`
4. `56 + unit-price GTR`
5. `56 + robust median GTR`

여기서 하나라도 2,060대 이하로 가면, 46점 점프의 원인은 trend family 또는 PL/no-PL blend였다고 보면 된다.

### Day 2

Day 1 최고 후보를 기준으로 한 축만 더 움직인다.

1. 최고 후보의 약한 blend 버전
2. 최고 후보의 강한 blend 버전
3. `OOT log-ratio adapter 30%`
4. `OOT log-ratio adapter 50%`
5. Final 방어용 `56` 또는 `56`에 가까운 보수 blend

## 최종 권고

지금 상태에서 46점 점프를 기대할 수 있는 유일한 남은 큰 축은 시간 외삽 보정이다. 새 모델은 대부분 GBDT와 같은 답으로 수렴했고, 약한 모델은 노이즈만 더한다.

따라서 다음 액션을 추천한다.

1. `strategy56`을 고정 기준선으로 둔다.
2. OOT 2025Q4를 이용해 `log(y/pred)` 기반 temporal adapter를 만든다.
3. GTR을 `log trend`, `unit-price trend`, `robust trend`, `recent slope`, `global+gu shrink`로 분기한다.
4. 제출은 1차원 blend/scale만 움직인다.

요약하면, 모델링 문제로 더 파고들기보다 `2026Q1의 위치를 얼마나 올리거나 내릴 것인가`를 맞추는 문제로 바꿔야 한다. 3위의 점프도 이 축에서 나왔을 가능성이 가장 높다.
