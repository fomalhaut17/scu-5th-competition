# [Gemini] Strategic Jump to Top 3: From Incremental to Structural Breakthrough

## 1. Diagnosis: The "Incremental Wall"
현재 우리 파이프라인은 **"모델 다양성 확보 $ightarrow$ Ridge 스태킹 $ightarrow$ PL2 증강"**이라는 정석적인 경로를 통해 최적점에 도달했습니다. 하지만 최근의 실험(전략 57~60)에서 확인했듯이, 단순히 모델을 추가하거나 하이퍼파라미터를 조정하는 **증분적 개선(Incremental Improvement)**으로는 더 이상 유의미한 점수 향상이 어렵습니다.

3위가 46점을 한 번에 개선했다는 것은 모델의 '종류'를 바꾼 것이 아니라, **데이터의 생성 원리(DGP)를 관통하는 새로운 '구조'를 발견했거나, RMSE의 주범인 '특정 세그먼트'를 완벽하게 제어**했을 가능성이 매우 높습니다.

---

## 2. Strategic Proposals for the "Jump"

### Proposal 1: Hybrid DGP Modeling (Linear Skeleton + Tree Residuals)
**가설**: 합성 데이터는 $	ext{Price} = 	ext{Base} 	imes 	ext{Factor}_1 	imes 	ext{Factor}_2 \dots$ 형태의 곱셈 구조를 가질 가능성이 큽니다. GBDT는 이를 계단식으로 근사하지만, 로그 변환 후의 선형 모델은 이 '골격'을 정확하게 포착합니다.

- **접근법**:
    1.  $\log(	ext{Target})$과 $\log(	ext{Features})$를 이용한 **단순 Ridge 모델(Linear Skeleton)**을 구축합니다.
    2.  이 모델은 개별 성능은 낮을 수 있으나, 합성 데이터의 전체적인 '수식'을 가장 잘 반영합니다.
    3.  **Residual Stacking**: [Linear Skeleton의 예측값] + [GBDT가 예측한 잔차(Residual)] 형태로 앙상블합니다. 
    4.  단순히 모델을 Ridge 스택에 넣는 것이 아니라, **"수식 기반 기본값 + 트리 기반 보정값"**의 구조로 변경하여 GBDT의 한계(계단식 예측)를 극복합니다.

### Proposal 2: Tail-End Specialization (The "RMSE-Killer" Focus)
**가설**: 현재 오차 분석 결과, 120㎡ 이상 대형 평수와 고가 아파트가 전체 RMSE의 대부분을 결정하고 있습니다. 일반적인 Ridge 스태킹은 모든 샘플을 동일하게 취급하지만, 상위권은 이 '꼬리 부분'을 특별 관리했을 것입니다.

- **접근법**:
    - **Weighted Meta-Learning**: 스태킹 Ridge 학습 시, `Area`가 크거나 `Price`가 높은 샘플에 더 높은 가중치(Sample Weight)를 부여합니다. 이를 통해 메타 모델이 "대형 평수를 잘 맞추는 모델"에 더 높은 가중치를 할당하게 만듭니다.
    - **Segmented Stacking**: 
        - Group A: 대형/고가 (Area $\ge 120$ or Price $\ge 7$억)
        - Group B: 일반
        - 두 그룹에 대해 **서로 다른 Ridge 가중치**를 학습시켜 적용합니다.

### Proposal 3: Consensus-based Ultra-Conservative PL (Noise Filtering)
**가설**: PL2의 50% 컷오프는 데이터 양을 늘리지만, 모델 간 의견이 갈리는 '경계선'의 노이즈까지 함께 유입시킵니다. 3위의 점프가 PL 관련이라면, 훨씬 더 정교한 필터링을 사용했을 것입니다.

- **접근법**:
    - **Consensus Filter**: 12개 모델의 예측값이 단순 신뢰도(Confidence) 상위 50%인 것뿐만 아니라, **모델 간의 표준편차(Standard Deviation)가 극도로 낮은(예: 하위 10%) 샘플**만 선택하여 PL을 수행합니다.
    - 즉, "모든 모델이 자신 있게, 그리고 일치하게 예측한" 데이터만 추가하여 PL의 순수도를 극대화합니다.

### Proposal 4: 2026 "Jump" Factor Calibration
**가설**: Train(2025년 말) $ightarrow$ Test(2026년 초)로 넘어갈 때, 단순한 선형 트렌드 이상의 '점프'나 '변곡점'이 존재할 수 있습니다.

- **접근법**:
    - **Relative Trend Correction**: 구별 단순 평균 보정(GTR) 대신, Test 데이터의 피처 분포(예: $	ext{Area}$의 평균 변화)를 이용하여 **보정 계수를 동적으로 조정**합니다.
    - 예를 들어, Test 셋에 대형 평수가 더 많다면, 대형 평수의 트렌드 가중치를 높이는 식의 분포 기반 보정을 시도합니다.

---

## 3. Execution Roadmap (Priority Order)

1.  **Priority 1 (High Impact)**: **Proposal 2 (Weighted Stacking)**. 구현이 간단하며, 현재 오차 구조(대형/고가 집중)를 직접적으로 해결하는 방법입니다.
2.  **Priority 2 (Structural)**: **Proposal 1 (Hybrid Skeleton)**. GBDT의 구조적 한계를 넘어서는 유일한 방법이며, 합성 데이터의 특성에 가장 부합합니다.
3.  **Priority 3 (Stability)**: **Proposal 3 (Consensus PL)**. PL의 효율을 극대화하여 Private LB에서의 안정성을 확보합니다.

**Gemini의 최종 제안**: 지금은 모델의 '개수'를 늘릴 때가 아니라, **"누구를 더 믿을 것인가(Weighting)"**와 **"어떤 구조로 예측할 것인가(Hybrid)"**를 고민해야 할 시점입니다. 특히 **Weighted Ridge Stacking**을 통한 고오차 영역 집중 방어를 강력히 추천합니다.
