# Gemini Advisor Report (2026-06-26)

## Executive Summary

현재 상황은 **"모델링의 한계(Noise Floor)에 근접한 상태에서, 합성 데이터의 특성을 이해하고 안정적인 Final Submission을 준비해야 하는 단계"**입니다. 
전략 28이 Public 1위(2,096.8)를 기록하고 있지만, 이는 OOF(2,196) 대비 상당히 높은 성적(100점 개선)이므로, Public LB의 특성(혹은 데이터의 분포)이 OOF와 다를 수 있음을 인지해야 합니다. 

이미 많은 방향(트렌드, 타겟 변환, 피처 변형 등)을 소진했으므로, 이제는 **'새로운 피처'를 찾는 것보다 '현재 모델의 조합(Ensemble)을 어떻게 구성할 것인가'와 '합성 데이터의 노이즈를 어떻게 견딜 것인가'**에 집중해야 합니다.

---

## Q1. 구조적 돌파구: RMSE를 더 줄일 수 있는 방향

전략 28(8모델 Ridge)이 최적점에 도달했다면, 이제는 '개별 모델의 성능'이 아닌 **'모델 간 오차의 독립성(Error Decorrelation)'**을 극대화하는 방향으로 가야 합니다.

1. **Target-Aware Cross-Blending (Target-Space Diversification)**
   - 현재는 `log`와 `raw` 스케일, 그리고 `Unit Price`를 사용 중입니다. 
   - **아이디어**: 단순히 타겟을 바꾸는 게 아니라, **'예측 대상의 성격'**을 분리합니다.
     - 모델 A: 전체 가격 예측 (Total Price)
     - 모델 B: 면적 대비 가격 예측 (Unit Price)
     - 모델 C: 층수에 따른 프리미엄 예측 (Floor-adjusted Price)
   - 이를 각각 학습시킨 후, Ridge 스태킹 시 입력 피처로 사용하는 것이 아니라, **각 모델의 결과물(Prediction)을 입력으로 하는 스태킹**을 수행하되, 모델들이 서로 다른 '관점'을 갖도록 강제해야 합니다.

2. **Extremely Shallow Meta-Learners (Information Compression)**
   - 2단 스태킹(전략 40)이 실패한 이유는 메타 모델(LGB)이 8개의 입력 사이의 비선형성을 찾으려다 과적합되었기 때문입니다.
   - **돌파구**: 메타 모델을 '예측'하는 용도가 아니라, **'모델별 신뢰도(Weight)를 동적으로 결정'**하는 용도로 사용합니다.
   - 예를 들어, `Gu`(구)나 `Dong`(동) 정보를 메타 모델의 입력으로 넣어, "강남구에서는 모델 A가 더 잘 맞고, 성동구에서는 모델 B가 더 잘 맞는다"는 식의 **Local Weighting**을 수행합니다. (이미 Ridge 스태킹에서 구별 트렌드 보정을 하고 있지만, 이를 모델 가중치 레벨에서 수행하는 것입니다.)

3. **Denoising Autoencoder (DAE) 기반 피처 추출**
   - 데이터가 합성 데이터라면, 특정 생성 알고리즘(Generator)에 의한 패턴이 존재할 것입니다.
   - **아이디어**: 데이터의 노이즈 구조를 학습하기 위해, 입력 피처에 인위적인 노이즈를 섞은 후 원래 값을 복원하도록 학습하는 DAE를 사용합니다. 이 과정에서 추출된 Latent Feature는 단순한 피처보다 데이터의 '구조적 본질'을 더 잘 담을 수 있습니다.

---

## Q2. Private LB 방어 및 Final Submission 구성

Public 1위는 매우 달콤하지만 위험합니다. OOF와 Public의 갭(100점)은 **'Public LB가 특정 패턴(예: 특정 구, 특정 가격대)에 편향되어 있음'**을 강력하게 시사합니다.

1. **위험 분석**
   - **위험 요소**: 만약 Public LB가 `PL2`나 `Unit Price` 모델이 잘 맞추는 특정 패턴에 과하게 최적화되어 있다면, Private에서는 이 모델들의 오차가 폭발할 수 있습니다.
   - **방어 전략**: **'다양성(Diversity) vs 성능(Performance)'**의 트레이드오프를 관리해야 합니다.

2. **Final Submission 구성 (2개 추천)**

   - **[Submission 1] 공격적/최적화형 (The Winner)**
     - **구성**: 현재 최선인 **전략 28 (PL2 + Unit Price 8-model Ridge)**.
     - **목표**: Public LB의 패턴을 그대로 따라가며 1위를 유지하거나 더 높임.
     - **비중**: 100% 전략 28.

   - **[Submission 2] 보수적/안정형 (The Safe Bet)**
     - **구성**: **전략 08 (Original 4-model + Scale Blend) 기반의 앙상블**.
     - **목표**: PL2나 Unit Price 모델이 가질 수 있는 '합성 데이터 특화 편향(Transductive Bias)'을 제거하고, 가장 기본적이고 일반화 능력이 검증된 모델들의 조합을 사용.
     - **방법**: 
       - PL2를 제외한 `Scale Blend` 모델들과 `Original Ridge` 모델을 7:3 또는 6:4로 블렌딩.
       - 혹은, `α=0.9` 수준의 보수적인 트렌드 보정 적용.

**결론**: 1개는 현재의 '최고점'을, 1개는 '가장 검증된 기본기'를 제출하여 리스크를 분산하십시오.

---

## Q3. 남은 12일 활용법

이미 많은 것을 시도했습니다. 이제는 **'확장'보다는 '정교화(Refinement)'**의 시간입니다.

1. **(1~3일차) Submission 2 후보 탐색 (Stability Check)**
   - PL2와 Unit Price를 제외했을 때, 가장 안정적인(OOT와 OOF의 괴리가 적은) 조합을 찾습니다. 
   - "어떤 모델을 뺐을 때 Private에서 뒤집힐 가능성이 가장 낮은가?"를 찾는 과정입니다.

2. **(4~7일차) Error Pattern Deep Dive (Residual Analysis)**
   - 현재 모델이 틀리는 데이터(OOF 상위 오차)를 다시 분석합니다.
   - 단순한 이상치(성동구 사례)인지, 아니면 특정 조건(예: '신축 대형 평수')에서의 구조적 결함인지 구분합니다.
   - 만약 구조적 결함이라면, 그 조건에만 특화된 **'Micro-Model'**을 만들어 블렌딩에 추가합니다.

3. **(8~12일차) Final Ensemble & Submission Management**
   - 하루 5회 제출권을 아껴서, 위에서 찾은 후보들을 정교하게 섞는 실험에 사용합니다.
   - 마지막 2일은 새로운 실험을 멈추고, 확정된 전략들의 최종 제출 파일 생성 및 검증에만 집중합니다.

---

## Q4. 합성 데이터 역이용: 생성 패턴 추적

데이터가 100% 합성이라는 점은, 이 데이터가 **'확률 분포(Distribution)'가 아닌 '함수(Function)'**에 의해 생성되었을 가능성을 의미합니다.

1. **Generator Signature 탐색**
   - 실제 데이터와 합성 데이터의 분포 차이를 분석하여, 생성기가 어떤 'Rule'을 따르는지 역추적합니다.
   - 예를 들어, `Area`와 `Price`의 관계가 단순 선형인지, 아니면 특정 `Floor`나 `Year_Built`에 따라 불연속적인 계단식 함수(Step function)를 갖는지 확인합니다. 
   - 만약 계단식 패턴이 있다면, 이를 피처로 만드는 것이 아니라 **모델의 Loss Function에 반영(Custom Loss)**하거나, **Decision Tree의 분기점**을 찾는 데 활용할 수 있습니다.

2. **Noise Structure Analysis (Error Modeling)**
   - 합성 데이터의 노이즈는 무작위(White Noise)가 아니라, 생성 알고리즘의 한계에서 오는 **'구조적 노이즈(Structured Noise)'**일 확률이 높습니다.
   - OOF 오차의 분포가 특정 피처(예: `Year_Built`)와 상관관계가 있는지 확인하십시오. 
   - 만약 상관관계가 있다면, 그 노이즈 자체를 예측하는 **'Noise Predictor'**를 만들어 최종 예측값에서 빼주는 방식(Residual-based Denoising)을 고려해 볼 수 있습니다.

3. **Adversarial Validation (Train vs Test)**
   - Train과 Test를 구분하는 분류기(Classifier)를 만들어 봅니다.
   - 만약 이 분류기가 매우 높은 성능을 보인다면, Train과 Test의 분포가 심각하게 다르게 생성된 것입니다.
   - 이 분류기가 중요하게 사용하는 피처가 바로 **'합성 데이터의 생성 규칙'**이 담긴 핵심 피처입니다. 이 피처를 중심으로 모델을 재구성하십시오.
