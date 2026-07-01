# 조언자 질문 & 답변 기록

외부 AI 조언자(Gemini, ChatGPT, Copilot, opencode 등)에게 리뷰를 요청할 때의 질문과 결과를 기록합니다.
리포 전체(특히 GUIDE.md)를 공유한 후 아래 질문을 던집니다.

## 답변 작성 방법

조언자에게 이 파일을 보여주고 아래 규칙대로 작성을 요청하세요.

1. **이 파일의 해당 날짜 → 본인 이름 답변란**에 요약을 작성합니다.
   - 1~3줄 요약 (핵심 제안만)
   - 상세 답변이 있으면 `조언자이름-Plan.md` 파일 링크 추가
2. **상세 답변**은 리포 루트에 `조언자이름-Plan.md` 파일로 작성합니다.
   - 예: `gemini-Plan.md`, `chatGPT-Plan.md`, `coPilot-Plan.md`, `opencode-Plan.md`
   - 기존 파일이 있으면 **내용을 교체** (날짜별 최신 답변으로 유지)
3. **답변 참고 자료**: `GUIDE.md` (전체 현황), `results.csv` (실험 이력), 이 파일의 이전 답변 기록

답변 예시:
```
#### Gemini
- [상세 답변은 gemini-Plan.md를 참조하십시오.](../gemini-Plan.md)
- **요약**: (1~3줄 핵심 제안)
```

---

## 2026-06-27 (대회 7일차, Public 4위 2,095)

### 현재 상황 요약
- **전략 45** 8모델(CB/LGB×log/raw×기존/평당가) + ExtraTrees 2모델(log/raw) = 10모델 Ridge+GTR → Public RMSE **2,094.9** (4위)
- 전략 28(Public 2,097)에서 ExtraTrees 추가로 -2점 개선
- ExtraTrees가 유일하게 앙상블에 기여한 비트리 모델 (RF/XGB/KNN/SVR 모두 실패)
- HP 변형(CB/LGB 다른 설정), 메타 가중치, 신뢰도 계층화, confidence 피처, 블렌딩 모두 실패
- Adversarial Validation: Train/Test 분포 동일 (AUC=0.505) → Private 안정적

### 질문

**Q1. ExtraTrees 원리의 확장**
> ExtraTrees가 유일하게 앙상블에 기여했습니다 (단독 OOF 3,561이지만 상관 0.963으로 다른 오차 패턴). 같은 원리 — "기존 GB 트리와 다른 오차 패턴을 만드는 알고리즘" — 로 다양성을 줄 수 있는 다른 모델이 있을까요?
> 시도 완료: RF(상관 0.968, +6 악화), XGB(상관 0.996, +2 악화), KNN(+0), SVR(+0), Ridge-poly(+0)

**Q2. 1~3위와의 격차를 줄이는 전략**
> 현재 4위인데, 1~3위는 어떤 접근을 쓰고 있을지 추측할 수 있을까요? 우리가 놓치고 있는 방향이 있다면?

**Q3. 전략 45 기반 Final Submission**
> 전략 45가 새로운 최선이 되었습니다. Final 1=전략 45, Final 2는 어떻게 구성해야 할까요?

### 답변

#### Gemini
- [상세 답변은 gemini-Plan.md를 참조하십시오.](../gemini-Plan.md)
- **요약**: 전략 45를 통해 Public 4위 달성. 핵심 돌파구는 ExtraTrees를 통한 알고리즘 다양성 확보임. 향후 LGBM-ET, KernelRidge 등 이질적 모델 추가와 합성 데이터 생성 규칙(DGP) 역공학, 고오차 영역(대형/고가) 집중 방어를 통해 우승을 노림. Final Submission은 '전략 45 기반 공격형'과 'PL2 배제 및 멀티시드 기반 안정형'의 조합을 제안함.

#### ChatGPT
- [상세 답변은 chatGPT-Plan.md를 참조하십시오.](../chatGPT-Plan.md)
- **요약**: 전략 45는 Final 1로 고정하되, 개선폭이 작으므로 추가 모델 확장은 LGBM `extra_trees=True`/Spline-Ridge 같은 소수 후보만 검증. Final 2는 전략 45 변형보다 `45:28=70:30`을 우선 추천하며, ET가 Public subset에만 맞았을 위험을 얕게 줄이는 방어 제출로 쓰는 것이 합리적.

#### Copilot
- [상세 답변은 coPilot-Plan.md를 참조하십시오.](../coPilot-Plan.md)
- **요약**: ExtraTrees의 성공은 "GB와 다른 오차 패턴"에서 비롯. LGBM `extra_trees=True` 옵션과 KernelRidge 추가로 2~5점 추가 개선 가능. Q2 대응으로 구간별 오차 분석(고가/대형 영역)과 Residual 보정을 병렬 진행. Final은 전략 45(공격) + 45:28=70:30 블렌딩(방어) 조합 추천. 남은 12일은 주 2회 제출 페이스로 검증 우선.

#### opencode
- [상세 답변은 opencode-Plan.md를 참조하십시오.](../opencode-Plan.md)
- **요약**: 전략 45(Public 2,094.9, 4위)는 전략 28 대비 -2점 개선에 그쳤고, 1위와 격차는 여전함. 전략 45의 ExtraTrees는 검증된 유일한 다양성 축이지만, 추가 ET 계열 확장은 수확 체감 구간. 새로운 돌파구는 **구간별 세분화 앙상블(Gu/가격대별 최적 Ridge 가중치)**과 **고오차 영역(대형/고가) 사후보정**이 가장 현실적. 1~3위와의 격차는 PL 임계값 최적화, ET 파라미터 튜닝, 수동 블렌딩에서 나왔을 가능성. Final 1=전략 45(공격), Final 2=전략 45(80%)+전략 28(20%) 블렌드 추천. 남은 리소스는 자동 실험보다 검증+제출에 집중할 것.

---

## 2026-06-26 (대회 6일차, Public 1위 2,097)

### 현재 상황 요약
- **전략 28** PL2+평당가 8모델 Ridge+GTR → Public RMSE 2,096.8 (1위)
- noise floor ~2,300인데 이미 그 아래 → 정상적 모델링의 한계에 근접
- OOT 검증으로 전략 순위 신뢰 확인, α sweep/2단 스태킹 모두 효과 없음
- 소진된 방향이 많음 (GUIDE.md "소진된 방향" 참고)

### 질문

**Q1. 구조적 돌파구**
> 전략 28(8모델 Ridge)이 현 구조 최적점입니다. OOT/α sweep/2단 스태킹 모두 효과 없었습니다. 완전히 다른 접근으로 RMSE를 더 줄일 수 있는 방향이 있을까요?
> (참고: 소진된 방향 — 트렌드 변형, 타겟 변환, 피처 변형, 반복PL, Quantile, 구별분리, 멀티시드, NN, Residual, 외부데이터 등)

**Q2. Private LB 방어**
> Public 1위인데 Private에서 밀릴 위험은 어느 정도일까요? Final submission 2개를 어떻게 구성해야 할까요? 보수적 제출(α=0.9)과 공격적 제출(α=1.0) 중 어떤 조합이 좋을까요?

**Q3. 남은 12일 활용법**
> 마감 07-08, 하루 5회 제출 제한. 소진된 방향이 많은 상황에서 남은 시간을 어떻게 활용해야 할까요?

**Q4. 합성 데이터 역이용**
> 데이터가 100% 합성(실거래 매칭 0%)이라는 걸 역이용할 방법이 있을까요? 생성 패턴 역추적, noise 구조 분석 등.

### 답변

#### Gemini
- [상세 답변은 gemini-Plan.md를 참조하십시오.](#gemini-advisor-report-2026-06-26)
- **요약**: 현재 모델링 한계점에 도달함. 새로운 피처보다는 모델 간 오차 독립성 극대화(Target-Aware Cross-Blending)와 합성 데이터의 구조적 노이즈 분석에 집중할 것을 권장함. Final Submission은 '전략 28(공격형)'과 '전략 08 기반 안정형'의 조합을 제안함.

#### ChatGPT
- [상세 답변은 chatGPT-Plan.md를 참조하십시오.](../chatGPT-Plan.md)
- **요약**: 전략 28은 공격 제출로 고정하는 것이 맞음. Private 리스크는 중간 이상이지만 OOF/OOT/Public 순위가 모두 28 우세라 단순 Public 과적합으로 보기는 어려움. Final 1은 전략 28(α=1.0), Final 2는 α=0.9 단독보다 `28:26 = 70:30` 또는 `80:20` 같은 구조적 방어 블렌드를 권장함. 남은 기간은 신규 모델 탐색보다 adversarial validation, OOF 잔차, 구별/고가/대형 구간 비교로 Final 2 선택 근거를 쌓는 데 쓰는 것이 좋음.

#### Copilot
- [상세 답변은 coPilot-Plan.md를 참조하십시오.](../coPilot-Plan.md)
- **요약**: 현재 단순 ensemble 최적화는 수확 체감. 남은 돌파구는 **구조적 이질성**(구별 가중치) + **사후보정**(고가 구간 재정정) + **합성 데이터 역이용** 3가지. Private LB 방어는 **전략 28(공격) + 안정형 블렌드(28:26=70:30)** 조합으로, 기존 제출이 모두 28 우위이므로 방어선을 너무 높이지 말 것. 남은 12일은 **주간 2회 제출**(검증 시간 우선)로 전략별 OOF/OOT 재현성을 확인한 뒤, Private 리스크 최소 지점에서 최종 2개 제출할 것을 권장.

#### opencode
- [상세 답변은 opencode-Plan.md를 참조하십시오.](../opencode-Plan.md)
- **요약**: 구조적 돌파구는 **타겟 분해 확장**(Area 외 다른 변수로 Target 분해 → 새로운 다양성 축)과 **신뢰도 계층화 예측**(PL2 상/하위 50% 각각 다른 가중치) 2개 축이 유효. Private 방어는 Final 1=28(α=1.0), Final 2=28:32(70:30) 추천 (32의 OOF 2,187이 28의 OOF 2,196보다 낮았고, seed 다양성이 Private에서 방어력 제공). 남은 12일은 검증 집중, 주 2회 제출. 합성 데이터는 **confidence 자체를 feature로 활용**하는 접근 제안.

#### 기타
- (대기 중)

---

## 2026-06-25 (대회 5일차)

### 질문
- GUIDE.md의 "외부 AI 리뷰" 섹션 참고
- Gemini: 원본 스케일 학습 제안 → **채택, 효과 확인 (Scale Blending)**
- ChatGPT: OOF 낙관성 경고, 26C 제출 추천 → **OOF 낙관성은 OOT에서 확인됨**
- Copilot: 요약만, 독자적 기여 없음
- opencode: 평당가(Unit Price) 아이디어 → **채택, 전략 28 핵심 기여**

### 교훈
- Gemini: 코드 직접 분석 + 실험까지 → 가장 유용
- ChatGPT: 리스크 분석 강점
- opencode: 새 아이디어 제공 강점
- Copilot: 요약만
- **서로 다른 AI의 아이디어를 조합했을 때 시너지 발생** (원본스케일 + 평당가 + PL2)

---

## 2026-06-28 (대회 7일차, Public 6위 2,086.6)

### 질문
3위가 2,073→2,027로 46점을 단번에 개선 → 우리도 비슷한 점프를 하려면?

### 답변 요약
- Gemini: **Skeleton+GBDT 잔차 하이브리드** + 고오차 세그먼트 가중치 → **채택, 핵심 기여**
- ChatGPT: 시간 외삽 보정(temporal adapter), blend 탐색 → 부분 채택
- opencode: 시간 외삽 분석, OOF-Public 갭 원인 분석 → 참고

### 결과
Gemini의 Skeleton+GBDT 잔차 아이디어 채택 → 전략 63(One-Hot Skeleton), 69(Per-Gu), 72, 75로 이어짐 (Public 2,087→2,028 개선, 핵심 돌파구)

### 상세
- 질문 파일: `docs/advisor_question_0628.md`
- Gemini 답변: `gemini-Plan.md`

---

## 2026-07-01 (대회 10일차, Public 5위 2,028.4)

### 현황
3-way 블렌딩(56+63+69, 50:30:20)으로 2,028.4 달성. 2위가 2,000.87로 격차가 큼.

### 질문
2위를 따라잡고 Under 2000을 달성하려면 어떤 구조적 돌파구가 필요한가?

### 답변 요약
- Gemini: Log-Log Skeleton(DGP 구조화) + 고오차 세그먼트별 가중치
- opencode: PySR Symbolic Regression(DGP 역공학), Skeleton 다양화 11종
- Copilot: Time Robustness Scoring(월별 안정성), Residual Nonlinear Stacking

### 결과
- **PySR**: 정수 인코딩 문제로 실패 (RMSE 4,734 — 현재 모델보다 나쁨)
- **Nonlinear Meta**: KernelRidge/Tree 전부 선형 Ridge보다 나쁨 → 폐기
- **Time Robustness**: 월별 RMSE 분석, 재조정 가중치 OOT 동일 → 효과 없음
- **Log-Log Skeleton**: OOF +164점 개선, OOT -360점 악화 → 폐기
- **pred_47 비율 변경**: BLEND_W53=0.8→0.0으로 Public 0.36점 소폭 개선

### 상세
- 질문 파일: `docs/advisor_question_0701.md`
- Gemini 답변: `gemini-Plan.md`, opencode: `opencode-Plan.md`, Copilot: `coPilot-Plan.md`
