# coPilot-Plan

정리 요약
- 현재 기준점: `08 STACK GU TREND` (OOF 2,234 / Public 2,155).
- 유력 후보: `26 PL2 + Scale Blending` 계열(OOF ~2,215) — OOF 개선이 있으나 낙관적일 위험 존재(pseudo-label leakage / transductive bias).
- Gemini 실험: 원본 스케일(Target) 직접 학습이 로그 스케일 대비 고가 샘플에서 절대 RMSE 개선이 큼. 스케일 혼합(로그+원본)과 단순 평균이 강력.

핵심 리스크
- PL2(의사 라벨)는 OOF 낙관성 가능성 — 반드시 보수적/외부 holdout으로 검증.
- 로그 스케일 학습은 고가/대형 아파트 과소예측을 유발할 수 있음.
- 이미 실패한 방향(트렌드 재조정, 복잡한 메타, 지리 피처 재시도 등)은 반복 금지.

우선순위 (오늘의 제출 권장 순서)
1. submission_l4_26_c.csv (26C, PL2+Scale, Ridge 방식 재현)
2. submission_l4_25_scale.csv (Scale Blending, PL2 없음)
3. submission_l4_24_pl2.csv (PL2 only, 평균 상승 성향 확인)
4. 방어용: 08 재제출 또는 08:26C 블렌드(70:30, 50:50)

검증 강화 (필수)
- 보수적 OOF 프로토콜: outer 5-fold 방식(챗GPT 제안)으로 PL2 OOF 재계산.
- OOT(holdout_months) 다양화: 2/3/4개월로 반복 비교.
- 모든 OOF는 "원본 train 영역만"으로 계산해 누수 방지.
- PL2 샘플 선정 기준 로그: 모델간 합의도 + fold 변동성 상위 50% 고정(실험 가능성 확인).

구현/코드 작업 (단계별)
- kaggle_notebooks/27_pl2_scale_ridge.py: `26_pl2_scale_blend.py` 복사 → PL 필터 50% 고정, 최종 방법 C(Ridge) 고정, submission.csv 출력 및 상세 로그.
- strategies/L4_blending/02_scale_blending.py: 4개 모델(M1~M4: CatBoost/LightGBM log/orig) OOF+Test 생성, A/B/C(단순/가중/Ridge) 실험 스크립트.
- 별도 스크립트: tools/compute_conservative_oof.py (outer-fold PL2-safe OOF 계산).

실험 설계
- Stage A: scale-blend 단독(PL2 없음) — 25 제출 전 재현성 확인.
- Stage B: PL2(상위50%) 적용 후 outer-fold conservative OOF 확인.
- Stage C: PL2+Scale-Blending 재학습 → Ridge stacking(방법 C) → GTR 보정 적용 → 제출 후보 선정.

운영 체크리스트
- 제출 후 results.csv에 public_rmse 반드시 기록.
- 제출용 노트북은 로컬에서 생성한 submission.csv 내용을 그대로 Kaggle 노트북에 복사. 버전명 규칙 준수.
- 로그: Stage별 OOF, PL 선택 건수, 방법별 OOF, 08 대비 평균 차이/표준편차 기록.

금지/제한
- 이미 실패로 기록된 실험 반복 금지(메타 복잡화, 동별 외부 트렌드 재시도, XGB/MLP 추가 등).
- PL2 적용 시 leakage 확인 불가 시 해당 결과는 제출금지.

간단 TODO
- [ ] `kaggle_notebooks/27_pl2_scale_ridge.py` 생성 및 로컬 재현
- [ ] `strategies/L4_blending/02_scale_blending.py` 구현
- [ ] `tools/compute_conservative_oof.py` 작성 및 실행
- [ ] 오늘 5회 제출 스케줄(26C,25,24,08:26C blends) 준비

참고 파일
- GUIDE.md, chatGPT-Plan.md, gemini-Plan.md(요약 반영)

작성자: coPilot (요약·우선순위·구현 지침)

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
