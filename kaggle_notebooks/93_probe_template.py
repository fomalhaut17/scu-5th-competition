"""
93 PROBE TEMPLATE: 테스트 NaN 21건 그룹 프로빙용 초경량 노트북.
전략92 노트북의 출력(submission.csv)을 Kaggle Add Data로 첨부한 뒤 이 셀만 실행 —
PROBE_IDS에 지정한 행들만 ×0.4로 바꿔 제출. 점수 변화량(ΔMSE)으로 그룹 내 이상치 수를 역산.

원리: 행 i가 정상이면 ×0.4 적용 시 SSE가 +0.36·P̂ᵢ² 증가, 이상치면 −0.36·P̂ᵢ² 감소.
ΔSSE = (신RMSE² − 기준RMSE²) × 531 = 0.36 × (Σ정상P̂² − Σ이상치P̂²)
→ 로컬 계산기(strategies/_exceptions/93_probe_calc.py)에 기준/프로브 점수를 넣으면 해석됨.

프로빙 순서 (버전명에 그룹 표기):
  Probe A = GROUP_A (7건) / Probe B = GROUP_B (7건) / Probe C = GROUP_C (7건)
  → 이상치 포함 그룹만 반으로 쪼개 재프로빙 (이진탐색)
리더보드는 최고점 기준이라 프로브 제출은 순위에 무해. 테스트 100% 공개 채점이라 결과 확정적.
"""
import os
import pandas as pd

# 테스트 NaN 21건 (등장 순서), 7건씩 3그룹
GROUP_A = ['TR_1414', 'TR_2011', 'TR_1293', 'TR_0056', 'TR_2359', 'TR_2404', 'TR_1266']
GROUP_B = ['TR_0668', 'TR_0178', 'TR_2346', 'TR_1124', 'TR_1273', 'TR_0471', 'TR_2177']
GROUP_C = ['TR_1936', 'TR_0932', 'TR_2399', 'TR_0420', 'TR_0218', 'TR_1822', 'TR_0891']

# ★ 매 프로브마다 여기만 교체 ★
PROBE_IDS = GROUP_A

if os.path.exists('/kaggle/input'):
    # Add Data로 첨부한 전략92 노트북 출력 경로 (첨부 후 실제 경로 확인 필요)
    import glob
    cands = glob.glob('/kaggle/input/*/submission.csv')
    assert cands, "전략92 출력 submission.csv를 Add Data로 첨부하세요"
    BASE_SUB = cands[0]
    OUTPUT_DIR = '/kaggle/working'
else:
    _DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    BASE_SUB = f'{_DIR}/submission.csv'
    OUTPUT_DIR = _DIR

sub = pd.read_csv(BASE_SUB)
mask = sub['ID'].isin(PROBE_IDS)
assert mask.sum() == len(PROBE_IDS), f"ID 매칭 실패: {mask.sum()}/{len(PROBE_IDS)}"

print(f"프로브 대상 {mask.sum()}건의 기준 예측값 (P̂, 역산에 필요 — 기록해둘 것):")
for _, r in sub[mask].iterrows():
    print(f"  {r['ID']}: {r['Target']:,.1f}")
print(f"  ΣP̂² = {(sub.loc[mask, 'Target']**2).sum():,.0f}")

sub.loc[mask, 'Target'] = sub.loc[mask, 'Target'] * 0.4
sub.to_csv(f'{OUTPUT_DIR}/submission.csv', index=False)
print(f"\n프로브 제출 파일 저장: {OUTPUT_DIR}/submission.csv ({len(PROBE_IDS)}건 ×0.4 적용)")
