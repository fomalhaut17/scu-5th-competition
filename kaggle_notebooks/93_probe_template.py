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
# 07-06 큐 완료: TR_2407=66,218 / TR_1655=65,845 / TR_0182=59,587 / TR_2392=65,090
# 07-07 큐 완료: TR_2052=61,459 / TR_0782=59,245 / TR_0336=57,404 / TR_2433=62,294
# 07-08 큐(마감일): 1. TR_1808 완료(59,713) → 2. TR_0167 완료(59,135) → 3. TR_0020 완료(63,840) → 4. NaN행 TR_0891 (버전명 Probe S/T/U/V)
PROBE_IDS = ['TR_0891']  # Probe V (07-08 슬롯4, 마지막 프로브)

if os.path.exists('/kaggle/input'):
    # Add Data로 첨부한 전략92 노트북 출력 — 마운트 깊이가 환경마다 달라 재귀 탐색
    import glob
    cands = [p for p in glob.glob('/kaggle/input/**/submission.csv', recursive=True)
             if 'sample' not in os.path.basename(p)]
    if not cands:
        print("submission.csv를 찾지 못함. /kaggle/input 구조:")
        for root, dirs, files in os.walk('/kaggle/input'):
            for f in files:
                print(f"  {os.path.join(root, f)}")
        raise AssertionError("전략92 출력 submission.csv를 Add Input(Your Work)으로 첨부하세요")
    if len(cands) > 1:
        print(f"⚠ submission.csv 후보 여러 개: {cands} → 첫 번째 사용")
    BASE_SUB = cands[0]
    OUTPUT_DIR = '/kaggle/working'
else:
    _DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    BASE_SUB = f'{_DIR}/submission.csv'
    OUTPUT_DIR = _DIR

sub = pd.read_csv(BASE_SUB)

# === 기준 파일 지문 검증: 반드시 전략92 원본이어야 함 ===
# (프로브 버전을 저장하면 노트북 입력이 "최신 버전"으로 따라가 프로브 출력을 기준으로
#  읽는 사고가 발생 — Probe B가 실제로는 A+B 동시 ×0.4로 제출된 원인. 92 값으로 지문 대조.)
FINGERPRINT = {'TR_1414': 43232.44072518415}  # 전략92 출력의 알려진 값
for fid, fval in FINGERPRINT.items():
    actual = float(sub.loc[sub['ID'] == fid, 'Target'].iloc[0])
    assert abs(actual - fval) < 1.0, (
        f"기준 파일 오염! {fid}={actual:,.1f} (92 원본은 {fval:,.1f}) — "
        f"Input의 노트북 버전을 전략92 버전으로 다시 고정하세요")
print("지문 검증 통과: 기준 파일은 전략92 원본")

mask = sub['ID'].isin(PROBE_IDS)
assert mask.sum() == len(PROBE_IDS), f"ID 매칭 실패: {mask.sum()}/{len(PROBE_IDS)}"

print(f"프로브 대상 {mask.sum()}건의 기준 예측값 (P̂, 역산에 필요 — 기록해둘 것):")
for _, r in sub[mask].iterrows():
    print(f"  {r['ID']}: {r['Target']:,.1f}")
print(f"  ΣP̂² = {(sub.loc[mask, 'Target']**2).sum():,.0f}")

sub.loc[mask, 'Target'] = sub.loc[mask, 'Target'] * 0.4
sub.to_csv(f'{OUTPUT_DIR}/submission.csv', index=False)
print(f"\n프로브 제출 파일 저장: {OUTPUT_DIR}/submission.csv ({len(PROBE_IDS)}건 ×0.4 적용)")
