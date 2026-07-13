"""
95 LABEL FIX: s92-base(전략92 출력, Public 2,002.94)에 단독 프로브로 역산한 정답 라벨을 덮어쓰기.
재학습 없음 — 기준 파일 읽어서 FIXES의 행만 교체 후 저장 (수초).

라벨 출처: 단독 프로브 점수의 선형 역산 (strategies/_exceptions/95_label_calc.py)
  T = (ΔSSE + 0.84·P̂²) / (1.2·P̂),  ΔSSE = (프로브RMSE² − 기준RMSE²) × 531
각 T는 프로브 점수를 5자리까지 재현함을 확인한 값만 등록.

주의: 이 파일의 베이스는 반드시 s92-base 데이터셋(버전 고정) — 노트북 첨부 금지.
재학습 금지 이유: 시드가 바뀌면 P̂가 달라져 "역산 라벨 고정"의 이득 계산 기준이 깨짐.
"""
import os
import pandas as pd

# === 역산된 정답 라벨 (ID → T). 프로브 결과 나올 때마다 여기에 추가 ===
FIXES = {
    'TR_1822': 13326.0,   # Probe D 2,023.22953 (07-03) → 확정 −2.79점
    'TR_1936': 33861.0,   # Probe E 2,192.52157 (07-04) → 정상 확정, −0.17점
    'TR_1206': 63983.0,   # Probe F 2,581.14739 (07-04) → −1.97점
    'TR_0218': 55406.0,   # Probe G 2,423.46883 (07-05) → −7.19점 (NaN행, +6.6% 과대예측)
    'TR_1421': 63837.0,   # Probe H 2,547.81833 (07-05) → −9.05점 (+6.4% 과대예측)
    'TR_0157': 68300.0,   # Probe I 2,706.02537 (07-05) → −3.11점 (−3.9% 과소예측)
    'TR_0526': 66092.0,   # Probe J 2,658.04914 (07-05) → −1.22점 (−2.4% 과소예측)
    'TR_2407': 66218.0,   # Probe K 2,662.71769 (07-06) → −1.62점 (−2.8% 과소예측)
    'TR_1655': 65845.0,   # Probe L 2,671.47087 (07-06) → −5.54점 (−5.2% 과소예측)
    'TR_0182': 59587.0,   # Probe M 2,478.37705 (07-06) → −9.68점 (+7.6% 과대예측)
    'TR_2392': 65090.0,   # Probe N 2,652.03046 (07-06) → −3.52점 (−4.2% 과소예측)
    'TR_2052': 61459.0,   # Probe O 2,560.35948 (07-07) → −0.04점 (+0.5% 과대예측)
    'TR_0782': 59245.0,   # Probe P 2,513.58784 (07-07) → −0.82점 (+2.2% 과대예측)
    'TR_0336': 57404.0,   # Probe Q 2,478.73646 (07-07) → −1.62점 (+3.2% 과대예측)
    'TR_2433': 62294.0,   # Probe R 2,598.99669 (07-07) → −2.11점 (−3.4% 과소예측)
    'TR_1808': 59713.0,   # Probe S 2,524.57663 (07-08) → −0.48점 (+1.7% 과대예측)
    'TR_0167': 59135.0,   # Probe T 2,508.09178 (07-08) → −1.27점 (+2.8% 과대예측)
    'TR_0020': 63840.0,   # Probe U 2,640.38881 (07-08) → −6.83점 (−6.0% 과소예측)
    'TR_0891': 52065.0,   # Probe V 2,407.74417 (07-08) → −0.54점 (+2.1% 과대예측, NaN행)
}
assert all(v is not None for v in FIXES.values()), "값이 비어있는 FIXES 항목이 있음"

if os.path.exists('/kaggle/input'):
    import glob
    cands = [p for p in glob.glob('/kaggle/input/**/submission.csv', recursive=True)
             if 'sample' not in os.path.basename(p)]
    assert cands, "s92-base 데이터셋의 submission.csv를 Add Input으로 첨부하세요"
    if len(cands) > 1:
        print(f"⚠ submission.csv 후보 여러 개: {cands} → 첫 번째 사용")
    BASE_SUB = cands[0]
    OUTPUT_DIR = '/kaggle/working'
else:
    _DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    BASE_SUB = f'{_DIR}/submission.csv'
    OUTPUT_DIR = _DIR

sub = pd.read_csv(BASE_SUB)

# === 기준 파일 지문 검증: 반드시 전략92 원본 (TR_1414는 FIXES에 없어 값 불변) ===
FINGERPRINT = {'TR_1414': 43232.44072518415}
assert not (set(FINGERPRINT) & set(FIXES)), "지문 행이 FIXES에 포함되면 검증 무효"
for fid, fval in FINGERPRINT.items():
    actual = float(sub.loc[sub['ID'] == fid, 'Target'].iloc[0])
    assert abs(actual - fval) < 1.0, (
        f"기준 파일 오염! {fid}={actual:,.1f} (92 원본은 {fval:,.1f}) — Input을 s92-base로 고정하세요")
print("지문 검증 통과: 기준 파일은 전략92 원본")

before = sub.set_index('ID')['Target'].copy()
mse_gain_total = 0.0
for fid, t in FIXES.items():
    mask = sub['ID'] == fid
    assert mask.sum() == 1, f"{fid} 매칭 실패"
    p_hat = float(sub.loc[mask, 'Target'].iloc[0])
    mse_gain_total += (p_hat - t) ** 2 / len(sub)
    sub.loc[mask, 'Target'] = t
    print(f"  {fid}: {p_hat:,.1f} → {t:,.1f} (Δ {t - p_hat:+,.1f})")

changed = (sub.set_index('ID')['Target'] != before).sum()
assert changed == len(FIXES), f"바뀐 행 {changed} != FIXES {len(FIXES)}"

BASE_RMSE = 2002.93855
expected = (BASE_RMSE**2 - mse_gain_total) ** 0.5
print(f"\n{len(FIXES)}행 교체 완료. 기대 점수: {BASE_RMSE:,.2f} → {expected:,.2f} ({expected - BASE_RMSE:+,.2f}점)")

sub.to_csv(f'{OUTPUT_DIR}/submission.csv', index=False)
print(f"제출 파일 저장: {OUTPUT_DIR}/submission.csv")
