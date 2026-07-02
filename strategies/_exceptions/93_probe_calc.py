"""
93 PROBE CALC: 프로빙 점수 해석 계산기 (로컬 전용, 제출 없음).

사용법:
  python strategies/_exceptions/93_probe_calc.py <기준RMSE> <프로브RMSE> <P̂1> <P̂2> ...
  예) python .../93_probe_calc.py 2020.5 2250.3 38000 41000 35000 39000 42000 36000 40000

원리 (전략92 기준 제출 대비, 그룹 S에 ×0.4 적용한 프로브):
  ΔSSE = (프로브RMSE² − 기준RMSE²) × 531
  행 i가 정상이면 기여 ≈ +0.36·P̂ᵢ² − εᵢ², 이상치면 ≈ −(0.6P̂ᵢ)² 수준의 감소
  → Σ_{이상치∈S} P̂ᵢ² = (0.36·ΣP̂²_S − ΔSSE) / 0.72
  → 그 합에 맞는 부분집합을 탐색해 이상치 후보 특정 (승법노이즈 ~5.1% 때문에 오차 허용)
"""
import sys
from itertools import combinations

N_TEST = 531
NOISE_REL = 0.051  # 승법 노이즈 표준편차 (DGP 분석 확정치)

def main():
    base_rmse = float(sys.argv[1])
    probe_rmse = float(sys.argv[2])
    preds = [float(x) for x in sys.argv[3:]]
    assert preds, "프로브 그룹의 P̂ 값들을 입력하세요"

    d_sse = (probe_rmse**2 - base_rmse**2) * N_TEST
    sum_p2 = sum(p*p for p in preds)
    # 이상치들의 P̂² 합 추정
    out_p2 = (0.36 * sum_p2 - d_sse) / 0.72
    print(f"ΔSSE = {d_sse:,.0f}")
    print(f"그룹 ΣP̂² = {sum_p2:,.0f}")
    print(f"추정 Σ(이상치 P̂²) = {out_p2:,.0f}")

    # 행당 승법노이즈로 인한 ΔSSE 불확실성: 정상행 (0.6+ε)² 항의 ε 교차항 ≈ 2×0.6×σ_rel×P̂²
    per_row_noise = 2 * 0.6 * NOISE_REL * (sum_p2 / len(preds))
    tol = 1.5 * per_row_noise * (len(preds) ** 0.5)
    print(f"허용 오차(±): {tol:,.0f}\n")

    if out_p2 < tol * 0.5:
        print("→ 이 그룹에 이상치 없음 (0건)")
        return

    print("부분집합 탐색 (P̂² 합이 추정치와 근접한 조합):")
    found = []
    for k in range(1, min(4, len(preds)) + 1):
        for combo in combinations(range(len(preds)), k):
            s = sum(preds[i]**2 for i in combo)
            if abs(s - out_p2) < tol:
                found.append((abs(s - out_p2), k, combo, s))
    found.sort()
    if not found:
        print("  일치 조합 없음 — 허용오차를 늘리거나 그룹을 쪼개서 재프로빙")
    for diff, k, combo, s in found[:8]:
        idxs = ', '.join(f"#{i+1}(P̂={preds[i]:,.0f})" for i in combo)
        print(f"  {k}건 조합 [{idxs}] ΣP̂²={s:,.0f} (오차 {diff:,.0f})")
    if len(found) > 1:
        print("\n  ⚠ 후보가 여러 개면 그룹을 쪼개 재프로빙으로 확정할 것")

if __name__ == '__main__':
    main()
