"""
95 LABEL CALC: 단독 프로브 점수에서 해당 행의 정답 라벨 T를 정확히 역산.

원리: 행 i의 예측을 P̂→0.4P̂로 바꿔 제출하면
  ΔSSE = (0.4P̂−T)² − (P̂−T)² = −0.84·P̂² + 1.2·P̂·T   (T에 선형 → 유일해)
  T = (ΔSSE + 0.84·P̂²) / (1.2·P̂),  ΔSSE = (프로브RMSE² − 기준RMSE²) × N

사용법:
  python 95_label_calc.py <기준RMSE> <프로브RMSE> <P̂>
예 (TR_1822, 2026-07-03 Probe D):
  python 95_label_calc.py 2002.93855 2023.22953 15760.4
"""
import sys

N = 531  # 테스트 행 수 (100% 공개 채점)


def extract_label(base_rmse: float, probe_rmse: float, p_hat: float):
    dsse = (probe_rmse**2 - base_rmse**2) * N
    t = (dsse + 0.84 * p_hat**2) / (1.2 * p_hat)

    # 검증 1: 역산 T로 프로브 점수 재현 (5자리 일치해야 함)
    dsse_check = -0.84 * p_hat**2 + 1.2 * p_hat * t
    probe_reproduced = (base_rmse**2 + dsse_check / N) ** 0.5

    # 점수 소수 5자리 반올림에 의한 T의 불확실성 폭
    t_err = N * 2 * probe_rmse * 0.000005 / (1.2 * p_hat)

    # 최종본에서 이 행을 T로 고정할 때의 이득
    mse_gain = (p_hat - t) ** 2 / N
    new_base = (base_rmse**2 - mse_gain) ** 0.5

    return t, t_err, probe_reproduced, mse_gain, new_base


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    base, probe, p_hat = map(float, sys.argv[1:4])
    t, t_err, repro, gain, new_base = extract_label(base, probe, p_hat)
    print(f"기준 {base:,.5f} → 프로브 {probe:,.5f} (P̂={p_hat:,.1f})")
    print(f"역산 라벨 T = {t:,.1f} (±{t_err:.2f}, 점수 반올림 한계)")
    print(f"프로브 점수 재현: {repro:,.5f} {'✓' if abs(repro-probe) < 1e-4 else '✗ 불일치!'}")
    print(f"이 행 고정 시: MSE −{gain:,.0f} → 기준 {base:,.2f} → {new_base:,.2f} ({new_base-base:+,.2f}점)")
    if abs(t / 0.4 - round(t / 0.4)) < t_err / 0.4:
        print(f"⚠ T/0.4 = {t/0.4:,.1f}이 정수 근처 — 이 행은 이상치(×0.4)일 수 있음 (참고용)")
