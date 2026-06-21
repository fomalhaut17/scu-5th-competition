"""
[L4-01] 최종 블렌딩
──────────────────────────
레이어  : L4 (블렌딩)
축약명  : BL
주요 전략: L3 튜닝 모델들의 예측을 가중 평균, 최적 비율 자동 탐색
결과    : OOF RMSE 2,242 ★ 최종 최선 (CB 50% + LGB 50%, LR 0%)
제출파일: submission_l4_01_bl.csv
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import glob
from utils import load_data, base_preprocess, add_feature_engineering, save_submission

train, test, sample_sub = load_data()
y_true = train['Target'].values

# L3 OOF/TEST 예측값 로드
oof_dir = os.path.join(os.path.dirname(__file__), '..', 'L3_tuning')
oof_files = sorted(glob.glob(os.path.join(oof_dir, '*_oof.npy')))
test_files = sorted(glob.glob(os.path.join(oof_dir, '*_test.npy')))

if not oof_files:
    print("L3 OOF 파일이 없습니다. L3 파일들을 먼저 실행하세요.")
    sys.exit(1)

model_names = []
oof_preds = []
test_preds = []

for oof_f, test_f in zip(oof_files, test_files):
    name = os.path.basename(oof_f).replace('_oof.npy', '')
    model_names.append(name)
    oof_preds.append(np.load(oof_f))
    test_preds.append(np.load(test_f))
    print(f"  로드: {name}")

print(f"\n블렌딩 대상: {len(model_names)}개 모델")

# 최적 가중치 탐색 (그리드 서치)
n_models = len(oof_preds)
best_rmse = float('inf')
best_weights = None

if n_models == 2:
    for w0 in np.arange(0, 1.05, 0.05):
        w1 = 1.0 - w0
        pred = w0 * oof_preds[0] + w1 * oof_preds[1]
        rmse = np.sqrt(np.mean((pred - y_true) ** 2))
        if rmse < best_rmse:
            best_rmse = rmse
            best_weights = [w0, w1]

elif n_models == 3:
    for w0 in np.arange(0, 1.05, 0.05):
        for w1 in np.arange(0, 1.05 - w0, 0.05):
            w2 = 1.0 - w0 - w1
            if w2 < 0:
                continue
            pred = w0 * oof_preds[0] + w1 * oof_preds[1] + w2 * oof_preds[2]
            rmse = np.sqrt(np.mean((pred - y_true) ** 2))
            if rmse < best_rmse:
                best_rmse = rmse
                best_weights = [w0, w1, w2]
else:
    # 단순 평균
    best_weights = [1.0 / n_models] * n_models
    pred = sum(w * p for w, p in zip(best_weights, oof_preds))
    best_rmse = np.sqrt(np.mean((pred - y_true) ** 2))

# 결과 출력
print(f"\n최적 가중치:")
for name, w in zip(model_names, best_weights):
    print(f"  {name}: {w:.2f}")

# 개별 모델 RMSE도 표시
print(f"\n개별 모델 OOF RMSE:")
for name, oof in zip(model_names, oof_preds):
    rmse = np.sqrt(np.mean((oof - y_true) ** 2))
    print(f"  {name}: {rmse:,.0f} 만원")

print(f"  ──────────────────────────")
print(f"  블렌딩 OOF RMSE: {best_rmse:,.0f} 만원")

# 최종 제출 파일
final_test = sum(w * p for w, p in zip(best_weights, test_preds))
save_submission(sample_sub, final_test, 'submission_l4_01_bl.csv')
