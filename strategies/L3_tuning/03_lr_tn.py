"""
[L3-03] Ridge 튜닝
──────────────────────────
레이어  : L3 (튜닝, 피처: L1 확정 FE)
축약명  : LR+TN
주요 전략: Optuna로 Ridge alpha 최적화 (5-Fold CV)
결과    : OOF RMSE 5,765 (튜닝 효과 없음, 선형 모델 한계)
제출파일: submission_l3_03_lr_tn.csv
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import optuna
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from utils import (load_data, base_preprocess, add_feature_engineering,
                   encode_categoricals, kfold_train_predict, save_submission,
                   N_SPLITS)

optuna.logging.set_verbosity(optuna.logging.WARNING)

train, test, sample_sub = load_data()

train_p = add_feature_engineering(base_preprocess(train))
test_p = add_feature_engineering(base_preprocess(test))
train_p, test_p = encode_categoricals(train_p, test_p, as_category=False)

X = train_p.drop(columns=['Target'])
y = np.log1p(train_p['Target'])
X_test = test_p


def objective(trial):
    alpha = trial.suggest_float('alpha', 1e-3, 1000.0, log=True)

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    oof = np.zeros(len(X))
    for tr_idx, va_idx in kf.split(X):
        m = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=alpha))])
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        oof[va_idx] = np.expm1(m.predict(X.iloc[va_idx]))
    return np.sqrt(np.mean((oof - np.expm1(y.values)) ** 2))


print("Optuna 탐색 (100 trials × 5-Fold)...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100, show_progress_bar=True)

print(f"\n최적 RMSE: {study.best_value:,.0f} 만원")
print(f"  alpha: {study.best_params['alpha']}")

best_alpha = study.best_params['alpha']


def model_fn():
    return Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=best_alpha))])


def fit_fn(model, X_tr, y_tr, X_va, y_va):
    model.fit(X_tr, y_tr)
    return model


print(f"\n[L3-03 LR+TN] 최적 파라미터로 5-Fold CV")
oof_pred, test_pred, rmse = kfold_train_predict(X, y, X_test, model_fn, fit_fn)

save_submission(sample_sub, test_pred, 'submission_l3_03_lr_tn.csv')

np.save(os.path.join(os.path.dirname(__file__), 'lr_tn_oof.npy'), oof_pred)
np.save(os.path.join(os.path.dirname(__file__), 'lr_tn_test.npy'), test_pred)
