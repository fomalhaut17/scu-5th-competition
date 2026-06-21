"""
[L3-02] LightGBM 튜닝
──────────────────────────
레이어  : L3 (튜닝, 피처: L1 확정 FE)
축약명  : LGB+TN
주요 전략: Optuna로 LightGBM 하이퍼파라미터 최적화 (5-Fold CV)
결과    : OOF RMSE 2,298
제출파일: submission_l3_02_lgb_tn.csv
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import KFold
from utils import (load_data, base_preprocess, add_feature_engineering,
                   encode_categoricals, kfold_train_predict, save_submission,
                   N_SPLITS)

optuna.logging.set_verbosity(optuna.logging.WARNING)

train, test, sample_sub = load_data()

train_p = add_feature_engineering(base_preprocess(train))
test_p = add_feature_engineering(base_preprocess(test))
train_p, test_p = encode_categoricals(train_p, test_p, as_category=True)

X = train_p.drop(columns=['Target'])
y = np.log1p(train_p['Target'])
X_test = test_p


def objective(trial):
    params = {
        'objective': 'regression', 'metric': 'rmse',
        'verbose': -1, 'random_state': 42, 'n_estimators': 2000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15),
        'num_leaves': trial.suggest_int('num_leaves', 16, 128),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
    }

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    oof = np.zeros(len(X))
    for tr_idx, va_idx in kf.split(X):
        m = lgb.LGBMRegressor(**params)
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx],
              eval_set=[(X.iloc[va_idx], y.iloc[va_idx])],
              callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
        oof[va_idx] = np.expm1(m.predict(X.iloc[va_idx]))
    return np.sqrt(np.mean((oof - np.expm1(y.values)) ** 2))


print("Optuna 탐색 (100 trials × 5-Fold)...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100, show_progress_bar=True)

print(f"\n최적 RMSE: {study.best_value:,.0f} 만원")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")

best = study.best_params


def model_fn():
    return lgb.LGBMRegressor(
        objective='regression', metric='rmse',
        verbose=-1, random_state=42, n_estimators=2000, **best)


def fit_fn(model, X_tr, y_tr, X_va, y_va):
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
    return model


print(f"\n[L3-02 LGB+TN] 최적 파라미터로 5-Fold CV")
oof_pred, test_pred, rmse = kfold_train_predict(X, y, X_test, model_fn, fit_fn)

save_submission(sample_sub, test_pred, 'submission_l3_02_lgb_tn.csv')

np.save(os.path.join(os.path.dirname(__file__), 'lgb_tn_oof.npy'), oof_pred)
np.save(os.path.join(os.path.dirname(__file__), 'lgb_tn_test.npy'), test_pred)
