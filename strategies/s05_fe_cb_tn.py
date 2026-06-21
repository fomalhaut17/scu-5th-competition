"""
[S05] 피처 엔지니어링 + CatBoost + 하이퍼파라미터 튜닝
──────────────────────────
축약명  : FE+CB+TN
구성    : S01(FE) + S02(CB) + S04(TN)
주요 전략: FE 추가 피처 + CatBoost 모델 + Optuna 튜닝
차별점  : 3개 전략의 시너지 확인
결과    : OOF RMSE 2,275 (baseline 대비 -512, -18.4%)
제출파일: submission_s05_fe_cb_tn.csv
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import optuna
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from utils import load_data, base_preprocess, add_feature_engineering, encode_categoricals, kfold_train_predict, save_submission, CAT_FEATURES, N_SPLITS

optuna.logging.set_verbosity(optuna.logging.WARNING)

train, test, sample_sub = load_data()

train_p = add_feature_engineering(base_preprocess(train))
test_p = add_feature_engineering(base_preprocess(test))
train_p, test_p = encode_categoricals(train_p, test_p, as_category=False)

X = train_p.drop(columns=['Target'])
y = np.log1p(train_p['Target'])
X_test = test_p

cat_indices = [X.columns.get_loc(c) for c in CAT_FEATURES]


def objective(trial):
    params = {
        'loss_function': 'RMSE',
        'random_seed': 42,
        'verbose': 0,
        'iterations': 2000,
        'early_stopping_rounds': 50,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'random_strength': trial.suggest_float('random_strength', 1e-3, 10.0, log=True),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 30),
    }

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    oof_pred = np.zeros(len(X))

    for train_idx, val_idx in kf.split(X):
        model = CatBoostRegressor(**params)
        model.fit(
            X.iloc[train_idx], y.iloc[train_idx],
            eval_set=(X.iloc[val_idx], y.iloc[val_idx]),
            cat_features=cat_indices,
        )
        oof_pred[val_idx] = np.expm1(model.predict(X.iloc[val_idx]))

    return np.sqrt(np.mean((oof_pred - np.expm1(y.values)) ** 2))


print("Optuna 탐색 (100 trials × 5-Fold)...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100, show_progress_bar=True)

print(f"\n최적 RMSE: {study.best_value:,.0f} 만원")
print(f"최적 파라미터:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")

best = study.best_params


def model_fn():
    return CatBoostRegressor(
        loss_function='RMSE',
        random_seed=42, verbose=0,
        iterations=2000, early_stopping_rounds=50,
        **best,
    )


def fit_fn(model, X_tr, y_tr, X_va, y_va):
    model.fit(X_tr, y_tr, eval_set=(X_va, y_va), cat_features=cat_indices)
    return model


print(f"\n[S05 FE+CB+TN] 최적 파라미터로 5-Fold CV")
oof_pred, test_pred, rmse = kfold_train_predict(X, y, X_test, model_fn, fit_fn)

save_submission(sample_sub, test_pred, 'submission_s05_fe_cb_tn.csv')
