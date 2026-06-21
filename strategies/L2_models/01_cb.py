"""
[L2-01] CatBoost
──────────────────────────
레이어  : L2 (모델 테스트, 피처: L1 확정 FE)
축약명  : CB
주요 전략: CatBoost 모델, 범주형 네이티브 처리
결과    : OOF RMSE 2,310 ★ L2 최선
제출파일: submission_l2_01_cb.csv
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from catboost import CatBoostRegressor
from utils import (load_data, base_preprocess, add_feature_engineering,
                   encode_categoricals, kfold_train_predict, save_submission,
                   CAT_FEATURES)

train, test, sample_sub = load_data()

train_p = add_feature_engineering(base_preprocess(train))
test_p = add_feature_engineering(base_preprocess(test))
train_p, test_p = encode_categoricals(train_p, test_p, as_category=False)

X = train_p.drop(columns=['Target'])
y = np.log1p(train_p['Target'])
X_test = test_p

cat_indices = [X.columns.get_loc(c) for c in CAT_FEATURES]


def model_fn():
    return CatBoostRegressor(
        loss_function='RMSE',
        learning_rate=0.05, iterations=1000,
        random_seed=42, verbose=0,
        early_stopping_rounds=50,
    )


def fit_fn(model, X_tr, y_tr, X_va, y_va):
    model.fit(X_tr, y_tr, eval_set=(X_va, y_va), cat_features=cat_indices)
    return model


print("[L2-01 CB] 5-Fold CV (피처: L1 확정 FE)")
oof_pred, test_pred, rmse = kfold_train_predict(X, y, X_test, model_fn, fit_fn)

save_submission(sample_sub, test_pred, 'submission_l2_01_cb.csv')
