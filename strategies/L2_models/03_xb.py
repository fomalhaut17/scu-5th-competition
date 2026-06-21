"""
[L2-03] XGBoost
──────────────────────────
레이어  : L2 (모델 테스트, 피처: L1 확정 FE)
축약명  : XB
주요 전략: XGBoost 모델
결과    : OOF RMSE 2,644
제출파일: submission_l2_03_xb.csv
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import xgboost as xgb
from utils import (load_data, base_preprocess, add_feature_engineering,
                   encode_categoricals, kfold_train_predict, save_submission)

train, test, sample_sub = load_data()

train_p = add_feature_engineering(base_preprocess(train))
test_p = add_feature_engineering(base_preprocess(test))
train_p, test_p = encode_categoricals(train_p, test_p, as_category=True)

X = train_p.drop(columns=['Target'])
y = np.log1p(train_p['Target'])
X_test = test_p


def model_fn():
    return xgb.XGBRegressor(
        objective='reg:squarederror', eval_metric='rmse',
        learning_rate=0.05, n_estimators=1000,
        random_state=42, verbosity=0,
        enable_categorical=True,
    )


def fit_fn(model, X_tr, y_tr, X_va, y_va):
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return model


print("[L2-03 XB] 5-Fold CV (피처: L1 확정 FE)")
oof_pred, test_pred, rmse = kfold_train_predict(X, y, X_test, model_fn, fit_fn)

save_submission(sample_sub, test_pred, 'submission_l2_03_xb.csv')
