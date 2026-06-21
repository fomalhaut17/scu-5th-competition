"""
[S03] 피처 엔지니어링 + CatBoost
──────────────────────────
축약명  : FE+CB
구성    : S01(FE) + S02(CB)
주요 전략: FE의 추가 피처 4개 + CB의 CatBoost 모델
차별점  : 두 전략의 개별 효과가 합산되는지 확인
결과    : OOF RMSE 2,310 (baseline 대비 -477, -17.1%)
제출파일: submission_s03_fe_cb.csv
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from catboost import CatBoostRegressor
from utils import load_data, base_preprocess, add_feature_engineering, encode_categoricals, kfold_train_predict, save_submission, CAT_FEATURES

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


print("[S03 FE+CB] 5-Fold CV")
oof_pred, test_pred, rmse = kfold_train_predict(X, y, X_test, model_fn, fit_fn)

save_submission(sample_sub, test_pred, 'submission_s03_fe_cb.csv')
