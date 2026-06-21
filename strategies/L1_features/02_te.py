"""
[L1-02] 타겟 인코딩
──────────────────────────
레이어  : L1 (피처 확정, 모델: LightGBM 고정)
축약명  : TE
주요 전략: Gu, Dong을 타겟 평균값으로 인코딩 (K-Fold 누수 방지 + smoothing)
차별점  : LabelEncoding의 무의미한 숫자 대신 실제 가격 정보를 피처에 반영
결과    : OOF RMSE 2,445
제출파일: submission_l1_02_te.csv
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import lightgbm as lgb
from utils import (load_data, base_preprocess, add_target_encoding,
                   encode_categoricals, kfold_train_predict, save_submission)

train, test, sample_sub = load_data()

train_p = base_preprocess(train)
test_p = base_preprocess(test)
train_p, test_p = add_target_encoding(train_p, test_p)
train_p, test_p = encode_categoricals(train_p, test_p, as_category=True)

X = train_p.drop(columns=['Target'])
y = np.log1p(train_p['Target'])
X_test = test_p


def model_fn():
    return lgb.LGBMRegressor(
        objective='regression', metric='rmse',
        learning_rate=0.05, n_estimators=1000,
        random_state=42, verbose=-1,
    )


def fit_fn(model, X_tr, y_tr, X_va, y_va):
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
    return model


print("[L1-02 TE] 5-Fold CV (모델: LightGBM 고정)")
oof_pred, test_pred, rmse = kfold_train_predict(X, y, X_test, model_fn, fit_fn)

save_submission(sample_sub, test_pred, 'submission_l1_02_te.csv')
