"""
[L2-02] Ridge 회귀
──────────────────────────
레이어  : L2 (모델 테스트, 피처: L1 확정 FE)
축약명  : LR
주요 전략: Ridge 선형 회귀, 트리와 다른 학습 방식으로 블렌딩 다양성 확보
결과    : OOF RMSE 5,765 (단독 성능 낮지만 블렌딩 다양성 기여 가능)
제출파일: submission_l2_02_lr.csv
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from utils import (load_data, base_preprocess, add_feature_engineering,
                   encode_categoricals, kfold_train_predict, save_submission)

train, test, sample_sub = load_data()

train_p = add_feature_engineering(base_preprocess(train))
test_p = add_feature_engineering(base_preprocess(test))
train_p, test_p = encode_categoricals(train_p, test_p, as_category=False)

X = train_p.drop(columns=['Target'])
y = np.log1p(train_p['Target'])
X_test = test_p


def model_fn():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=1.0)),
    ])


def fit_fn(model, X_tr, y_tr, X_va, y_va):
    model.fit(X_tr, y_tr)
    return model


print("[L2-02 LR] 5-Fold CV (피처: L1 확정 FE)")
oof_pred, test_pred, rmse = kfold_train_predict(X, y, X_test, model_fn, fit_fn)

save_submission(sample_sub, test_pred, 'submission_l2_02_lr.csv')
