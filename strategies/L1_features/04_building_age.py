"""
[L1-04] Building Age (거래 시점 기준)
──────────────────────────
레이어  : L1 (피처 확정, 모델: LightGBM 고정)
축약명  : BAGE
주요 전략: Age를 2026 고정 대신 Transaction_Year - Year_Built로 변경
차별점  : 거래 당시 실제 건물 나이 반영
제출파일: submission_l1_04_bage.csv
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import lightgbm as lgb
from utils import (load_data, add_feature_engineering,
                   encode_categoricals, kfold_train_predict,
                   save_submission, record_result)


def base_preprocess_building_age(df):
    df = df.copy()
    df['Distance_to_Subway'] = df['Distance_to_Subway'].fillna(df['Distance_to_Subway'].median())
    df['Year'] = df['Transaction_YearMonth'] // 100
    df['Month'] = df['Transaction_YearMonth'] % 100
    df['Age'] = df['Year'] - df['Year_Built']
    df = df.drop(columns=['ID', 'Transaction_YearMonth', 'Year_Built'])
    return df


train, test, sample_sub = load_data()

train_p = add_feature_engineering(base_preprocess_building_age(train))
test_p = add_feature_engineering(base_preprocess_building_age(test))
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


print("[L1-04 BAGE] 5-Fold CV (모델: LightGBM 고정)")
print("변경점: Age = Transaction_Year - Year_Built (거래 시점 기준)")
oof_pred, test_pred, rmse = kfold_train_predict(X, y, X_test, model_fn, fit_fn)

save_submission(sample_sub, test_pred, 'submission_l1_04_bage.csv')
record_result('L1', 4, 'BAGE', 'Building Age (거래시점 기준 Age)', rmse, 'tested')
