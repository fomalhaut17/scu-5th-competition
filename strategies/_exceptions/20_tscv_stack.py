"""
[L4-20] TimeSeriesSplit 스태킹
──────────────────────────
축약명  : TSCV+STK
주요 전략: KFold 대신 TimeSeriesSplit으로 시간 순서 보존
차별점  : 과거→미래 예측을 CV에서도 재현
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
from catboost import CatBoostRegressor
import lightgbm as lgb
from utils import load_data, record_result
import warnings
warnings.filterwarnings('ignore')

N_SPLITS = 5
CAT_FEATURES = ['Gu', 'Dong']

# === 데이터 로드 ===
train, test, sample_sub = load_data()
y_true = train['Target'].values

# === 구별 트렌드 보정 계수 계산 ===
last_train_ym = train['Transaction_YearMonth'].max()
last_train_seq = (last_train_ym // 100 - 2024) * 12 + last_train_ym % 100

gu_growth = {}
for gu in train['Gu'].unique():
    monthly = train[train['Gu'] == gu].groupby('Transaction_YearMonth')['Target'].mean()
    gu_growth[gu] = monthly.pct_change().dropna().mean()

test_seq = (test['Transaction_YearMonth'] // 100 - 2024) * 12 + test['Transaction_YearMonth'] % 100
months_ahead = test_seq.values - last_train_seq
trend_correction = np.array([(1 + gu_growth[gu]) ** m for gu, m in zip(test['Gu'], months_ahead)])

# === 전처리 ===
def base_preprocess(df):
    df = df.copy()
    df['Distance_to_Subway'] = df['Distance_to_Subway'].fillna(df['Distance_to_Subway'].median())
    df['Age'] = 2026 - df['Year_Built']
    df['Year'] = df['Transaction_YearMonth'] // 100
    df['Month'] = df['Transaction_YearMonth'] % 100
    df = df.drop(columns=['ID', 'Transaction_YearMonth', 'Year_Built'])
    return df

def add_feature_engineering(df):
    df = df.copy()
    df['YearMonth_Seq'] = (df['Year'] - 2024) * 12 + df['Month']
    df['Area_x_Floor'] = df['Exclusive_Area'] * df['Floor']
    df['Floor_per_Area'] = df['Floor'] / df['Exclusive_Area']
    df['Brand_x_Area'] = df['Brand_Apartment'] * df['Exclusive_Area']
    return df

def encode_categoricals(train_df, test_df, as_category=False):
    train_df, test_df = train_df.copy(), test_df.copy()
    for col in CAT_FEATURES:
        le = LabelEncoder()
        combined = list(train_df[col].astype(str)) + list(test_df[col].astype(str))
        le.fit(combined)
        train_df[col] = le.transform(train_df[col].astype(str))
        test_df[col] = le.transform(test_df[col].astype(str))
        if as_category:
            train_df[col] = train_df[col].astype('category')
            test_df[col] = test_df[col].astype('category')
    return train_df, test_df

# 시간 순서로 정렬
sort_idx = train.sort_values('Transaction_YearMonth').index
train_sorted = train.loc[sort_idx].reset_index(drop=True)
y_true_sorted = train_sorted['Target'].values

train_p = add_feature_engineering(base_preprocess(train_sorted))
test_p = add_feature_engineering(base_preprocess(test))

# ========================================
# 1단계: 베이스 모델 OOF (TimeSeriesSplit)
# ========================================
train_cb, test_cb = encode_categoricals(train_p, test_p, as_category=False)
X_cb = train_cb.drop(columns=['Target'])
y_cb = np.log1p(train_cb['Target'])
X_test_cb = test_cb
cat_indices = [X_cb.columns.get_loc(c) for c in CAT_FEATURES]

train_lgb, test_lgb = encode_categoricals(train_p, test_p, as_category=True)
X_lgb = train_lgb.drop(columns=['Target'])
y_lgb = np.log1p(train_lgb['Target'])
X_test_lgb = test_lgb

CB_PARAMS = {
    'learning_rate': 0.010118898857677389,
    'depth': 3,
    'l2_leaf_reg': 4.944272225334265,
    'bagging_temperature': 1.4823308606638113,
    'random_strength': 0.4685604025205004,
    'min_data_in_leaf': 46,
}

LGB_PARAMS = {
    'learning_rate': 0.022992006545037823,
    'num_leaves': 110,
    'max_depth': 3,
    'min_child_samples': 27,
    'subsample': 0.9312452053625488,
    'colsample_bytree': 0.8234901310320267,
    'reg_alpha': 0.012423757285817386,
    'reg_lambda': 0.04673443002441543,
}

tscv = TimeSeriesSplit(n_splits=N_SPLITS)

cb_oof = np.full(len(X_cb), np.nan)
cb_test_pred = np.zeros(len(X_test_cb))
lgb_oof = np.full(len(X_lgb), np.nan)
lgb_test_pred = np.zeros(len(X_test_lgb))

print("=" * 50)
print("[1/3] CatBoost 학습 (TimeSeriesSplit)")
print("=" * 50)
for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_cb)):
    print(f"  Fold {fold+1}/{N_SPLITS} (train: {len(tr_idx)}, val: {len(va_idx)})", end=" → ")
    m = CatBoostRegressor(loss_function='RMSE', random_seed=42, verbose=0,
                          iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
    m.fit(X_cb.iloc[tr_idx], y_cb.iloc[tr_idx],
          eval_set=(X_cb.iloc[va_idx], y_cb.iloc[va_idx]), cat_features=cat_indices)
    cb_oof[va_idx] = np.expm1(m.predict(X_cb.iloc[va_idx]))
    cb_test_pred += np.expm1(m.predict(X_test_cb)) / N_SPLITS
    fold_rmse = np.sqrt(np.mean((cb_oof[va_idx] - y_true_sorted[va_idx]) ** 2))
    print(f"RMSE: {fold_rmse:,.0f}")

valid_mask = ~np.isnan(cb_oof)
cb_oof_rmse = np.sqrt(np.mean((cb_oof[valid_mask] - y_true_sorted[valid_mask]) ** 2))
print(f"  OOF RMSE (val만): {cb_oof_rmse:,.0f} ({valid_mask.sum()}건)")

print("\n" + "=" * 50)
print("[2/3] LightGBM 학습 (TimeSeriesSplit)")
print("=" * 50)
for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_lgb)):
    print(f"  Fold {fold+1}/{N_SPLITS} (train: {len(tr_idx)}, val: {len(va_idx)})", end=" → ")
    m = lgb.LGBMRegressor(objective='regression', metric='rmse',
                          verbose=-1, random_state=42, n_estimators=3000, **LGB_PARAMS)
    m.fit(X_lgb.iloc[tr_idx], y_lgb.iloc[tr_idx],
          eval_set=[(X_lgb.iloc[va_idx], y_lgb.iloc[va_idx])],
          callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])
    lgb_oof[va_idx] = np.expm1(m.predict(X_lgb.iloc[va_idx]))
    lgb_test_pred += np.expm1(m.predict(X_test_lgb)) / N_SPLITS
    fold_rmse = np.sqrt(np.mean((lgb_oof[va_idx] - y_true_sorted[va_idx]) ** 2))
    print(f"RMSE: {fold_rmse:,.0f}")

lgb_oof_rmse = np.sqrt(np.mean((lgb_oof[valid_mask] - y_true_sorted[valid_mask]) ** 2))
print(f"  OOF RMSE (val만): {lgb_oof_rmse:,.0f} ({valid_mask.sum()}건)")

# ========================================
# 2단계: Ridge 스태킹 (OOF 있는 샘플만)
# ========================================
print("\n" + "=" * 50)
print("[3/3] Ridge 스태킹 + 구별 트렌드 보정")
print("=" * 50)

stack_train = np.column_stack([cb_oof[valid_mask], lgb_oof[valid_mask]])
stack_y = y_true_sorted[valid_mask]
stack_test = np.column_stack([cb_test_pred, lgb_test_pred])

stack_oof = np.zeros(len(stack_y))
stack_test_pred = np.zeros(len(stack_test))

# 스태킹도 TimeSeriesSplit
tscv_meta = TimeSeriesSplit(n_splits=N_SPLITS)
for fold, (tr_idx, va_idx) in enumerate(tscv_meta.split(stack_train)):
    meta = Ridge(alpha=1.0)
    meta.fit(stack_train[tr_idx], stack_y[tr_idx])
    stack_oof[va_idx] = meta.predict(stack_train[va_idx])
    stack_test_pred += meta.predict(stack_test) / N_SPLITS

meta_valid = ~np.isnan(stack_oof) & (stack_oof != 0)
stack_rmse = np.sqrt(np.mean((stack_oof[meta_valid] - stack_y[meta_valid]) ** 2))
print(f"스태킹 OOF RMSE: {stack_rmse:,.0f}")
print(f"Ridge 계수: CB={meta.coef_[0]:.4f}, LGB={meta.coef_[1]:.4f}, 절편={meta.intercept_:.1f}")

# 구별 트렌드 보정
final_pred = stack_test_pred * trend_correction

print(f"\n트렌드 보정 전 평균: {stack_test_pred.mean():,.0f}")
print(f"트렌드 보정 후 평균: {final_pred.mean():,.0f}")

# === 비교 ===
print("\n" + "=" * 50)
print("비교: 전략 08 OOF 2,234 / Public 2,155")
print("=" * 50)

# === 제출 파일 & 기록 ===
sub = sample_sub.copy()
sub['Target'] = final_pred
sub.to_csv(os.path.join(os.path.dirname(__file__), '..', '..', 'submission_l4_20_tscv.csv'), index=False)
print("제출 파일 생성 완료")

record_result('L4', 20, 'TSCV+STK', 'TimeSeriesSplit CB+LGB Ridge 스태킹 + 구별 트렌드', stack_rmse, 'tested')
