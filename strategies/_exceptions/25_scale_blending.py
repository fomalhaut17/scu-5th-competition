"""
[L4-25] 하이브리드 타겟 스케일 블렌딩 (Scale-Blending)
──────────────────────────
축약명  : SCALE+STK
주요 전략: log1p 모델 + 원본 스케일 모델 4개를 Ridge 스태킹
차별점  : 고가 아파트 과소예측 보정 (log→RMSLE 최적화 vs 원본→RMSE 최적화)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from catboost import CatBoostRegressor
import lightgbm as lgb
from utils import load_data, record_result
import warnings
warnings.filterwarnings('ignore')

N_SPLITS = 5
CAT_FEATURES = ['Gu', 'Dong']

train, test, sample_sub = load_data()
y_true = train['Target'].values

# === 구별 트렌드 보정 계수 ===
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

# === 전처리 실행 ===
train_p = add_feature_engineering(base_preprocess(train))
test_p = add_feature_engineering(base_preprocess(test))

train_cb, test_cb = encode_categoricals(train_p, test_p, as_category=False)
X_cb = train_cb.drop(columns=['Target'])
X_test_cb = test_cb
cat_indices = [X_cb.columns.get_loc(c) for c in CAT_FEATURES]

train_lgb, test_lgb = encode_categoricals(train_p, test_p, as_category=True)
X_lgb = train_lgb.drop(columns=['Target'])
X_test_lgb = test_lgb

y_log = np.log1p(train_p['Target'])
y_raw = train_p['Target'].values

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# ========================================
# 4개 베이스 모델 OOF + Test 예측
# M1: CatBoost log, M2: CatBoost raw, M3: LGB log, M4: LGB raw
# ========================================
print("=" * 60)
print("[전략 25] 하이브리드 타겟 스케일 블렌딩")
print("=" * 60)

oof = {k: np.zeros(len(X_cb)) for k in ['cb_log', 'cb_raw', 'lgb_log', 'lgb_raw']}
test_pred = {k: np.zeros(len(X_test_cb)) for k in ['cb_log', 'cb_raw', 'lgb_log', 'lgb_raw']}

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_cb)):
    print(f"\n  Fold {fold+1}/{N_SPLITS}")

    # M1: CatBoost + log1p target
    m1 = CatBoostRegressor(loss_function='RMSE', random_seed=42, verbose=0,
                            iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
    m1.fit(X_cb.iloc[tr_idx], y_log.iloc[tr_idx],
           eval_set=(X_cb.iloc[va_idx], y_log.iloc[va_idx]), cat_features=cat_indices)
    oof['cb_log'][va_idx] = np.expm1(m1.predict(X_cb.iloc[va_idx]))
    test_pred['cb_log'] += np.expm1(m1.predict(X_test_cb)) / N_SPLITS
    rmse1 = np.sqrt(np.mean((oof['cb_log'][va_idx] - y_true[va_idx]) ** 2))
    print(f"    M1 CB-log  RMSE: {rmse1:,.0f}")

    # M2: CatBoost + raw target
    m2 = CatBoostRegressor(loss_function='RMSE', random_seed=42, verbose=0,
                            iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
    m2.fit(X_cb.iloc[tr_idx], y_raw[tr_idx],
           eval_set=(X_cb.iloc[va_idx], y_raw[va_idx]), cat_features=cat_indices)
    oof['cb_raw'][va_idx] = m2.predict(X_cb.iloc[va_idx])
    test_pred['cb_raw'] += m2.predict(X_test_cb) / N_SPLITS
    rmse2 = np.sqrt(np.mean((oof['cb_raw'][va_idx] - y_true[va_idx]) ** 2))
    print(f"    M2 CB-raw  RMSE: {rmse2:,.0f}")

    # M3: LightGBM + log1p target
    m3 = lgb.LGBMRegressor(objective='regression', metric='rmse',
                            verbose=-1, random_state=42, n_estimators=3000, **LGB_PARAMS)
    m3.fit(X_lgb.iloc[tr_idx], y_log.iloc[tr_idx],
           eval_set=[(X_lgb.iloc[va_idx], y_log.iloc[va_idx])],
           callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])
    oof['lgb_log'][va_idx] = np.expm1(m3.predict(X_lgb.iloc[va_idx]))
    test_pred['lgb_log'] += np.expm1(m3.predict(X_test_lgb)) / N_SPLITS
    rmse3 = np.sqrt(np.mean((oof['lgb_log'][va_idx] - y_true[va_idx]) ** 2))
    print(f"    M3 LGB-log RMSE: {rmse3:,.0f}")

    # M4: LightGBM + raw target
    m4 = lgb.LGBMRegressor(objective='regression', metric='rmse',
                            verbose=-1, random_state=42, n_estimators=3000, **LGB_PARAMS)
    m4.fit(X_lgb.iloc[tr_idx], y_raw[tr_idx],
           eval_set=[(X_lgb.iloc[va_idx], y_raw[va_idx])],
           callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])
    oof['lgb_raw'][va_idx] = m4.predict(X_lgb.iloc[va_idx])
    test_pred['lgb_raw'] += m4.predict(X_test_lgb) / N_SPLITS
    rmse4 = np.sqrt(np.mean((oof['lgb_raw'][va_idx] - y_true[va_idx]) ** 2))
    print(f"    M4 LGB-raw RMSE: {rmse4:,.0f}")

# ========================================
# 개별 모델 OOF RMSE
# ========================================
print(f"\n{'=' * 60}")
print("개별 모델 OOF RMSE")
print(f"{'=' * 60}")
for name, vals in oof.items():
    rmse = np.sqrt(np.mean((vals - y_true) ** 2))
    print(f"  {name:10s}: {rmse:,.0f}")

# ========================================
# 방법 A: 4모델 단순 평균
# ========================================
avg_oof = (oof['cb_log'] + oof['cb_raw'] + oof['lgb_log'] + oof['lgb_raw']) / 4
avg_test = (test_pred['cb_log'] + test_pred['cb_raw'] + test_pred['lgb_log'] + test_pred['lgb_raw']) / 4
rmse_a = np.sqrt(np.mean((avg_oof - y_true) ** 2))
print(f"\n  [A] 4모델 단순 평균 OOF RMSE: {rmse_a:,.0f}")

# ========================================
# 방법 B: 가중 평균 (원본 스케일에 가중치)
# ========================================
weights_list = [
    (0.25, 0.25, 0.25, 0.25, "균등"),
    (0.20, 0.30, 0.20, 0.30, "원본 60%"),
    (0.15, 0.35, 0.15, 0.35, "원본 70%"),
    (0.10, 0.40, 0.10, 0.40, "원본 80%"),
    (0.10, 0.30, 0.10, 0.50, "Gemini 추천"),
]

print(f"\n{'=' * 60}")
print("가중 평균 실험")
print(f"{'=' * 60}")
best_w_rmse = float('inf')
best_w_config = None
for w1, w2, w3, w4, label in weights_list:
    w_oof = w1*oof['cb_log'] + w2*oof['cb_raw'] + w3*oof['lgb_log'] + w4*oof['lgb_raw']
    w_rmse = np.sqrt(np.mean((w_oof - y_true) ** 2))
    print(f"  [{label:10s}] ({w1:.2f},{w2:.2f},{w3:.2f},{w4:.2f}) → RMSE: {w_rmse:,.0f}")
    if w_rmse < best_w_rmse:
        best_w_rmse = w_rmse
        best_w_config = (w1, w2, w3, w4, label)

w1, w2, w3, w4, wlabel = best_w_config
best_w_test = w1*test_pred['cb_log'] + w2*test_pred['cb_raw'] + w3*test_pred['lgb_log'] + w4*test_pred['lgb_raw']

# ========================================
# 방법 C: Ridge 스태킹
# ========================================
stack_train = np.column_stack([oof['cb_log'], oof['cb_raw'], oof['lgb_log'], oof['lgb_raw']])
stack_test = np.column_stack([test_pred['cb_log'], test_pred['cb_raw'], test_pred['lgb_log'], test_pred['lgb_raw']])

print(f"\n{'=' * 60}")
print("Ridge 스태킹")
print(f"{'=' * 60}")

best_ridge_rmse = float('inf')
best_ridge_alpha = None
best_ridge_oof = None
best_ridge_test = None

for alpha in [0.1, 0.5, 1.0, 5.0, 10.0]:
    stack_oof = np.zeros(len(y_true))
    stack_test_pred = np.zeros(len(X_test_cb))

    for fold, (tr_idx, va_idx) in enumerate(kf.split(stack_train)):
        meta = Ridge(alpha=alpha)
        meta.fit(stack_train[tr_idx], y_true[tr_idx])
        stack_oof[va_idx] = meta.predict(stack_train[va_idx])
        stack_test_pred += meta.predict(stack_test) / N_SPLITS

    ridge_rmse = np.sqrt(np.mean((stack_oof - y_true) ** 2))
    print(f"  Ridge(alpha={alpha:5.1f}) OOF RMSE: {ridge_rmse:,.0f}")

    if ridge_rmse < best_ridge_rmse:
        best_ridge_rmse = ridge_rmse
        best_ridge_alpha = alpha
        best_ridge_oof = stack_oof.copy()
        best_ridge_test = stack_test_pred.copy()

# ========================================
# 최종 비교 + 트렌드 보정
# ========================================
print(f"\n{'=' * 60}")
print("최종 비교 (트렌드 보정 전)")
print(f"{'=' * 60}")
print(f"  [A] 단순 평균          : {rmse_a:,.0f}")
print(f"  [B] 가중 평균 ({wlabel}): {best_w_rmse:,.0f}")
print(f"  [C] Ridge(α={best_ridge_alpha}) : {best_ridge_rmse:,.0f}")
print(f"  ───────────────────────────")
print(f"  전략 08 (log only)     : 2,234")

results = {
    'A': (avg_test, rmse_a, '4모델 단순 평균'),
    'B': (best_w_test, best_w_rmse, f'가중 평균 ({wlabel})'),
    'C': (best_ridge_test, best_ridge_rmse, f'Ridge(α={best_ridge_alpha})'),
}

best_key = min(results, key=lambda k: results[k][1])
best_test_pred, best_rmse, best_desc = results[best_key]

final_pred = best_test_pred * trend_correction

print(f"\n  최선: [{best_key}] {best_desc} → OOF {best_rmse:,.0f}")

# === 제출 파일 ===
sub = sample_sub.copy()
sub['Target'] = final_pred
sub.to_csv(os.path.join(os.path.dirname(__file__), '..', '..', 'submission_l4_25_scale.csv'), index=False)
print("제출 파일 생성 완료: submission_l4_25_scale.csv")

record_result('L4', 25, 'SCALE+STK',
              f'Scale Blending ({best_desc}) + 구별 트렌드', best_rmse, 'tested')
