"""
OOT(Out-of-Time) 검증 테스트
──────────────────────────
목적: 마지막 3개월(202510~202512)을 holdout하여
      트렌드 보정 효과를 로컬에서 측정
기준: 전략 08 (CB+LGB Ridge 스태킹 + 구별 트렌드 보정)
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
from utils import load_data, oot_split
import warnings
warnings.filterwarnings('ignore')

N_SPLITS = 5
CAT_FEATURES = ['Gu', 'Dong']

train_raw, test_raw, sample_sub = load_data()

# === OOT 분할 ===
train_idx, val_idx, cutoff_ym = oot_split(train_raw, holdout_months=3)
oot_train = train_raw.iloc[train_idx].reset_index(drop=True)
oot_val = train_raw.iloc[val_idx].reset_index(drop=True)

print(f"OOT 분할: Train {len(oot_train)}건 (< {cutoff_ym}), Val {len(oot_val)}건 (>= {cutoff_ym})")
print(f"Val YearMonth: {sorted(oot_val['Transaction_YearMonth'].unique())}")

y_train_all = oot_train['Target'].values
y_val_true = oot_val['Target'].values

# === 구별 트렌드 보정 계수 (OOT train 기준) ===
last_train_ym = oot_train['Transaction_YearMonth'].max()
last_train_seq = (last_train_ym // 100 - 2024) * 12 + last_train_ym % 100

gu_growth = {}
for gu in oot_train['Gu'].unique():
    monthly = oot_train[oot_train['Gu'] == gu].groupby('Transaction_YearMonth')['Target'].mean()
    gu_growth[gu] = monthly.pct_change().dropna().mean()

val_seq = (oot_val['Transaction_YearMonth'] // 100 - 2024) * 12 + oot_val['Transaction_YearMonth'] % 100
months_ahead_val = val_seq.values - last_train_seq
trend_correction_val = np.array([(1 + gu_growth.get(gu, 0)) ** m for gu, m in zip(oot_val['Gu'], months_ahead_val)])

print(f"\n구별 월성장률 (OOT train 기준):")
for gu, g in sorted(gu_growth.items(), key=lambda x: -x[1]):
    print(f"  {gu:15s}: {g*100:+.2f}%")
print(f"보정 범위: {(trend_correction_val.min()-1)*100:+.1f}% ~ {(trend_correction_val.max()-1)*100:+.1f}%")

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

def encode_categoricals(train_df, val_df, as_category=False):
    train_df, val_df = train_df.copy(), val_df.copy()
    for col in CAT_FEATURES:
        le = LabelEncoder()
        combined = list(train_df[col].astype(str)) + list(val_df[col].astype(str))
        le.fit(combined)
        train_df[col] = le.transform(train_df[col].astype(str))
        val_df[col] = le.transform(val_df[col].astype(str))
        if as_category:
            train_df[col] = train_df[col].astype('category')
            val_df[col] = val_df[col].astype('category')
    return train_df, val_df

train_p = add_feature_engineering(base_preprocess(oot_train))
val_p = add_feature_engineering(base_preprocess(oot_val))

# ========================================
# 1단계: 베이스 모델 (5-Fold on OOT train → predict OOT val)
# ========================================
train_cb, val_cb = encode_categoricals(train_p, val_p, as_category=False)
X_cb = train_cb.drop(columns=['Target'])
y_cb = np.log1p(train_cb['Target'])
X_val_cb = val_cb.drop(columns=['Target'])
cat_indices = [X_cb.columns.get_loc(c) for c in CAT_FEATURES]

train_lgb, val_lgb = encode_categoricals(train_p, val_p, as_category=True)
X_lgb = train_lgb.drop(columns=['Target'])
y_lgb = np.log1p(train_lgb['Target'])
X_val_lgb = val_lgb.drop(columns=['Target'])

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

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

cb_oof = np.zeros(len(X_cb))
cb_val_pred = np.zeros(len(X_val_cb))
lgb_oof = np.zeros(len(X_lgb))
lgb_val_pred = np.zeros(len(X_val_lgb))

print("\n" + "=" * 50)
print("[1/3] CatBoost 학습")
print("=" * 50)
for fold, (tr_idx, va_idx) in enumerate(kf.split(X_cb)):
    print(f"  Fold {fold+1}/{N_SPLITS}", end=" → ")
    m = CatBoostRegressor(loss_function='RMSE', random_seed=42, verbose=0,
                          iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
    m.fit(X_cb.iloc[tr_idx], y_cb.iloc[tr_idx],
          eval_set=(X_cb.iloc[va_idx], y_cb.iloc[va_idx]), cat_features=cat_indices)
    cb_oof[va_idx] = np.expm1(m.predict(X_cb.iloc[va_idx]))
    cb_val_pred += np.expm1(m.predict(X_val_cb)) / N_SPLITS
    fold_rmse = np.sqrt(np.mean((cb_oof[va_idx] - y_train_all[va_idx]) ** 2))
    print(f"RMSE: {fold_rmse:,.0f}")
print(f"  OOF RMSE: {np.sqrt(np.mean((cb_oof - y_train_all) ** 2)):,.0f}")

print("\n" + "=" * 50)
print("[2/3] LightGBM 학습")
print("=" * 50)
for fold, (tr_idx, va_idx) in enumerate(kf.split(X_lgb)):
    print(f"  Fold {fold+1}/{N_SPLITS}", end=" → ")
    m = lgb.LGBMRegressor(objective='regression', metric='rmse',
                          verbose=-1, random_state=42, n_estimators=3000, **LGB_PARAMS)
    m.fit(X_lgb.iloc[tr_idx], y_lgb.iloc[tr_idx],
          eval_set=[(X_lgb.iloc[va_idx], y_lgb.iloc[va_idx])],
          callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])
    lgb_oof[va_idx] = np.expm1(m.predict(X_lgb.iloc[va_idx]))
    lgb_val_pred += np.expm1(m.predict(X_val_lgb)) / N_SPLITS
    fold_rmse = np.sqrt(np.mean((lgb_oof[va_idx] - y_train_all[va_idx]) ** 2))
    print(f"RMSE: {fold_rmse:,.0f}")
print(f"  OOF RMSE: {np.sqrt(np.mean((lgb_oof - y_train_all) ** 2)):,.0f}")

# ========================================
# 2단계: Ridge 스태킹
# ========================================
print("\n" + "=" * 50)
print("[3/3] Ridge 스태킹")
print("=" * 50)

stack_train = np.column_stack([cb_oof, lgb_oof])
stack_val = np.column_stack([cb_val_pred, lgb_val_pred])

stack_oof = np.zeros(len(y_train_all))
stack_val_pred = np.zeros(len(stack_val))

for fold, (tr_idx, va_idx) in enumerate(kf.split(stack_train)):
    meta = Ridge(alpha=1.0)
    meta.fit(stack_train[tr_idx], y_train_all[tr_idx])
    stack_oof[va_idx] = meta.predict(stack_train[va_idx])
    stack_val_pred += meta.predict(stack_val) / N_SPLITS

oof_rmse = np.sqrt(np.mean((stack_oof - y_train_all) ** 2))
print(f"스태킹 OOF RMSE: {oof_rmse:,.0f}")

# ========================================
# 3단계: 다양한 보정 방식 OOT 비교
# ========================================

def calc_rmse(pred, true):
    return np.sqrt(np.mean((pred - true) ** 2))

def make_gu_correction(growth_dict, gu_series, months_ahead):
    return np.array([(1 + growth_dict.get(gu, 0)) ** m for gu, m in zip(gu_series, months_ahead)])

# --- A. 보정 없음 ---
no_correction_rmse = calc_rmse(stack_val_pred, y_val_true)

# --- B. 구별 평균 트렌드 (기존 전략 08) ---
mean_correction = make_gu_correction(gu_growth, oot_val['Gu'], months_ahead_val)
mean_rmse = calc_rmse(stack_val_pred * mean_correction, y_val_true)

# --- C. 구별 중앙값 트렌드 ---
gu_growth_median = {}
for gu in oot_train['Gu'].unique():
    monthly = oot_train[oot_train['Gu'] == gu].groupby('Transaction_YearMonth')['Target'].median()
    gu_growth_median[gu] = monthly.pct_change().dropna().mean()

median_correction = make_gu_correction(gu_growth_median, oot_val['Gu'], months_ahead_val)
median_rmse = calc_rmse(stack_val_pred * median_correction, y_val_true)

# --- D. 평당 가격 트렌드 ---
gu_growth_per_area = {}
for gu in oot_train['Gu'].unique():
    sub = oot_train[oot_train['Gu'] == gu].copy()
    sub['price_per_area'] = sub['Target'] / sub['Exclusive_Area']
    monthly = sub.groupby('Transaction_YearMonth')['price_per_area'].mean()
    gu_growth_per_area[gu] = monthly.pct_change().dropna().mean()

per_area_correction = make_gu_correction(gu_growth_per_area, oot_val['Gu'], months_ahead_val)
per_area_rmse = calc_rmse(stack_val_pred * per_area_correction, y_val_true)

# --- E. Cap 보정 (월성장률 상한 2%) ---
gu_growth_capped = {gu: min(max(g, -0.02), 0.02) for gu, g in gu_growth.items()}
capped_correction = make_gu_correction(gu_growth_capped, oot_val['Gu'], months_ahead_val)
capped_rmse = calc_rmse(stack_val_pred * capped_correction, y_val_true)

# --- F. Cap 보정 (월성장률 상한 1%) ---
gu_growth_capped1 = {gu: min(max(g, -0.01), 0.01) for gu, g in gu_growth.items()}
capped1_correction = make_gu_correction(gu_growth_capped1, oot_val['Gu'], months_ahead_val)
capped1_rmse = calc_rmse(stack_val_pred * capped1_correction, y_val_true)

# --- G. 중앙값 + Cap 2% ---
gu_growth_median_capped = {gu: min(max(g, -0.02), 0.02) for gu, g in gu_growth_median.items()}
median_capped_correction = make_gu_correction(gu_growth_median_capped, oot_val['Gu'], months_ahead_val)
median_capped_rmse = calc_rmse(stack_val_pred * median_capped_correction, y_val_true)

# === 결과 출력 ===
print("\n" + "=" * 60)
print("OOT 보정 방식 비교")
print("=" * 60)
print(f"  (참고) OOF RMSE           : {oof_rmse:,.0f}")
print(f"  {'─' * 45}")

results = [
    ("A. 보정 없음", no_correction_rmse),
    ("B. 구별 평균 트렌드 (전략08)", mean_rmse),
    ("C. 구별 중앙값 트렌드", median_rmse),
    ("D. 평당가격 트렌드", per_area_rmse),
    ("E. 평균 + Cap 2%", capped_rmse),
    ("F. 평균 + Cap 1%", capped1_rmse),
    ("G. 중앙값 + Cap 2%", median_capped_rmse),
]

results.sort(key=lambda x: x[1])
best_rmse = results[0][1]

for name, rmse in results:
    diff = rmse - no_correction_rmse
    marker = " ★" if rmse == best_rmse else ""
    print(f"  {name:30s}: {rmse:,.0f}  ({diff:+,.0f}){marker}")

print(f"\n  ★ = OOT 기준 최선")

# 성장률 비교
print(f"\n{'─' * 60}")
print("구별 월성장률 비교")
print(f"{'─' * 60}")
print(f"  {'구':15s} {'평균':>8s} {'중앙값':>8s} {'평당가':>8s} {'Cap2%':>8s}")
for gu in sorted(gu_growth.keys()):
    print(f"  {gu:15s} {gu_growth[gu]*100:+7.2f}% {gu_growth_median.get(gu,0)*100:+7.2f}% {gu_growth_per_area.get(gu,0)*100:+7.2f}% {gu_growth_capped.get(gu,0)*100:+7.2f}%")
