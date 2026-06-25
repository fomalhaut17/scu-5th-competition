"""
[L4-35] Quantile Regression 앙상블
──────────────────────────
축약명  : QUANTILE ENS
주요 전략: RMSE 모델 + Quantile(10/25/50/75/90) 모델 → 다양성 극대화
차별점  : loss 함수 자체가 다른 모델로 새로운 다양성 축 확보
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

train_orig, test_orig, sample_sub = load_data()
y_true_orig = train_orig['Target'].values
n_orig = len(train_orig)
area_train_orig = train_orig['Exclusive_Area'].values
area_test = test_orig['Exclusive_Area'].values

# === 구별 트렌드 보정 ===
last_train_ym = train_orig['Transaction_YearMonth'].max()
last_train_seq = (last_train_ym // 100 - 2024) * 12 + last_train_ym % 100

gu_growth = {}
for gu in train_orig['Gu'].unique():
    monthly = train_orig[train_orig['Gu'] == gu].groupby('Transaction_YearMonth')['Target'].mean()
    gu_growth[gu] = monthly.pct_change().dropna().mean()

test_seq = (test_orig['Transaction_YearMonth'] // 100 - 2024) * 12 + test_orig['Transaction_YearMonth'] % 100
months_ahead = test_seq.values - last_train_seq
trend_correction = np.array([(1 + gu_growth[gu]) ** m for gu, m in zip(test_orig['Gu'], months_ahead)])

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

def prepare_data(train_df, test_df):
    train_p = add_feature_engineering(base_preprocess(train_df))
    test_p = add_feature_engineering(base_preprocess(test_df))
    train_cb, test_cb = encode_categoricals(train_p, test_p, as_category=False)
    train_lgb, test_lgb = encode_categoricals(train_p, test_p, as_category=True)
    return train_cb, test_cb, train_lgb, test_lgb

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
train_p = add_feature_engineering(base_preprocess(train_orig))
test_p = add_feature_engineering(base_preprocess(test_orig))

train_cb, test_cb = encode_categoricals(train_p, test_p, as_category=False)
X_cb = train_cb.drop(columns=['Target'])
X_test_cb = test_cb
cat_indices = [X_cb.columns.get_loc(c) for c in CAT_FEATURES]

train_lgb, test_lgb = encode_categoricals(train_p, test_p, as_category=True)
X_lgb = train_lgb.drop(columns=['Target'])
X_test_lgb = test_lgb

y_raw = y_true_orig.astype(float)

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# ========================================
# Part 1: 기존 RMSE 8모델 (전략 28 baseline, PL 없이)
# ========================================
print("=" * 60)
print("[Part 1] 기존 RMSE 8모델 (기존 + 평당가)")
print("=" * 60)

MODELS = ['cb_log', 'cb_raw', 'lgb_log', 'lgb_raw']
y_log = np.log1p(y_true_orig)
up = y_true_orig / area_train_orig
y_up_log = np.log1p(up)
y_up_raw = up.astype(float)

oof_rmse = {}
tpred_rmse = {}

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_cb)):
    print(f"  Fold {fold+1}/{N_SPLITS}")

    # CB log
    m = CatBoostRegressor(loss_function='RMSE', random_seed=42, verbose=0,
                          iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
    m.fit(X_cb.iloc[tr_idx], y_log[tr_idx],
          eval_set=(X_cb.iloc[va_idx], y_log[va_idx]), cat_features=cat_indices)
    oof_rmse.setdefault('cb_log', np.zeros(n_orig))[va_idx] = np.expm1(m.predict(X_cb.iloc[va_idx]))
    tpred_rmse['cb_log'] = tpred_rmse.get('cb_log', np.zeros(len(X_test_cb))) + np.expm1(m.predict(X_test_cb)) / N_SPLITS

    # CB raw
    m = CatBoostRegressor(loss_function='RMSE', random_seed=42, verbose=0,
                          iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
    m.fit(X_cb.iloc[tr_idx], y_raw[tr_idx],
          eval_set=(X_cb.iloc[va_idx], y_raw[va_idx]), cat_features=cat_indices)
    oof_rmse.setdefault('cb_raw', np.zeros(n_orig))[va_idx] = m.predict(X_cb.iloc[va_idx])
    tpred_rmse['cb_raw'] = tpred_rmse.get('cb_raw', np.zeros(len(X_test_cb))) + m.predict(X_test_cb) / N_SPLITS

    # LGB log
    m = lgb.LGBMRegressor(objective='regression', metric='rmse',
                          verbose=-1, random_state=42, n_estimators=3000, **LGB_PARAMS)
    m.fit(X_lgb.iloc[tr_idx], y_log[tr_idx],
          eval_set=[(X_lgb.iloc[va_idx], y_log[va_idx])],
          callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])
    oof_rmse.setdefault('lgb_log', np.zeros(n_orig))[va_idx] = np.expm1(m.predict(X_lgb.iloc[va_idx]))
    tpred_rmse['lgb_log'] = tpred_rmse.get('lgb_log', np.zeros(len(X_test_cb))) + np.expm1(m.predict(X_test_lgb)) / N_SPLITS

    # LGB raw
    m = lgb.LGBMRegressor(objective='regression', metric='rmse',
                          verbose=-1, random_state=42, n_estimators=3000, **LGB_PARAMS)
    m.fit(X_lgb.iloc[tr_idx], y_raw[tr_idx],
          eval_set=[(X_lgb.iloc[va_idx], y_raw[va_idx])],
          callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])
    oof_rmse.setdefault('lgb_raw', np.zeros(n_orig))[va_idx] = m.predict(X_lgb.iloc[va_idx])
    tpred_rmse['lgb_raw'] = tpred_rmse.get('lgb_raw', np.zeros(len(X_test_cb))) + m.predict(X_test_lgb) / N_SPLITS

    # 평당가 CB log
    m = CatBoostRegressor(loss_function='RMSE', random_seed=42, verbose=0,
                          iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
    m.fit(X_cb.iloc[tr_idx], y_up_log[tr_idx],
          eval_set=(X_cb.iloc[va_idx], y_up_log[va_idx]), cat_features=cat_indices)
    oof_rmse.setdefault('up_cb_log', np.zeros(n_orig))[va_idx] = np.expm1(m.predict(X_cb.iloc[va_idx])) * area_train_orig[va_idx]
    tpred_rmse['up_cb_log'] = tpred_rmse.get('up_cb_log', np.zeros(len(X_test_cb))) + np.expm1(m.predict(X_test_cb)) * area_test / N_SPLITS

    # 평당가 CB raw
    m = CatBoostRegressor(loss_function='RMSE', random_seed=42, verbose=0,
                          iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
    m.fit(X_cb.iloc[tr_idx], y_up_raw[tr_idx],
          eval_set=(X_cb.iloc[va_idx], y_up_raw[va_idx]), cat_features=cat_indices)
    oof_rmse.setdefault('up_cb_raw', np.zeros(n_orig))[va_idx] = m.predict(X_cb.iloc[va_idx]) * area_train_orig[va_idx]
    tpred_rmse['up_cb_raw'] = tpred_rmse.get('up_cb_raw', np.zeros(len(X_test_cb))) + m.predict(X_test_cb) * area_test / N_SPLITS

    # 평당가 LGB log
    m = lgb.LGBMRegressor(objective='regression', metric='rmse',
                          verbose=-1, random_state=42, n_estimators=3000, **LGB_PARAMS)
    m.fit(X_lgb.iloc[tr_idx], y_up_log[tr_idx],
          eval_set=[(X_lgb.iloc[va_idx], y_up_log[va_idx])],
          callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])
    oof_rmse.setdefault('up_lgb_log', np.zeros(n_orig))[va_idx] = np.expm1(m.predict(X_lgb.iloc[va_idx])) * area_train_orig[va_idx]
    tpred_rmse['up_lgb_log'] = tpred_rmse.get('up_lgb_log', np.zeros(len(X_test_cb))) + np.expm1(m.predict(X_test_lgb)) * area_test / N_SPLITS

    # 평당가 LGB raw
    m = lgb.LGBMRegressor(objective='regression', metric='rmse',
                          verbose=-1, random_state=42, n_estimators=3000, **LGB_PARAMS)
    m.fit(X_lgb.iloc[tr_idx], y_up_raw[tr_idx],
          eval_set=[(X_lgb.iloc[va_idx], y_up_raw[va_idx])],
          callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])
    oof_rmse.setdefault('up_lgb_raw', np.zeros(n_orig))[va_idx] = m.predict(X_lgb.iloc[va_idx]) * area_train_orig[va_idx]
    tpred_rmse['up_lgb_raw'] = tpred_rmse.get('up_lgb_raw', np.zeros(len(X_test_cb))) + m.predict(X_test_lgb) * area_test / N_SPLITS

ALL_8 = ['cb_log', 'cb_raw', 'lgb_log', 'lgb_raw', 'up_cb_log', 'up_cb_raw', 'up_lgb_log', 'up_lgb_raw']

# ========================================
# Part 2: Quantile 모델 (CB/LGB × 5분위수 × raw target)
# ========================================
print(f"\n{'=' * 60}")
print("[Part 2] Quantile 모델")
print("=" * 60)

QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]
oof_q = {}
tpred_q = {}
q_names = []

for q in QUANTILES:
    print(f"\n  --- Quantile {q:.2f} ---")

    cb_name = f'cb_q{int(q*100)}'
    lgb_name = f'lgb_q{int(q*100)}'
    q_names.extend([cb_name, lgb_name])

    oof_q[cb_name] = np.zeros(n_orig)
    oof_q[lgb_name] = np.zeros(n_orig)
    tpred_q[cb_name] = np.zeros(len(X_test_cb))
    tpred_q[lgb_name] = np.zeros(len(X_test_cb))

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_cb)):
        # CB quantile
        m = CatBoostRegressor(loss_function=f'Quantile:alpha={q}', random_seed=42, verbose=0,
                              iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
        m.fit(X_cb.iloc[tr_idx], y_raw[tr_idx],
              eval_set=(X_cb.iloc[va_idx], y_raw[va_idx]), cat_features=cat_indices)
        oof_q[cb_name][va_idx] = m.predict(X_cb.iloc[va_idx])
        tpred_q[cb_name] += m.predict(X_test_cb) / N_SPLITS

        # LGB quantile
        m = lgb.LGBMRegressor(objective='quantile', alpha=q, metric='quantile',
                              verbose=-1, random_state=42, n_estimators=3000, **LGB_PARAMS)
        m.fit(X_lgb.iloc[tr_idx], y_raw[tr_idx],
              eval_set=[(X_lgb.iloc[va_idx], y_raw[va_idx])],
              callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])
        oof_q[lgb_name][va_idx] = m.predict(X_lgb.iloc[va_idx])
        tpred_q[lgb_name] += m.predict(X_test_lgb) / N_SPLITS

    rmse_cb = np.sqrt(np.mean((oof_q[cb_name] - y_true_orig) ** 2))
    rmse_lgb = np.sqrt(np.mean((oof_q[lgb_name] - y_true_orig) ** 2))
    print(f"    CB  q{int(q*100):02d} OOF RMSE: {rmse_cb:,.0f}")
    print(f"    LGB q{int(q*100):02d} OOF RMSE: {rmse_lgb:,.0f}")

# ========================================
# Part 3: 블렌딩 비교
# ========================================
print(f"\n{'=' * 60}")
print("블렌딩 비교")
print("=" * 60)

# 기존 8모델만
base_oof = np.mean([oof_rmse[k] for k in ALL_8], axis=0)
rmse_base = np.sqrt(np.mean((base_oof - y_true_orig) ** 2))
print(f"  기존 8모델 단순평균: {rmse_base:,.0f}")

# Quantile만
q_oof = np.mean([oof_q[k] for k in q_names], axis=0)
rmse_q = np.sqrt(np.mean((q_oof - y_true_orig) ** 2))
print(f"  Quantile 10모델 단순평균: {rmse_q:,.0f}")

# 8 + Quantile = 18모델
all_names = ALL_8 + q_names
all_oof_list = [oof_rmse[k] for k in ALL_8] + [oof_q[k] for k in q_names]
all_test_list = [tpred_rmse[k] for k in ALL_8] + [tpred_q[k] for k in q_names]

avg_18_oof = np.mean(all_oof_list, axis=0)
rmse_18 = np.sqrt(np.mean((avg_18_oof - y_true_orig) ** 2))
print(f"  18모델 단순평균: {rmse_18:,.0f}")

# Ridge 스태킹
stack_tr = np.column_stack(all_oof_list)
stack_te = np.column_stack(all_test_list)

best_ridge_rmse = float('inf')
best_ridge_test = None
best_ridge_alpha = None

for alpha in [1.0, 5.0, 10.0, 50.0, 100.0]:
    s_oof = np.zeros(n_orig)
    s_test = np.zeros(len(X_test_cb))
    for fold, (tr_idx, va_idx) in enumerate(kf.split(stack_tr)):
        meta = Ridge(alpha=alpha)
        meta.fit(stack_tr[tr_idx], y_true_orig[tr_idx])
        s_oof[va_idx] = meta.predict(stack_tr[va_idx])
        s_test += meta.predict(stack_te) / N_SPLITS
    rmse = np.sqrt(np.mean((s_oof - y_true_orig) ** 2))
    print(f"  18모델 Ridge(α={alpha:5.1f}): {rmse:,.0f}")
    if rmse < best_ridge_rmse:
        best_ridge_rmse = rmse
        best_ridge_test = s_test.copy()
        best_ridge_alpha = alpha

# 8 + 선별 Quantile (50분위만, 또는 25+75만 등)
print(f"\n  --- 선별 Quantile 추가 ---")
q_subsets = {
    'q50만': ['cb_q50', 'lgb_q50'],
    'q25+75': ['cb_q25', 'lgb_q25', 'cb_q75', 'lgb_q75'],
    'q10+90': ['cb_q10', 'lgb_q10', 'cb_q90', 'lgb_q90'],
    'q25+50+75': ['cb_q25', 'lgb_q25', 'cb_q50', 'lgb_q50', 'cb_q75', 'lgb_q75'],
}

for label, q_sel in q_subsets.items():
    sel_oof = [oof_rmse[k] for k in ALL_8] + [oof_q[k] for k in q_sel]
    sel_test = [tpred_rmse[k] for k in ALL_8] + [tpred_q[k] for k in q_sel]
    st = np.column_stack(sel_oof)
    ste = np.column_stack(sel_test)

    best_r = float('inf')
    for alpha in [10.0, 50.0]:
        s_oof = np.zeros(n_orig)
        for fold, (tr_idx, va_idx) in enumerate(kf.split(st)):
            meta = Ridge(alpha=alpha)
            meta.fit(st[tr_idx], y_true_orig[tr_idx])
            s_oof[va_idx] = meta.predict(st[va_idx])
        r = np.sqrt(np.mean((s_oof - y_true_orig) ** 2))
        best_r = min(best_r, r)
    n_total = len(sel_oof)
    print(f"    8+{label:12s} ({n_total:2d}모델): {best_r:,.0f}")

# ========================================
# 최종 비교
# ========================================
print(f"\n{'=' * 60}")
print("최종 비교")
print("=" * 60)
print(f"  기존 8모델 단순평균      : {rmse_base:,.0f}")
print(f"  18모델 단순평균          : {rmse_18:,.0f}")
print(f"  18모델 Ridge(α={best_ridge_alpha})    : {best_ridge_rmse:,.0f}")
print(f"  ─────────────────────────────")
print(f"  전략 28 (PL2+8모델 Ridge): OOF 2,196 / Public 2,096.8")
print(f"  (위 결과는 PL 없는 비교. PL 추가 시 추가 개선 기대)")

# 제출 파일 (최선)
final_pred = best_ridge_test * trend_correction

sub = sample_sub.copy()
sub['Target'] = final_pred
sub.to_csv(os.path.join(os.path.dirname(__file__), '..', '..', 'submission_l4_35_quantile.csv'), index=False)
print(f"\n제출 파일 생성: submission_l4_35_quantile.csv")

record_result('L4', 35, 'QUANTILE ENS',
              f'RMSE 8모델 + Quantile 10모델 Ridge(α={best_ridge_alpha}) + GTR',
              best_ridge_rmse, 'tested')
