"""
[L4-51] 고오차 사후보정 + DGP 역공학
──────────────────────────
축약명  : RESID+DGP
주요 전략: 전략47 파이프라인(12모델) 학습 후, 2가지 메타 접근 비교
  A. 고오차 사후보정: OOF 잔차 패턴 → 구간별 보정 계수
  B. DGP 역공학: 합성 데이터의 곱셈 구조를 피처로 활용
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb
from utils import load_data, record_result
import warnings
warnings.filterwarnings('ignore')

N_SPLITS = 5
CAT_FEATURES = ['Gu', 'Dong']
MODELS = ['cb_log', 'cb_raw', 'lgb_log', 'lgb_raw']

train_orig, test_orig, sample_sub = load_data()
y_true_orig = train_orig['Target'].values
n_orig = len(train_orig)
area_train_orig = train_orig['Exclusive_Area'].values
area_test = test_orig['Exclusive_Area'].values

# === GTR ===
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
    'learning_rate': 0.010118898857677389, 'depth': 3,
    'l2_leaf_reg': 4.944272225334265, 'bagging_temperature': 1.4823308606638113,
    'random_strength': 0.4685604025205004, 'min_data_in_leaf': 46,
}
LGB_PARAMS = {
    'learning_rate': 0.022992006545037823, 'num_leaves': 110, 'max_depth': 3,
    'min_child_samples': 27, 'subsample': 0.9312452053625488,
    'colsample_bytree': 0.8234901310320267, 'reg_alpha': 0.012423757285817386,
    'reg_lambda': 0.04673443002441543,
}
LGB_ET_PARAMS = {
    'extra_trees': True, 'max_depth': 3, 'num_leaves': 31,
    'feature_fraction': 0.6, 'bagging_fraction': 0.7, 'bagging_freq': 1,
    'min_child_samples': 30, 'learning_rate': 0.02,
}

# ========================================
# Stage 1: PL2
# ========================================
print("=" * 60)
print("[Stage 1] PL2")
print("=" * 60)

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
train_cb, test_cb, train_lgb, test_lgb = prepare_data(train_orig, test_orig)

X_cb = train_cb.drop(columns=['Target'])
X_test_cb = test_cb
cat_idx = [X_cb.columns.get_loc(c) for c in CAT_FEATURES]
X_lgb = train_lgb.drop(columns=['Target'])
X_test_lgb = test_lgb

oof_s1 = {k: np.zeros(n_orig) for k in MODELS}
tpred_s1 = {k: np.zeros(len(test_orig)) for k in MODELS}
fold_preds_s1 = {k: [] for k in MODELS}

for fold, (tr, va) in enumerate(kf.split(X_cb)):
    print(f"  [S1] Fold {fold+1}/5")
    y_log = np.log1p(train_cb['Target']); y_raw = train_cb['Target'].values

    m = CatBoostRegressor(loss_function='RMSE', random_seed=42, verbose=0, iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
    m.fit(X_cb.iloc[tr], y_log.iloc[tr], eval_set=(X_cb.iloc[va], y_log.iloc[va]), cat_features=cat_idx)
    oof_s1['cb_log'][va] = np.expm1(m.predict(X_cb.iloc[va]))
    fp = np.expm1(m.predict(X_test_cb)); tpred_s1['cb_log'] += fp/5; fold_preds_s1['cb_log'].append(fp)

    m = CatBoostRegressor(loss_function='RMSE', random_seed=42, verbose=0, iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
    m.fit(X_cb.iloc[tr], y_raw[tr], eval_set=(X_cb.iloc[va], y_raw[va]), cat_features=cat_idx)
    oof_s1['cb_raw'][va] = m.predict(X_cb.iloc[va])
    fp = m.predict(X_test_cb); tpred_s1['cb_raw'] += fp/5; fold_preds_s1['cb_raw'].append(fp)

    m = lgb.LGBMRegressor(objective='regression', metric='rmse', verbose=-1, random_state=42, n_estimators=3000, **LGB_PARAMS)
    m.fit(X_lgb.iloc[tr], y_log.iloc[tr], eval_set=[(X_lgb.iloc[va], y_log.iloc[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_s1['lgb_log'][va] = np.expm1(m.predict(X_lgb.iloc[va]))
    fp = np.expm1(m.predict(X_test_lgb)); tpred_s1['lgb_log'] += fp/5; fold_preds_s1['lgb_log'].append(fp)

    m = lgb.LGBMRegressor(objective='regression', metric='rmse', verbose=-1, random_state=42, n_estimators=3000, **LGB_PARAMS)
    m.fit(X_lgb.iloc[tr], y_raw[tr], eval_set=[(X_lgb.iloc[va], y_raw[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_s1['lgb_raw'][va] = m.predict(X_lgb.iloc[va])
    fp = m.predict(X_test_lgb); tpred_s1['lgb_raw'] += fp/5; fold_preds_s1['lgb_raw'].append(fp)

pseudo_labels = np.mean([tpred_s1[k] for k in MODELS], axis=0)
model_means = np.array([tpred_s1[k] for k in MODELS])
disagree = np.std(model_means, axis=0) / np.mean(model_means, axis=0)
fcvs = [np.std(np.array(fold_preds_s1[k]), axis=0) / np.mean(np.array(fold_preds_s1[k]), axis=0) for k in MODELS]
confidence = 1 - (disagree + np.mean(fcvs, axis=0)) / 2
mask_pl = confidence >= np.percentile(confidence, 50)

test_selected = test_orig[mask_pl].copy()
test_selected['Target'] = pseudo_labels[mask_pl]
train_aug = pd.concat([train_orig, test_selected], ignore_index=True)
area_train_aug = train_aug['Exclusive_Area'].values

# ========================================
# Stage 2: 12 베이스 모델
# ========================================
print(f"\n{'=' * 60}")
print("[Stage 2] 12 베이스 모델 학습")
print("=" * 60)

kf2 = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
train_cb2, test_cb2, train_lgb2, test_lgb2 = prepare_data(train_aug, test_orig)

X_cb2 = train_cb2.drop(columns=['Target'])
X_test_cb2 = test_cb2
cat_idx2 = [X_cb2.columns.get_loc(c) for c in CAT_FEATURES]
X_lgb2 = train_lgb2.drop(columns=['Target'])
X_test_lgb2 = test_lgb2

y_log_aug = np.log1p(train_aug['Target'])
y_raw_aug = train_aug['Target'].values
y_up_log = np.log1p(train_aug['Target'].values / area_train_aug)
y_up_raw = (train_aug['Target'].values / area_train_aug).astype(float)

all_oof = {}
all_tpred = {}

for fold, (tr, va) in enumerate(kf2.split(X_cb2)):
    print(f"  Fold {fold+1}/5")

    m = CatBoostRegressor(loss_function='RMSE', random_seed=42, verbose=0, iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
    m.fit(X_cb2.iloc[tr], y_log_aug.iloc[tr], eval_set=(X_cb2.iloc[va], y_log_aug.iloc[va]), cat_features=cat_idx2)
    all_oof.setdefault('cb_log', np.zeros(len(X_cb2)))[va] = np.expm1(m.predict(X_cb2.iloc[va]))
    all_tpred['cb_log'] = all_tpred.get('cb_log', np.zeros(len(X_test_cb2))) + np.expm1(m.predict(X_test_cb2)) / 5

    m = CatBoostRegressor(loss_function='RMSE', random_seed=42, verbose=0, iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
    m.fit(X_cb2.iloc[tr], y_raw_aug[tr], eval_set=(X_cb2.iloc[va], y_raw_aug[va]), cat_features=cat_idx2)
    all_oof.setdefault('cb_raw', np.zeros(len(X_cb2)))[va] = m.predict(X_cb2.iloc[va])
    all_tpred['cb_raw'] = all_tpred.get('cb_raw', np.zeros(len(X_test_cb2))) + m.predict(X_test_cb2) / 5

    m = lgb.LGBMRegressor(objective='regression', metric='rmse', verbose=-1, random_state=42, n_estimators=3000, **LGB_PARAMS)
    m.fit(X_lgb2.iloc[tr], y_log_aug.iloc[tr], eval_set=[(X_lgb2.iloc[va], y_log_aug.iloc[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
    all_oof.setdefault('lgb_log', np.zeros(len(X_cb2)))[va] = np.expm1(m.predict(X_lgb2.iloc[va]))
    all_tpred['lgb_log'] = all_tpred.get('lgb_log', np.zeros(len(X_test_lgb2))) + np.expm1(m.predict(X_test_lgb2)) / 5

    m = lgb.LGBMRegressor(objective='regression', metric='rmse', verbose=-1, random_state=42, n_estimators=3000, **LGB_PARAMS)
    m.fit(X_lgb2.iloc[tr], y_raw_aug[tr], eval_set=[(X_lgb2.iloc[va], y_raw_aug[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
    all_oof.setdefault('lgb_raw', np.zeros(len(X_cb2)))[va] = m.predict(X_lgb2.iloc[va])
    all_tpred['lgb_raw'] = all_tpred.get('lgb_raw', np.zeros(len(X_test_lgb2))) + m.predict(X_test_lgb2) / 5

    m = CatBoostRegressor(loss_function='RMSE', random_seed=42, verbose=0, iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
    m.fit(X_cb2.iloc[tr], y_up_log[tr], eval_set=(X_cb2.iloc[va], y_up_log[va]), cat_features=cat_idx2)
    all_oof.setdefault('cb_up_log', np.zeros(len(X_cb2)))[va] = np.expm1(m.predict(X_cb2.iloc[va]))
    all_tpred['cb_up_log'] = all_tpred.get('cb_up_log', np.zeros(len(X_test_cb2))) + np.expm1(m.predict(X_test_cb2)) / 5

    m = CatBoostRegressor(loss_function='RMSE', random_seed=42, verbose=0, iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
    m.fit(X_cb2.iloc[tr], y_up_raw[tr], eval_set=(X_cb2.iloc[va], y_up_raw[va]), cat_features=cat_idx2)
    all_oof.setdefault('cb_up_raw', np.zeros(len(X_cb2)))[va] = m.predict(X_cb2.iloc[va])
    all_tpred['cb_up_raw'] = all_tpred.get('cb_up_raw', np.zeros(len(X_test_cb2))) + m.predict(X_test_cb2) / 5

    m = lgb.LGBMRegressor(objective='regression', metric='rmse', verbose=-1, random_state=42, n_estimators=3000, **LGB_PARAMS)
    m.fit(X_lgb2.iloc[tr], y_up_log[tr], eval_set=[(X_lgb2.iloc[va], y_up_log[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
    all_oof.setdefault('lgb_up_log', np.zeros(len(X_cb2)))[va] = np.expm1(m.predict(X_lgb2.iloc[va]))
    all_tpred['lgb_up_log'] = all_tpred.get('lgb_up_log', np.zeros(len(X_test_lgb2))) + np.expm1(m.predict(X_test_lgb2)) / 5

    m = lgb.LGBMRegressor(objective='regression', metric='rmse', verbose=-1, random_state=42, n_estimators=3000, **LGB_PARAMS)
    m.fit(X_lgb2.iloc[tr], y_up_raw[tr], eval_set=[(X_lgb2.iloc[va], y_up_raw[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
    all_oof.setdefault('lgb_up_raw', np.zeros(len(X_cb2)))[va] = m.predict(X_lgb2.iloc[va])
    all_tpred['lgb_up_raw'] = all_tpred.get('lgb_up_raw', np.zeros(len(X_test_lgb2))) + m.predict(X_test_lgb2) / 5

    scaler = StandardScaler()
    X_sc_tr = scaler.fit_transform(X_cb2.iloc[tr])
    X_sc_va = scaler.transform(X_cb2.iloc[va])
    X_sc_te = scaler.transform(X_test_cb2)

    m = ExtraTreesRegressor(n_estimators=500, max_depth=12, min_samples_leaf=10, random_state=42, n_jobs=-1)
    m.fit(X_sc_tr, y_log_aug.iloc[tr])
    all_oof.setdefault('et_log', np.zeros(len(X_cb2)))[va] = np.expm1(m.predict(X_sc_va))
    all_tpred['et_log'] = all_tpred.get('et_log', np.zeros(len(X_test_cb2))) + np.expm1(m.predict(X_sc_te)) / 5

    m = ExtraTreesRegressor(n_estimators=500, max_depth=12, min_samples_leaf=10, random_state=42, n_jobs=-1)
    m.fit(X_sc_tr, y_raw_aug[tr])
    all_oof.setdefault('et_raw', np.zeros(len(X_cb2)))[va] = m.predict(X_sc_va)
    all_tpred['et_raw'] = all_tpred.get('et_raw', np.zeros(len(X_test_cb2))) + m.predict(X_sc_te) / 5

    m = lgb.LGBMRegressor(objective='regression', metric='rmse', verbose=-1, random_state=42, n_estimators=3000, **LGB_ET_PARAMS)
    m.fit(X_lgb2.iloc[tr], y_log_aug.iloc[tr], eval_set=[(X_lgb2.iloc[va], y_log_aug.iloc[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
    all_oof.setdefault('lgbet_log', np.zeros(len(X_cb2)))[va] = np.expm1(m.predict(X_lgb2.iloc[va]))
    all_tpred['lgbet_log'] = all_tpred.get('lgbet_log', np.zeros(len(X_test_lgb2))) + np.expm1(m.predict(X_test_lgb2)) / 5

    m = lgb.LGBMRegressor(objective='regression', metric='rmse', verbose=-1, random_state=42, n_estimators=3000, **LGB_ET_PARAMS)
    m.fit(X_lgb2.iloc[tr], y_raw_aug[tr], eval_set=[(X_lgb2.iloc[va], y_raw_aug[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
    all_oof.setdefault('lgbet_raw', np.zeros(len(X_cb2)))[va] = m.predict(X_lgb2.iloc[va])
    all_tpred['lgbet_raw'] = all_tpred.get('lgbet_raw', np.zeros(len(X_test_lgb2))) + m.predict(X_test_lgb2) / 5

for k in ['cb_up_log', 'cb_up_raw', 'lgb_up_log', 'lgb_up_raw']:
    all_oof[k] = all_oof[k][:n_orig] * area_train_orig
    all_tpred[k] = all_tpred[k] * area_test
for k in [kk for kk in all_oof if kk not in ['cb_up_log', 'cb_up_raw', 'lgb_up_log', 'lgb_up_raw']]:
    all_oof[k] = all_oof[k][:n_orig]

base_12 = ['cb_log', 'cb_raw', 'lgb_log', 'lgb_raw',
           'cb_up_log', 'cb_up_raw', 'lgb_up_log', 'lgb_up_raw',
           'et_log', 'et_raw', 'lgbet_log', 'lgbet_raw']

stack_train = np.column_stack([all_oof[k] for k in base_12])
stack_test = np.column_stack([all_tpred[k] for k in base_12])

# 전역 Ridge 기준선
global_oof = np.zeros(n_orig)
global_test = np.zeros(len(test_orig))
for fold, (tr, va) in enumerate(kf.split(stack_train)):
    meta = Ridge(alpha=1.0)
    meta.fit(stack_train[tr], y_true_orig[tr])
    global_oof[va] = meta.predict(stack_train[va])
    global_test += meta.predict(stack_test) / N_SPLITS
global_rmse = np.sqrt(np.mean((global_oof - y_true_orig)**2))
print(f"\n  전역 Ridge OOF: {global_rmse:,.0f}")

# ========================================
# 파트 A: 고오차 영역 분석 + 사후보정
# ========================================
print(f"\n{'=' * 60}")
print("[파트 A] 고오차 영역 분석")
print("=" * 60)

residuals = global_oof - y_true_orig
abs_residuals = np.abs(residuals)

# 구별 잔차
print("\n--- 구별 잔차 분석 ---")
print(f"  {'구':15s} {'건수':>5s} {'RMSE':>8s} {'편향':>8s} {'|편향|/RMSE':>10s}")
print("  " + "-" * 50)
for gu in sorted(train_orig['Gu'].unique()):
    mask = train_orig['Gu'].values == gu
    n_gu = mask.sum()
    rmse_gu = np.sqrt(np.mean(residuals[mask]**2))
    bias_gu = np.mean(residuals[mask])
    print(f"  {gu:15s} {n_gu:5d} {rmse_gu:>8,.0f} {bias_gu:>+8,.0f} {abs(bias_gu)/rmse_gu:>10.2f}")

# 면적 구간별
print("\n--- 면적 구간별 잔차 ---")
area_bins = [(0, 60), (60, 85), (85, 120), (120, 300)]
for lo, hi in area_bins:
    mask = (area_train_orig >= lo) & (area_train_orig < hi)
    if mask.sum() == 0: continue
    rmse_seg = np.sqrt(np.mean(residuals[mask]**2))
    bias_seg = np.mean(residuals[mask])
    print(f"  {lo:3d}~{hi:3d}㎡ ({mask.sum():4d}건): RMSE {rmse_seg:>6,.0f}, 편향 {bias_seg:>+6,.0f}")

# 가격 구간별
print("\n--- 가격 구간별 잔차 ---")
price_bins = [(0, 20000), (20000, 30000), (30000, 40000), (40000, 50000), (50000, 70000)]
for lo, hi in price_bins:
    mask = (y_true_orig >= lo) & (y_true_orig < hi)
    if mask.sum() == 0: continue
    rmse_seg = np.sqrt(np.mean(residuals[mask]**2))
    bias_seg = np.mean(residuals[mask])
    print(f"  {lo//10000:1d}~{hi//10000:1d}만 ({mask.sum():4d}건): RMSE {rmse_seg:>6,.0f}, 편향 {bias_seg:>+6,.0f}")

# 오차 상위 10% 분석
top10_mask = abs_residuals >= np.percentile(abs_residuals, 90)
print(f"\n--- 오차 상위 10% ({top10_mask.sum()}건) 특성 ---")
top10_df = train_orig[top10_mask]
all_df = train_orig
for col in ['Gu', 'Exclusive_Area', 'Floor']:
    if col == 'Gu':
        for gu in sorted(all_df['Gu'].unique()):
            ratio_top = (top10_df['Gu'] == gu).mean()
            ratio_all = (all_df['Gu'] == gu).mean()
            if ratio_top > ratio_all * 1.3:
                print(f"  {gu}: 전체 {ratio_all:.1%} → 오차상위10% {ratio_top:.1%} (과대)")
    elif col == 'Exclusive_Area':
        print(f"  면적: 전체 평균 {all_df[col].mean():.1f} → 오차상위10% {top10_df[col].mean():.1f}")
    elif col == 'Floor':
        print(f"  층수: 전체 평균 {all_df[col].mean():.1f} → 오차상위10% {top10_df[col].mean():.1f}")

# === 사후보정 시도 ===
print(f"\n{'=' * 60}")
print("[파트 A] 사후보정 시도")
print("=" * 60)

best_a_rmse = global_rmse
best_a_test = global_test.copy()
best_a_name = "기준(보정없음)"

# A1. 구별 편향 보정
print("\n--- A1. 구별 편향 보정 ---")
for shrink in [0.3, 0.5, 0.7, 1.0]:
    corrected_oof = global_oof.copy()
    corrected_test = global_test.copy()

    for fold, (tr, va) in enumerate(kf.split(stack_train)):
        for gu in train_orig['Gu'].unique():
            tr_gu_mask = train_orig['Gu'].values[tr] == gu
            va_gu_mask = train_orig['Gu'].values[va] == gu
            te_gu_mask = test_orig['Gu'].values == gu

            if tr_gu_mask.sum() < 5: continue
            bias = np.mean(global_oof[tr][tr_gu_mask] - y_true_orig[tr][tr_gu_mask])
            corrected_oof[va[va_gu_mask]] -= bias * shrink
            # test는 전체 train의 편향 사용
        # test 보정은 fold 밖에서

    # test 보정: 전체 train의 구별 편향
    for gu in train_orig['Gu'].unique():
        gu_mask_tr = train_orig['Gu'].values == gu
        gu_mask_te = test_orig['Gu'].values == gu
        bias = np.mean(global_oof[gu_mask_tr] - y_true_orig[gu_mask_tr])
        corrected_test[gu_mask_te] -= bias * shrink

    r = np.sqrt(np.mean((corrected_oof - y_true_orig)**2))
    diff = r - global_rmse
    marker = ""
    if r < best_a_rmse:
        best_a_rmse = r
        best_a_test = corrected_test.copy()
        best_a_name = f"구별편향 shrink={shrink}"
        marker = " ★"
    print(f"  shrink={shrink:.1f}: OOF {r:,.0f} ({diff:+,.0f}){marker}")

# A2. Residual 모델 (잔차를 피처로 예측)
print("\n--- A2. Residual 모델 (잔차 예측) ---")
resid_features_train = train_orig[['Exclusive_Area', 'Floor', 'Distance_to_Subway', 'Nearby_Parks', 'Brand_Apartment']].copy()
resid_features_train['Distance_to_Subway'] = resid_features_train['Distance_to_Subway'].fillna(resid_features_train['Distance_to_Subway'].median())
resid_features_train['pred'] = global_oof

resid_features_test = test_orig[['Exclusive_Area', 'Floor', 'Distance_to_Subway', 'Nearby_Parks', 'Brand_Apartment']].copy()
resid_features_test['Distance_to_Subway'] = resid_features_test['Distance_to_Subway'].fillna(resid_features_train['Distance_to_Subway'].median())
resid_features_test['pred'] = global_test

# Gu를 label encode
le_gu = LabelEncoder()
resid_features_train['Gu'] = le_gu.fit_transform(train_orig['Gu'])
resid_features_test['Gu'] = le_gu.transform(test_orig['Gu'])

for max_depth in [2, 3, 4]:
    for lr in [0.01, 0.05]:
        corrected_oof = global_oof.copy()
        corrected_test_r = np.zeros(len(test_orig))

        for fold, (tr, va) in enumerate(kf.split(resid_features_train)):
            resid_model = GradientBoostingRegressor(
                n_estimators=200, max_depth=max_depth, learning_rate=lr,
                min_samples_leaf=20, subsample=0.8, random_state=42)
            resid_model.fit(resid_features_train.iloc[tr], residuals[tr])
            corrected_oof[va] -= resid_model.predict(resid_features_train.iloc[va])
            corrected_test_r += resid_model.predict(resid_features_test) / N_SPLITS

        corrected_test_final = global_test - corrected_test_r
        r = np.sqrt(np.mean((corrected_oof - y_true_orig)**2))
        diff = r - global_rmse
        marker = ""
        if r < best_a_rmse:
            best_a_rmse = r
            best_a_test = corrected_test_final.copy()
            best_a_name = f"Residual GBR(d={max_depth},lr={lr})"
            marker = " ★"
        print(f"  GBR depth={max_depth} lr={lr}: OOF {r:,.0f} ({diff:+,.0f}){marker}")

# A3. Sample Weighted Ridge (고오차 영역 가중)
print("\n--- A3. Sample Weighted Ridge ---")
for weight_factor in [1.5, 2.0, 3.0]:
    s_oof = np.zeros(n_orig)
    s_test = np.zeros(len(test_orig))

    for fold, (tr, va) in enumerate(kf.split(stack_train)):
        weights = np.ones(len(tr))
        # 고오차 구 (성동, 용산)에 가중
        for gu in ['Seongdong-gu', 'Yongsan-gu']:
            gu_mask = train_orig['Gu'].values[tr] == gu
            weights[gu_mask] *= weight_factor
        # 대형 면적에 가중
        large_mask = area_train_orig[tr] >= 120
        weights[large_mask] *= weight_factor

        meta = Ridge(alpha=1.0)
        meta.fit(stack_train[tr], y_true_orig[tr], sample_weight=weights)
        s_oof[va] = meta.predict(stack_train[va])
        s_test += meta.predict(stack_test) / N_SPLITS

    r = np.sqrt(np.mean((s_oof - y_true_orig)**2))
    diff = r - global_rmse
    marker = ""
    if r < best_a_rmse:
        best_a_rmse = r
        best_a_test = s_test.copy()
        best_a_name = f"Weighted Ridge(w={weight_factor})"
        marker = " ★"
    print(f"  weight={weight_factor}: OOF {r:,.0f} ({diff:+,.0f}){marker}")

print(f"\n  파트A 최선: {best_a_name} → OOF {best_a_rmse:,.0f}")

# ========================================
# 파트 B: DGP 역공학
# ========================================
print(f"\n{'=' * 60}")
print("[파트 B] DGP 역공학 (곱셈 구조 분석)")
print("=" * 60)

# B1. Log-Log 선형 회귀: log(Price) = a*log(Area) + b*log(Floor) + ...
print("\n--- B1. Log-Log 선형 회귀 ---")
log_features = pd.DataFrame()
log_features['log_area'] = np.log1p(train_orig['Exclusive_Area'])
log_features['log_floor'] = np.log1p(train_orig['Floor'].clip(lower=1))
log_features['log_subway'] = np.log1p(train_orig['Distance_to_Subway'].fillna(train_orig['Distance_to_Subway'].median()))
log_features['log_parks'] = np.log1p(train_orig['Nearby_Parks'])
log_features['brand'] = train_orig['Brand_Apartment']
log_features['age'] = 2026 - train_orig['Year_Built']
log_features['log_age'] = np.log1p(log_features['age'])

# Gu dummy
for gu in train_orig['Gu'].unique():
    log_features[f'gu_{gu}'] = (train_orig['Gu'] == gu).astype(int)

log_target = np.log1p(y_true_orig)

lr = LinearRegression()
lr.fit(log_features, log_target)
log_pred = lr.predict(log_features)
log_rmse = np.sqrt(np.mean((np.expm1(log_pred) - y_true_orig)**2))
r2 = lr.score(log_features, log_target)
print(f"  Log-Log R²={r2:.4f}, OOF(naive) RMSE={log_rmse:,.0f}")

print("  주요 계수:")
for feat, coef in sorted(zip(log_features.columns, lr.coef_), key=lambda x: -abs(x[1])):
    if abs(coef) > 0.01:
        print(f"    {feat:20s}: {coef:+.4f}")

# B2. 곱셈 피처를 베이스 모델에 추가
print(f"\n--- B2. 곱셈 피처 추가 학습 ---")
# DGP 역공학 결과를 새 피처로 추가: log(Area)*log(Floor), Area*Floor*Brand 등
# 이 피처들로 잔차를 예측

# 먼저 곱셈 피처 생성
def make_mult_features(df):
    feat = pd.DataFrame(index=df.index)
    area = df['Exclusive_Area'].values
    floor = df['Floor'].values.clip(min=1)
    brand = df['Brand_Apartment'].values
    subway = df['Distance_to_Subway'].fillna(df['Distance_to_Subway'].median()).values
    parks = df['Nearby_Parks'].values
    age = 2026 - df['Year_Built'].values

    feat['log_area_x_log_floor'] = np.log1p(area) * np.log1p(floor)
    feat['area_x_floor_x_brand'] = area * floor * brand
    feat['area_x_age'] = area * age
    feat['log_area_x_log_age'] = np.log1p(area) * np.log1p(age)
    feat['price_proxy'] = area * floor * (1 + brand * 0.3) / (1 + subway * 0.001)
    feat['log_price_proxy'] = np.log1p(feat['price_proxy'])
    return feat

mult_train = make_mult_features(train_orig)
mult_test = make_mult_features(test_orig)

# 잔차 예측에 곱셈 피처 사용
for use_pred in [True, False]:
    resid_feat_tr = mult_train.copy()
    resid_feat_te = mult_test.copy()
    if use_pred:
        resid_feat_tr['pred'] = global_oof
        resid_feat_te['pred'] = global_test
        label_suffix = "+pred"
    else:
        label_suffix = ""

    corrected_oof = global_oof.copy()
    corrected_test_r = np.zeros(len(test_orig))

    for fold, (tr, va) in enumerate(kf.split(resid_feat_tr)):
        resid_model = GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            min_samples_leaf=20, subsample=0.8, random_state=42)
        resid_model.fit(resid_feat_tr.iloc[tr], residuals[tr])
        corrected_oof[va] -= resid_model.predict(resid_feat_tr.iloc[va])
        corrected_test_r += resid_model.predict(resid_feat_te) / N_SPLITS

    corrected_test_final = global_test - corrected_test_r
    r = np.sqrt(np.mean((corrected_oof - y_true_orig)**2))
    diff = r - global_rmse
    marker = ""
    if r < best_a_rmse:
        best_a_rmse = r
        best_a_test = corrected_test_final.copy()
        best_a_name = f"DGP Residual GBR{label_suffix}"
        marker = " ★"
    print(f"  곱셈피처 잔차보정{label_suffix}: OOF {r:,.0f} ({diff:+,.0f}){marker}")

# B3. 곱셈 피처를 스태킹에 직접 추가
print(f"\n--- B3. 곱셈 피처를 스태킹에 추가 ---")
mult_train_arr = mult_train.values
mult_test_arr = mult_test.values
stack_train_ext = np.column_stack([stack_train, mult_train_arr])
stack_test_ext = np.column_stack([stack_test, mult_test_arr])

for alpha in [1.0, 10.0, 50.0]:
    s_oof = np.zeros(n_orig)
    s_test = np.zeros(len(test_orig))
    for fold, (tr, va) in enumerate(kf.split(stack_train_ext)):
        meta = Ridge(alpha=alpha)
        meta.fit(stack_train_ext[tr], y_true_orig[tr])
        s_oof[va] = meta.predict(stack_train_ext[va])
        s_test += meta.predict(stack_test_ext) / N_SPLITS
    r = np.sqrt(np.mean((s_oof - y_true_orig)**2))
    diff = r - global_rmse
    marker = ""
    if r < best_a_rmse:
        best_a_rmse = r
        best_a_test = s_test.copy()
        best_a_name = f"12모델+곱셈피처 Ridge(α={alpha})"
        marker = " ★"
    print(f"  12모델+곱셈피처 α={alpha:5.1f}: OOF {r:,.0f} ({diff:+,.0f}){marker}")

# ========================================
# 최종 결과
# ========================================
print(f"\n{'=' * 60}")
print("최종 결과")
print("=" * 60)
print(f"  기준 (전략47): OOF {global_rmse:,.0f}")
print(f"  최선: {best_a_name} → OOF {best_a_rmse:,.0f}")

final_pred = best_a_test * trend_correction
sub = sample_sub.copy()
sub['Target'] = final_pred
sub.to_csv(os.path.join(os.path.dirname(__file__), '..', '..', 'submission_l4_51_resid_dgp.csv'), index=False)
print(f"\n제출 파일 생성: submission_l4_51_resid_dgp.csv")

record_result('L4', 51, 'RESID+DGP', f'고오차보정+DGP ({best_a_name}) + GTR', best_a_rmse, 'tested')
