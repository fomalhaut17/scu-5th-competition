"""
[L4-48] SplineTransformer 비트리 다양성
──────────────────────────
축약명  : SPLINE DIV
주요 전략: 전략47(12모델) + SplineTransformer×2변형 = 14모델 Ridge
차별점  : 트리와 다른 스플라인 기반 (corr~0.87), knots=10/deg=2 + knots=15/deg=3
참고    : KernelRidge(RBF)는 OOF 5,676+ → 앙상블 기여 불가로 제외
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import SplineTransformer
from sklearn.pipeline import Pipeline
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
# PL2
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
# Stage 2: 기존 12모델 + Spline 2모델
# ========================================
print(f"\n{'=' * 60}")
print("[Stage 2] 12모델(기존) + SplineTransformer 2 = 14모델")
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

    # --- CB/LGB 기존 4모델 ---
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

    # --- 평당가 4모델 ---
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

    # --- sklearn ExtraTrees ---
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

    # --- LGBM extra_trees ---
    m = lgb.LGBMRegressor(objective='regression', metric='rmse', verbose=-1, random_state=42, n_estimators=3000, **LGB_ET_PARAMS)
    m.fit(X_lgb2.iloc[tr], y_log_aug.iloc[tr], eval_set=[(X_lgb2.iloc[va], y_log_aug.iloc[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
    all_oof.setdefault('lgbet_log', np.zeros(len(X_cb2)))[va] = np.expm1(m.predict(X_lgb2.iloc[va]))
    all_tpred['lgbet_log'] = all_tpred.get('lgbet_log', np.zeros(len(X_test_lgb2))) + np.expm1(m.predict(X_test_lgb2)) / 5

    m = lgb.LGBMRegressor(objective='regression', metric='rmse', verbose=-1, random_state=42, n_estimators=3000, **LGB_ET_PARAMS)
    m.fit(X_lgb2.iloc[tr], y_raw_aug[tr], eval_set=[(X_lgb2.iloc[va], y_raw_aug[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
    all_oof.setdefault('lgbet_raw', np.zeros(len(X_cb2)))[va] = m.predict(X_lgb2.iloc[va])
    all_tpred['lgbet_raw'] = all_tpred.get('lgbet_raw', np.zeros(len(X_test_lgb2))) + m.predict(X_test_lgb2) / 5

    # ============================================
    # SplineTransformer 2변형 (NEW)
    # ============================================

    # Spline A: knots=10, degree=2, raw
    spl_a = Pipeline([
        ('scaler', StandardScaler()),
        ('spline', SplineTransformer(n_knots=10, degree=2, include_bias=False)),
        ('ridge', Ridge(alpha=0.1))
    ])
    spl_a.fit(X_cb2.iloc[tr].values, y_raw_aug[tr])
    all_oof.setdefault('spl_a', np.zeros(len(X_cb2)))[va] = spl_a.predict(X_cb2.iloc[va].values)
    all_tpred['spl_a'] = all_tpred.get('spl_a', np.zeros(len(X_test_cb2))) + spl_a.predict(X_test_cb2.values) / 5

    # Spline B: knots=15, degree=3, raw
    spl_b = Pipeline([
        ('scaler', StandardScaler()),
        ('spline', SplineTransformer(n_knots=15, degree=3, include_bias=False)),
        ('ridge', Ridge(alpha=0.1))
    ])
    spl_b.fit(X_cb2.iloc[tr].values, y_raw_aug[tr])
    all_oof.setdefault('spl_b', np.zeros(len(X_cb2)))[va] = spl_b.predict(X_cb2.iloc[va].values)
    all_tpred['spl_b'] = all_tpred.get('spl_b', np.zeros(len(X_test_cb2))) + spl_b.predict(X_test_cb2.values) / 5

# 평당가 → 원래 스케일 환산
for k in ['cb_up_log', 'cb_up_raw', 'lgb_up_log', 'lgb_up_raw']:
    all_oof[k] = all_oof[k][:n_orig] * area_train_orig
    all_tpred[k] = all_tpred[k] * area_test

for k in [kk for kk in all_oof if kk not in ['cb_up_log', 'cb_up_raw', 'lgb_up_log', 'lgb_up_raw']]:
    all_oof[k] = all_oof[k][:n_orig]

# ========================================
# 개별 모델 OOF + 다양성 분석
# ========================================
print(f"\n{'=' * 60}")
print("개별 모델 OOF + 기존 12모델 평균과의 상관")
print("=" * 60)

base_12 = ['cb_log', 'cb_raw', 'lgb_log', 'lgb_raw',
           'cb_up_log', 'cb_up_raw', 'lgb_up_log', 'lgb_up_raw',
           'et_log', 'et_raw', 'lgbet_log', 'lgbet_raw']
new_models = ['spl_a', 'spl_b']
existing_avg = np.mean([all_oof[k] for k in base_12], axis=0)

for k in sorted(all_oof.keys()):
    r = np.sqrt(np.mean((all_oof[k] - y_true_orig)**2))
    corr = np.corrcoef(existing_avg, all_oof[k])[0, 1]
    tag = " [NEW]" if k in new_models else ""
    print(f"  {k:15s}: OOF {r:>6,.0f}, corr={corr:.4f}{tag}")

# ========================================
# 앙상블 조합 비교
# ========================================
print(f"\n{'=' * 60}")
print("앙상블 조합 비교")
print("=" * 60)

def ridge_stack(keys, alpha):
    st_tr = np.column_stack([all_oof[k] for k in keys])
    st_te = np.column_stack([all_tpred[k] for k in keys])
    s_oof = np.zeros(n_orig)
    s_test = np.zeros(len(test_orig))
    for fold, (tr, va) in enumerate(kf.split(st_tr)):
        meta = Ridge(alpha=alpha)
        meta.fit(st_tr[tr], y_true_orig[tr])
        s_oof[va] = meta.predict(st_tr[va])
        s_test += meta.predict(st_te) / N_SPLITS
    r = np.sqrt(np.mean((s_oof - y_true_orig)**2))
    return s_oof, s_test, r

# 기준: 전략47 (12모델)
best_rmse = float('inf')
best_test = None
best_name = None

for alpha in [0.1, 0.5, 1.0, 5.0, 10.0, 50.0]:
    _, s_test, r = ridge_stack(base_12, alpha)
    if r < best_rmse:
        best_rmse = r
        best_test = s_test.copy()
        best_name = f"12모델 α={alpha}"
    print(f"  12모델(전략47)           α={alpha:5.1f}: OOF {r:,.0f}")

rmse_12 = best_rmse
print(f"\n  기준(전략47): {best_name} → OOF {rmse_12:,.0f}")

combos = {
    "12+SplA": base_12 + ['spl_a'],
    "12+SplB": base_12 + ['spl_b'],
    "12+SplA+SplB(=14)": base_12 + new_models,
}

for name, keys in combos.items():
    for alpha in [0.1, 0.5, 1.0, 5.0, 10.0, 50.0]:
        _, s_test, r = ridge_stack(keys, alpha)
        diff = r - rmse_12
        marker = ""
        if r < best_rmse:
            best_rmse = r
            best_test = s_test.copy()
            best_name = f"{name} α={alpha}"
            marker = " ★"
        print(f"  {name:25s} α={alpha:5.1f}: OOF {r:,.0f} ({diff:+,.0f}){marker}")

print(f"\n  ★ 최선: {best_name} → OOF {best_rmse:,.0f}")
print(f"  (전략 47: {rmse_12:,.0f})")

# 제출 파일
final_pred = best_test * trend_correction
sub = sample_sub.copy()
sub['Target'] = final_pred
sub.to_csv(os.path.join(os.path.dirname(__file__), '..', '..', 'submission_l4_48_spline.csv'), index=False)
print(f"\n제출 파일 생성: submission_l4_48_spline.csv")

record_result('L4', 48, 'SPLINE DIV', f'SplineTransformer 다양성 ({best_name}) + GTR', best_rmse, 'tested')
