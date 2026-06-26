"""
[L4-44] 알고리즘 다양성 앙상블
──────────────────────────
축약명  : DIVERSE
주요 전략: 기존 8모델(CB/LGB 트리) + 비트리 모델(KNN/Ridge-poly/ExtraTrees/SVR)
          → 확장된 Ridge 스태킹
차별점  : 지금까지 트리 모델만 사용. 알고리즘 차원의 다양성 추가
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.svm import SVR
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

# ========================================
# PL2 데이터 증강 (전략 28 동일)
# ========================================
print("=" * 60)
print("[Stage 1] PL2 데이터 증강")
print("=" * 60)

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
train_cb, test_cb, train_lgb, test_lgb = prepare_data(train_orig, test_orig)

def train_4tree(train_cb, test_cb, train_lgb, test_lgb, kf, label=""):
    X_cb = train_cb.drop(columns=['Target'])
    X_test_cb = test_cb
    cat_idx = [X_cb.columns.get_loc(c) for c in CAT_FEATURES]
    X_lgb = train_lgb.drop(columns=['Target'])
    X_test_lgb = test_lgb
    y_log = np.log1p(train_cb['Target'])
    y_raw = train_cb['Target'].values

    oof = {k: np.zeros(len(X_cb)) for k in MODELS}
    tpred = {k: np.zeros(len(X_test_cb)) for k in MODELS}
    fold_preds = {k: [] for k in MODELS}

    for fold, (tr, va) in enumerate(kf.split(X_cb)):
        print(f"  [{label}] Fold {fold+1}/{kf.n_splits}")
        m = CatBoostRegressor(loss_function='RMSE', random_seed=42, verbose=0,
                              iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
        m.fit(X_cb.iloc[tr], y_log.iloc[tr], eval_set=(X_cb.iloc[va], y_log.iloc[va]), cat_features=cat_idx)
        oof['cb_log'][va] = np.expm1(m.predict(X_cb.iloc[va]))
        fp = np.expm1(m.predict(X_test_cb)); tpred['cb_log'] += fp/kf.n_splits; fold_preds['cb_log'].append(fp)

        m = CatBoostRegressor(loss_function='RMSE', random_seed=42, verbose=0,
                              iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
        m.fit(X_cb.iloc[tr], y_raw[tr], eval_set=(X_cb.iloc[va], y_raw[va]), cat_features=cat_idx)
        oof['cb_raw'][va] = m.predict(X_cb.iloc[va])
        fp = m.predict(X_test_cb); tpred['cb_raw'] += fp/kf.n_splits; fold_preds['cb_raw'].append(fp)

        m = lgb.LGBMRegressor(objective='regression', metric='rmse', verbose=-1, random_state=42, n_estimators=3000, **LGB_PARAMS)
        m.fit(X_lgb.iloc[tr], y_log.iloc[tr], eval_set=[(X_lgb.iloc[va], y_log.iloc[va])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        oof['lgb_log'][va] = np.expm1(m.predict(X_lgb.iloc[va]))
        fp = np.expm1(m.predict(X_test_lgb)); tpred['lgb_log'] += fp/kf.n_splits; fold_preds['lgb_log'].append(fp)

        m = lgb.LGBMRegressor(objective='regression', metric='rmse', verbose=-1, random_state=42, n_estimators=3000, **LGB_PARAMS)
        m.fit(X_lgb.iloc[tr], y_raw[tr], eval_set=[(X_lgb.iloc[va], y_raw[va])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        oof['lgb_raw'][va] = m.predict(X_lgb.iloc[va])
        fp = m.predict(X_test_lgb); tpred['lgb_raw'] += fp/kf.n_splits; fold_preds['lgb_raw'].append(fp)
    return oof, tpred, fold_preds

oof_s1, tpred_s1, fold_preds_s1 = train_4tree(train_cb, test_cb, train_lgb, test_lgb, kf, "S1")

pseudo_labels = np.mean([tpred_s1[k] for k in MODELS], axis=0)
model_means = np.array([tpred_s1[k] for k in MODELS])
model_disagreement = np.std(model_means, axis=0) / np.mean(model_means, axis=0)
fold_cvs = [np.std(np.array(fold_preds_s1[k]), axis=0) / np.mean(np.array(fold_preds_s1[k]), axis=0) for k in MODELS]
confidence = 1 - (model_disagreement + np.mean(fold_cvs, axis=0)) / 2
threshold = np.percentile(confidence, 50)
mask = confidence >= threshold

test_selected = test_orig[mask].copy()
test_selected['Target'] = pseudo_labels[mask]
train_aug = pd.concat([train_orig, test_selected], ignore_index=True)
area_train_aug = train_aug['Exclusive_Area'].values

# ========================================
# 기존 8모델 (전략 28 동일)
# ========================================
print(f"\n{'=' * 60}")
print("[Stage 2] 기존 8모델 + 비트리 모델 학습")
print("=" * 60)

kf2 = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
train_cb2, test_cb2, train_lgb2, test_lgb2 = prepare_data(train_aug, test_orig)

print("\n--- 기존 4모델 ---")
oof_base, tpred_base, _ = train_4tree(train_cb2, test_cb2, train_lgb2, test_lgb2, kf2, "기존")
for k in MODELS: oof_base[k] = oof_base[k][:n_orig]

print("\n--- 평당가 4모델 ---")
y_up_log = np.log1p(train_aug['Target'].values / area_train_aug)
y_up_raw = (train_aug['Target'].values / area_train_aug).astype(float)

oof_unit = {k: np.zeros(len(train_aug)) for k in MODELS}
tpred_unit = {k: np.zeros(len(test_orig)) for k in MODELS}

X_cb_a = train_cb2.drop(columns=['Target'])
X_test_cb_a = test_cb2
cat_idx = [X_cb_a.columns.get_loc(c) for c in CAT_FEATURES]
X_lgb_a = train_lgb2.drop(columns=['Target'])
X_test_lgb_a = test_lgb2

for fold, (tr, va) in enumerate(kf2.split(X_cb_a)):
    print(f"  [평당가] Fold {fold+1}/5")
    m = CatBoostRegressor(loss_function='RMSE', random_seed=42, verbose=0, iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
    m.fit(X_cb_a.iloc[tr], y_up_log[tr], eval_set=(X_cb_a.iloc[va], y_up_log[va]), cat_features=cat_idx)
    oof_unit['cb_log'][va] = np.expm1(m.predict(X_cb_a.iloc[va]))
    tpred_unit['cb_log'] += np.expm1(m.predict(X_test_cb_a)) / 5

    m = CatBoostRegressor(loss_function='RMSE', random_seed=42, verbose=0, iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
    m.fit(X_cb_a.iloc[tr], y_up_raw[tr], eval_set=(X_cb_a.iloc[va], y_up_raw[va]), cat_features=cat_idx)
    oof_unit['cb_raw'][va] = m.predict(X_cb_a.iloc[va])
    tpred_unit['cb_raw'] += m.predict(X_test_cb_a) / 5

    m = lgb.LGBMRegressor(objective='regression', metric='rmse', verbose=-1, random_state=42, n_estimators=3000, **LGB_PARAMS)
    m.fit(X_lgb_a.iloc[tr], y_up_log[tr], eval_set=[(X_lgb_a.iloc[va], y_up_log[va])],
          callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_unit['lgb_log'][va] = np.expm1(m.predict(X_lgb_a.iloc[va]))
    tpred_unit['lgb_log'] += np.expm1(m.predict(X_test_lgb_a)) / 5

    m = lgb.LGBMRegressor(objective='regression', metric='rmse', verbose=-1, random_state=42, n_estimators=3000, **LGB_PARAMS)
    m.fit(X_lgb_a.iloc[tr], y_up_raw[tr], eval_set=[(X_lgb_a.iloc[va], y_up_raw[va])],
          callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_unit['lgb_raw'][va] = m.predict(X_lgb_a.iloc[va])
    tpred_unit['lgb_raw'] += m.predict(X_test_lgb_a) / 5

for k in MODELS:
    oof_unit[k] = oof_unit[k][:n_orig] * area_train_orig
    tpred_unit[k] = tpred_unit[k] * area_test

# ========================================
# 비트리 모델 학습 (KNN, Ridge-poly, ExtraTrees, SVR)
# ========================================
print(f"\n{'=' * 60}")
print("비트리 모델 학습")
print("=" * 60)

# 스케일링 필요 (KNN, SVR, Ridge-poly)
X_scaled_full = train_cb2.drop(columns=['Target']).copy()
X_test_scaled_full = test_cb2.copy() if 'Target' not in test_cb2.columns else test_cb2.drop(columns=['Target']).copy()

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_scaled_full), columns=X_scaled_full.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test_scaled_full), columns=X_test_scaled_full.columns)

y_aug = train_aug['Target'].values
y_aug_log = np.log1p(y_aug)

new_models = {}

# --- 1. KNN ---
print("\n--- KNN ---")
for k_val in [5, 10, 20, 50]:
    oof_knn = np.zeros(len(X_scaled))
    tpred_knn = np.zeros(len(X_test_scaled))
    for fold, (tr, va) in enumerate(kf2.split(X_scaled)):
        m = KNeighborsRegressor(n_neighbors=k_val, weights='distance')
        m.fit(X_scaled.iloc[tr], y_aug_log[tr])
        oof_knn[va] = np.expm1(m.predict(X_scaled.iloc[va]))
        tpred_knn += np.expm1(m.predict(X_test_scaled)) / N_SPLITS
    oof_knn_orig = oof_knn[:n_orig]
    r = np.sqrt(np.mean((oof_knn_orig - y_true_orig)**2))
    print(f"  KNN(k={k_val:2d}): OOF {r:,.0f}")
    if f'knn_{k_val}' not in new_models or r < new_models.get(f'knn_best', (None, None, float('inf')))[2]:
        new_models['knn_best'] = (oof_knn_orig, tpred_knn, r, f'KNN(k={k_val})')

# --- 2. Ridge + Polynomial Features ---
print("\n--- Ridge + Polynomial ---")
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = pd.DataFrame(poly.fit_transform(X_scaled))
X_test_poly = pd.DataFrame(poly.transform(X_test_scaled))

for alpha in [1.0, 10.0, 100.0, 1000.0]:
    oof_rp = np.zeros(len(X_poly))
    tpred_rp = np.zeros(len(X_test_poly))
    for fold, (tr, va) in enumerate(kf2.split(X_poly)):
        m = Ridge(alpha=alpha)
        m.fit(X_poly.iloc[tr], y_aug[tr])
        oof_rp[va] = m.predict(X_poly.iloc[va])
        tpred_rp += m.predict(X_test_poly) / N_SPLITS
    oof_rp_orig = oof_rp[:n_orig]
    r = np.sqrt(np.mean((oof_rp_orig - y_true_orig)**2))
    print(f"  Ridge-poly(α={alpha:6.0f}): OOF {r:,.0f}")
    if 'rpoly_best' not in new_models or r < new_models['rpoly_best'][2]:
        new_models['rpoly_best'] = (oof_rp_orig, tpred_rp, r, f'Ridge-poly(α={alpha})')

# --- 3. ExtraTrees ---
print("\n--- ExtraTrees ---")
for n_est, max_d in [(500, 8), (500, 12), (1000, 8), (1000, 12)]:
    oof_et = np.zeros(len(X_scaled))
    tpred_et = np.zeros(len(X_test_scaled))
    for fold, (tr, va) in enumerate(kf2.split(X_scaled)):
        m = ExtraTreesRegressor(n_estimators=n_est, max_depth=max_d,
                                 min_samples_leaf=10, random_state=42, n_jobs=-1)
        m.fit(X_scaled.iloc[tr], y_aug_log[tr])
        oof_et[va] = np.expm1(m.predict(X_scaled.iloc[va]))
        tpred_et += np.expm1(m.predict(X_test_scaled)) / N_SPLITS
    oof_et_orig = oof_et[:n_orig]
    r = np.sqrt(np.mean((oof_et_orig - y_true_orig)**2))
    print(f"  ExtraTrees(n={n_est}, d={max_d:2d}): OOF {r:,.0f}")
    if 'et_best' not in new_models or r < new_models['et_best'][2]:
        new_models['et_best'] = (oof_et_orig, tpred_et, r, f'ExtraTrees(n={n_est},d={max_d})')

# --- 4. SVR ---
print("\n--- SVR ---")
for C_val in [1.0, 10.0, 100.0]:
    oof_svr = np.zeros(len(X_scaled))
    tpred_svr = np.zeros(len(X_test_scaled))
    for fold, (tr, va) in enumerate(kf2.split(X_scaled)):
        m = SVR(kernel='rbf', C=C_val, epsilon=0.1)
        m.fit(X_scaled.iloc[tr], y_aug_log[tr])
        oof_svr[va] = np.expm1(m.predict(X_scaled.iloc[va]))
        tpred_svr += np.expm1(m.predict(X_test_scaled)) / N_SPLITS
    oof_svr_orig = oof_svr[:n_orig]
    r = np.sqrt(np.mean((oof_svr_orig - y_true_orig)**2))
    print(f"  SVR(C={C_val:5.1f}): OOF {r:,.0f}")
    if 'svr_best' not in new_models or r < new_models['svr_best'][2]:
        new_models['svr_best'] = (oof_svr_orig, tpred_svr, r, f'SVR(C={C_val})')

# ========================================
# 다양성 분석: 기존 모델과 상관관계
# ========================================
print(f"\n{'=' * 60}")
print("다양성 분석 (기존 8모델 평균과의 상관)")
print("=" * 60)

existing_avg = np.mean([oof_base[k] for k in MODELS] + [oof_unit[k] for k in MODELS], axis=0)

for name, (oof_new, _, rmse_new, desc) in new_models.items():
    corr = np.corrcoef(existing_avg, oof_new)[0, 1]
    print(f"  {desc:30s}: OOF {rmse_new:,.0f}, corr={corr:.4f}")

# ========================================
# 확장 앙상블: 8모델 + 비트리 모델 → Ridge
# ========================================
print(f"\n{'=' * 60}")
print("확장 앙상블: 기존 8모델 + 비트리 모델")
print("=" * 60)

stack_8 = np.column_stack([oof_base[k] for k in MODELS] + [oof_unit[k] for k in MODELS])
stack_8_test = np.column_stack([tpred_base[k] for k in MODELS] + [tpred_unit[k] for k in MODELS])

# 기준: 8모델만
s_oof_8 = np.zeros(n_orig)
s_test_8 = np.zeros(len(test_orig))
for fold, (tr, va) in enumerate(kf.split(stack_8)):
    meta = Ridge(alpha=10.0)
    meta.fit(stack_8[tr], y_true_orig[tr])
    s_oof_8[va] = meta.predict(stack_8[va])
    s_test_8 += meta.predict(stack_8_test) / N_SPLITS
rmse_8 = np.sqrt(np.mean((s_oof_8 - y_true_orig)**2))
print(f"\n  기준 (8모델 Ridge): OOF {rmse_8:,.0f}")

# 각 비트리 모델을 1개씩 추가
for name, (oof_new, tpred_new, _, desc) in new_models.items():
    stack_ext = np.column_stack([stack_8, oof_new])
    stack_ext_test = np.column_stack([stack_8_test, tpred_new])

    for alpha in [10.0, 50.0, 100.0]:
        s_oof = np.zeros(n_orig)
        s_test = np.zeros(len(test_orig))
        for fold, (tr, va) in enumerate(kf.split(stack_ext)):
            meta = Ridge(alpha=alpha)
            meta.fit(stack_ext[tr], y_true_orig[tr])
            s_oof[va] = meta.predict(stack_ext[va])
            s_test += meta.predict(stack_ext_test) / N_SPLITS
        r = np.sqrt(np.mean((s_oof - y_true_orig)**2))
        diff = r - rmse_8
        print(f"  +{desc:30s} α={alpha:5.0f}: OOF {r:,.0f} ({diff:+,.0f})")

# 전부 추가
all_new_oof = [v[0] for v in new_models.values()]
all_new_test = [v[1] for v in new_models.values()]
stack_all = np.column_stack([stack_8] + all_new_oof)
stack_all_test = np.column_stack([stack_8_test] + all_new_test)

print(f"\n  --- 전체 추가 (8 + {len(new_models)}개 비트리) ---")
for alpha in [10.0, 50.0, 100.0, 500.0]:
    s_oof = np.zeros(n_orig)
    s_test = np.zeros(len(test_orig))
    for fold, (tr, va) in enumerate(kf.split(stack_all)):
        meta = Ridge(alpha=alpha)
        meta.fit(stack_all[tr], y_true_orig[tr])
        s_oof[va] = meta.predict(stack_all[va])
        s_test += meta.predict(stack_all_test) / N_SPLITS
    r = np.sqrt(np.mean((s_oof - y_true_orig)**2))
    diff = r - rmse_8
    marker = " ★" if diff < 0 else ""
    print(f"  전체 Ridge(α={alpha:5.0f}): OOF {r:,.0f} ({diff:+,.0f}){marker}")

    if diff < 0:
        best_all_test = s_test.copy()
        best_all_rmse = r

# 제출 파일 생성
if 'best_all_test' in dir() and 'best_all_rmse' in dir():
    final_pred = best_all_test * trend_correction
    sub = sample_sub.copy()
    sub['Target'] = final_pred
    sub.to_csv(os.path.join(os.path.dirname(__file__), '..', '..', 'submission_l4_44_diverse.csv'), index=False)
    print(f"\n제출 파일 생성: submission_l4_44_diverse.csv (OOF {best_all_rmse:,.0f})")
    record_result('L4', 44, 'DIVERSE', f'8모델+비트리 Ridge + GTR', best_all_rmse, 'tested')
else:
    print(f"\n개선 없음 — 기존 전략 28 유지")
    record_result('L4', 44, 'DIVERSE', f'비트리 모델 추가 실험 (효과 없음)', rmse_8, 'tested')
