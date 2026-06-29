"""
63 LINEAR SKELETON + GBDT RESIDUAL
선형 모델이 시간 외삽(구조) 담당 → GBDT가 잔차(비선형) 학습 → GTR 불필요
핵심: 트리는 2026을 외삽할 수 없지만, 선형 모델은 가능
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

if os.path.exists('/kaggle/input'):
    INPUT_DIR = '/kaggle/input/competitions/scu-5th-ai-competition'
    OUTPUT_DIR = '/kaggle/working'
else:
    _DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    INPUT_DIR = _DIR
    OUTPUT_DIR = _DIR

N_SPLITS = 5
CAT_FEATURES = ['Gu', 'Dong']
SEED = 42

train_orig = pd.read_csv(f'{INPUT_DIR}/seoul_real_estate_train.csv')
test_orig = pd.read_csv(f'{INPUT_DIR}/seoul_real_estate_test.csv')
sample_sub = pd.read_csv(f'{INPUT_DIR}/sample_submission.csv')
y_true = train_orig['Target'].values
n_orig = len(train_orig)
area_train = train_orig['Exclusive_Area'].values
area_test = test_orig['Exclusive_Area'].values

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
    tr_cb, te_cb = encode_categoricals(train_p, test_p, as_category=False)
    tr_lgb, te_lgb = encode_categoricals(train_p, test_p, as_category=True)
    return tr_cb, te_cb, tr_lgb, te_lgb

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

train_cb, test_cb, train_lgb, test_lgb = prepare_data(train_orig, test_orig)
X_cb = train_cb.drop(columns=['Target'])
X_test_cb = test_cb
cat_idx = [X_cb.columns.get_loc(c) for c in CAT_FEATURES]
X_lgb = train_lgb.drop(columns=['Target'])
X_test_lgb = test_lgb

# =============================================
# STEP 1: SKELETON MODEL (선형 모델 → 시간 외삽 담당)
# =============================================
print("=" * 60)
print("STEP 1: Linear Skeleton (Ridge on log-scale)")
print("=" * 60)

scaler_skel = StandardScaler()
X_skel_all = scaler_skel.fit_transform(X_cb.values.astype(float))
X_skel_test = scaler_skel.transform(X_test_cb.values.astype(float))
y_log = np.log1p(y_true)

skeleton_oof = np.zeros(n_orig)
skeleton_test = np.zeros(len(test_orig))
kf_skel = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

best_alpha_skel = None
best_rmse_skel = float('inf')
for alpha in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
    oof_tmp = np.zeros(n_orig)
    for tr, va in kf_skel.split(X_skel_all):
        m = Ridge(alpha=alpha)
        m.fit(X_skel_all[tr], y_log[tr])
        oof_tmp[va] = m.predict(X_skel_all[va])
    rmse = np.sqrt(np.mean((np.expm1(oof_tmp) - y_true) ** 2))
    if rmse < best_rmse_skel:
        best_rmse_skel = rmse
        best_alpha_skel = alpha

print(f"  Skeleton best alpha: {best_alpha_skel}, OOF RMSE: {best_rmse_skel:,.0f}")

for tr, va in kf_skel.split(X_skel_all):
    m = Ridge(alpha=best_alpha_skel)
    m.fit(X_skel_all[tr], y_log[tr])
    skeleton_oof[va] = np.expm1(m.predict(X_skel_all[va]))

m_full = Ridge(alpha=best_alpha_skel)
m_full.fit(X_skel_all, y_log)
skeleton_test = np.expm1(m_full.predict(X_skel_test))

residuals = y_true - skeleton_oof
print(f"  Skeleton OOF RMSE: {best_rmse_skel:,.0f}")
print(f"  Residual mean: {residuals.mean():,.0f}, std: {residuals.std():,.0f}")

# =============================================
# STEP 2: GBDT on RESIDUALS (비선형 패턴 학습)
# =============================================
print(f"\n{'=' * 60}")
print("STEP 2: GBDT Residual Models (4시드 평균)")
print("=" * 60)

y_resid = residuals.astype(float)
y_resid_up = (residuals / area_train).astype(float)

def train_residual_models(X_cb, X_test_cb, X_lgb, X_test_lgb, cat_idx,
                          y_resid, y_resid_up, area_tr, area_te,
                          n_orig, kf, seed, label):
    oof, tpred = {}, {}
    n_test = len(X_test_cb)

    for fold, (tr, va) in enumerate(kf.split(X_cb)):
        print(f"  [{label}] Fold {fold+1}/5")

        # CB raw residual
        m = CatBoostRegressor(loss_function='RMSE', random_seed=seed, verbose=0,
                              iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
        m.fit(X_cb.iloc[tr], y_resid[tr], eval_set=(X_cb.iloc[va], y_resid[va]), cat_features=cat_idx)
        oof.setdefault('cb_resid', np.zeros(len(X_cb)))[va] = m.predict(X_cb.iloc[va])
        tpred['cb_resid'] = tpred.get('cb_resid', np.zeros(n_test)) + m.predict(X_test_cb) / 5

        # LGB raw residual
        m = lgb.LGBMRegressor(objective='regression', metric='rmse', verbose=-1,
                               random_state=seed, n_estimators=3000, **LGB_PARAMS)
        m.fit(X_lgb.iloc[tr], y_resid[tr], eval_set=[(X_lgb.iloc[va], y_resid[va])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        oof.setdefault('lgb_resid', np.zeros(len(X_cb)))[va] = m.predict(X_lgb.iloc[va])
        tpred['lgb_resid'] = tpred.get('lgb_resid', np.zeros(n_test)) + m.predict(X_test_lgb) / 5

        # CB unit residual (residual/area)
        m = CatBoostRegressor(loss_function='RMSE', random_seed=seed, verbose=0,
                              iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
        m.fit(X_cb.iloc[tr], y_resid_up[tr], eval_set=(X_cb.iloc[va], y_resid_up[va]), cat_features=cat_idx)
        oof.setdefault('cb_resid_up', np.zeros(len(X_cb)))[va] = m.predict(X_cb.iloc[va])
        tpred['cb_resid_up'] = tpred.get('cb_resid_up', np.zeros(n_test)) + m.predict(X_test_cb) / 5

        # LGB unit residual
        m = lgb.LGBMRegressor(objective='regression', metric='rmse', verbose=-1,
                               random_state=seed, n_estimators=3000, **LGB_PARAMS)
        m.fit(X_lgb.iloc[tr], y_resid_up[tr], eval_set=[(X_lgb.iloc[va], y_resid_up[va])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        oof.setdefault('lgb_resid_up', np.zeros(len(X_cb)))[va] = m.predict(X_lgb.iloc[va])
        tpred['lgb_resid_up'] = tpred.get('lgb_resid_up', np.zeros(n_test)) + m.predict(X_test_lgb) / 5

        # ET raw residual
        scaler = StandardScaler()
        X_sc_tr = scaler.fit_transform(X_cb.iloc[tr])
        X_sc_va = scaler.transform(X_cb.iloc[va])
        X_sc_te = scaler.transform(X_test_cb)

        m = ExtraTreesRegressor(n_estimators=500, max_depth=12, min_samples_leaf=10,
                                 random_state=seed, n_jobs=-1)
        m.fit(X_sc_tr, y_resid[tr])
        oof.setdefault('et_resid', np.zeros(len(X_cb)))[va] = m.predict(X_sc_va)
        tpred['et_resid'] = tpred.get('et_resid', np.zeros(n_test)) + m.predict(X_sc_te) / 5

        # LGBET raw residual
        m = lgb.LGBMRegressor(objective='regression', metric='rmse', verbose=-1,
                               random_state=seed, n_estimators=3000, **LGB_ET_PARAMS)
        m.fit(X_lgb.iloc[tr], y_resid[tr], eval_set=[(X_lgb.iloc[va], y_resid[va])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        oof.setdefault('lgbet_resid', np.zeros(len(X_cb)))[va] = m.predict(X_lgb.iloc[va])
        tpred['lgbet_resid'] = tpred.get('lgbet_resid', np.zeros(n_test)) + m.predict(X_test_lgb) / 5

    # unit residual → raw residual 환산
    for k in ['cb_resid_up', 'lgb_resid_up']:
        oof[k] = oof[k][:n_orig] * area_tr[:n_orig]
        tpred[k] = tpred[k] * area_te
    for k in [kk for kk in oof if kk not in ['cb_resid_up', 'lgb_resid_up']]:
        oof[k] = oof[k][:n_orig]

    return oof, tpred

# 4시드 평균
seed_tests = []
for seed in [42, 123, 456, 789]:
    print(f"\n--- Seed {seed} ---")
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    oof_r, tpred_r = train_residual_models(
        X_cb, X_test_cb, X_lgb, X_test_lgb, cat_idx,
        y_resid, y_resid_up, area_train, area_test,
        n_orig, kf, seed, f"Resid-{seed}"
    )

    # Ridge stack residuals
    model_names = ['cb_resid', 'lgb_resid', 'cb_resid_up', 'lgb_resid_up', 'et_resid', 'lgbet_resid']
    st_tr = np.column_stack([oof_r[k] for k in model_names])
    st_te = np.column_stack([tpred_r[k] for k in model_names])

    best_rmse, best_test = float('inf'), None
    kf_meta = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    for alpha in [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]:
        s_oof = np.zeros(n_orig)
        s_test = np.zeros(len(test_orig))
        for tr, va in kf_meta.split(st_tr):
            meta = Ridge(alpha=alpha)
            meta.fit(st_tr[tr], y_resid[tr])
            s_oof[va] = meta.predict(st_tr[va])
            s_test += meta.predict(st_te) / N_SPLITS
        full_oof = skeleton_oof + s_oof
        rmse = np.sqrt(np.mean((full_oof - y_true) ** 2))
        if rmse < best_rmse:
            best_rmse = rmse
            best_test = s_test.copy()

    print(f"  Skeleton+Resid OOF RMSE: {best_rmse:,.0f}")
    seed_tests.append(best_test)

resid_test_avg = np.mean(seed_tests, axis=0)

# =============================================
# STEP 3: 최종 예측 (Skeleton + Residual, GTR 없음)
# =============================================
print(f"\n{'=' * 60}")
print("STEP 3: Final = Skeleton + Residual (NO GTR)")
print("=" * 60)

final_pred = skeleton_test + resid_test_avg

print(f"  Skeleton test mean: {skeleton_test.mean():,.0f}")
print(f"  Residual test mean: {resid_test_avg.mean():,.0f}")
print(f"  Final test mean: {final_pred.mean():,.0f}")

sub = sample_sub.copy()
sub['Target'] = final_pred
sub.to_csv(f'{OUTPUT_DIR}/submission.csv', index=False)
print(f"\n제출 파일 저장 완료: {OUTPUT_DIR}/submission.csv")
