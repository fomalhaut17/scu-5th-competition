"""
76 4-WAY BLEND: 56 (GBDT+GTR) + 63 (Per-Gu Skeleton) + 69 (One-Hot Skeleton) + up (UnitPrice Skeleton)
전략75(3-way 50:30:20, Public 2,028.4)에 4번째 컴포넌트(평당가 기반 Per-Gu Skeleton)를 추가.

가중치 56:63:69:up = 40:15:25:20은 OOT(시간분리 holdout, oot_split holdout_months=3)로
검증해서 정함. naive in-sample OOF는 신뢰 불가(이전에 90:10:0을 골라 확정된 사실과 상충했던
전례 있음). OOT 기준 56단독 2,631 -> 60:40 2,601 -> 3-way(50:30:20) 2,595.1 ->
4-way(40:15:25:20) 2,587.1로 일관되게 개선(사니티체크 통과).
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
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
MODELS = ['cb_log', 'cb_raw', 'lgb_log', 'lgb_raw']
BLEND_W53 = 0.8

train_orig = pd.read_csv(f'{INPUT_DIR}/seoul_real_estate_train.csv')
test_orig = pd.read_csv(f'{INPUT_DIR}/seoul_real_estate_test.csv')
sample_sub = pd.read_csv(f'{INPUT_DIR}/sample_submission.csv')
y_true_orig = train_orig['Target'].values
n_orig = len(train_orig)
area_train = train_orig['Exclusive_Area'].values
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

train_cb, test_cb, train_lgb, test_lgb = prepare_data(train_orig, test_orig)
X_cb = train_cb.drop(columns=['Target'])
X_test_cb = test_cb
cat_idx = [X_cb.columns.get_loc(c) for c in CAT_FEATURES]
X_lgb = train_lgb.drop(columns=['Target'])
X_test_lgb = test_lgb

def train_12models(X_cb, X_test_cb, X_lgb, X_test_lgb, cat_idx,
                   y_log, y_raw, y_up_log, y_up_raw, area_tr, area_te,
                   n_orig, kf, seed, label):
    all_oof = {}
    all_tpred = {}

    for fold, (tr, va) in enumerate(kf.split(X_cb)):
        print(f"  [{label}] Fold {fold+1}/5")

        m = CatBoostRegressor(loss_function='RMSE', random_seed=seed, verbose=0, iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
        m.fit(X_cb.iloc[tr], y_log[tr] if isinstance(y_log, np.ndarray) else y_log.iloc[tr],
              eval_set=(X_cb.iloc[va], y_log[va] if isinstance(y_log, np.ndarray) else y_log.iloc[va]), cat_features=cat_idx)
        all_oof.setdefault('cb_log', np.zeros(len(X_cb)))[va] = np.expm1(m.predict(X_cb.iloc[va]))
        all_tpred['cb_log'] = all_tpred.get('cb_log', np.zeros(len(X_test_cb))) + np.expm1(m.predict(X_test_cb)) / 5

        m = CatBoostRegressor(loss_function='RMSE', random_seed=seed, verbose=0, iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
        m.fit(X_cb.iloc[tr], y_raw[tr], eval_set=(X_cb.iloc[va], y_raw[va]), cat_features=cat_idx)
        all_oof.setdefault('cb_raw', np.zeros(len(X_cb)))[va] = m.predict(X_cb.iloc[va])
        all_tpred['cb_raw'] = all_tpred.get('cb_raw', np.zeros(len(X_test_cb))) + m.predict(X_test_cb) / 5

        m = lgb.LGBMRegressor(objective='regression', metric='rmse', verbose=-1, random_state=seed, n_estimators=3000, **LGB_PARAMS)
        m.fit(X_lgb.iloc[tr], y_log[tr] if isinstance(y_log, np.ndarray) else y_log.iloc[tr],
              eval_set=[(X_lgb.iloc[va], y_log[va] if isinstance(y_log, np.ndarray) else y_log.iloc[va])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        all_oof.setdefault('lgb_log', np.zeros(len(X_cb)))[va] = np.expm1(m.predict(X_lgb.iloc[va]))
        all_tpred['lgb_log'] = all_tpred.get('lgb_log', np.zeros(len(X_test_cb))) + np.expm1(m.predict(X_test_lgb)) / 5

        m = lgb.LGBMRegressor(objective='regression', metric='rmse', verbose=-1, random_state=seed, n_estimators=3000, **LGB_PARAMS)
        m.fit(X_lgb.iloc[tr], y_raw[tr], eval_set=[(X_lgb.iloc[va], y_raw[va])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        all_oof.setdefault('lgb_raw', np.zeros(len(X_cb)))[va] = m.predict(X_lgb.iloc[va])
        all_tpred['lgb_raw'] = all_tpred.get('lgb_raw', np.zeros(len(X_test_cb))) + m.predict(X_test_lgb) / 5

        m = CatBoostRegressor(loss_function='RMSE', random_seed=seed, verbose=0, iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
        m.fit(X_cb.iloc[tr], y_up_log[tr], eval_set=(X_cb.iloc[va], y_up_log[va]), cat_features=cat_idx)
        all_oof.setdefault('cb_up_log', np.zeros(len(X_cb)))[va] = np.expm1(m.predict(X_cb.iloc[va]))
        all_tpred['cb_up_log'] = all_tpred.get('cb_up_log', np.zeros(len(X_test_cb))) + np.expm1(m.predict(X_test_cb)) / 5

        m = CatBoostRegressor(loss_function='RMSE', random_seed=seed, verbose=0, iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
        m.fit(X_cb.iloc[tr], y_up_raw[tr], eval_set=(X_cb.iloc[va], y_up_raw[va]), cat_features=cat_idx)
        all_oof.setdefault('cb_up_raw', np.zeros(len(X_cb)))[va] = m.predict(X_cb.iloc[va])
        all_tpred['cb_up_raw'] = all_tpred.get('cb_up_raw', np.zeros(len(X_test_cb))) + m.predict(X_test_cb) / 5

        m = lgb.LGBMRegressor(objective='regression', metric='rmse', verbose=-1, random_state=seed, n_estimators=3000, **LGB_PARAMS)
        m.fit(X_lgb.iloc[tr], y_up_log[tr], eval_set=[(X_lgb.iloc[va], y_up_log[va])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        all_oof.setdefault('lgb_up_log', np.zeros(len(X_cb)))[va] = np.expm1(m.predict(X_lgb.iloc[va]))
        all_tpred['lgb_up_log'] = all_tpred.get('lgb_up_log', np.zeros(len(X_test_cb))) + np.expm1(m.predict(X_test_lgb)) / 5

        m = lgb.LGBMRegressor(objective='regression', metric='rmse', verbose=-1, random_state=seed, n_estimators=3000, **LGB_PARAMS)
        m.fit(X_lgb.iloc[tr], y_up_raw[tr], eval_set=[(X_lgb.iloc[va], y_up_raw[va])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        all_oof.setdefault('lgb_up_raw', np.zeros(len(X_cb)))[va] = m.predict(X_lgb.iloc[va])
        all_tpred['lgb_up_raw'] = all_tpred.get('lgb_up_raw', np.zeros(len(X_test_cb))) + m.predict(X_test_lgb) / 5

        scaler = StandardScaler()
        X_sc_tr = scaler.fit_transform(X_cb.iloc[tr])
        X_sc_va = scaler.transform(X_cb.iloc[va])
        X_sc_te = scaler.transform(X_test_cb)

        m = ExtraTreesRegressor(n_estimators=500, max_depth=12, min_samples_leaf=10, random_state=seed, n_jobs=-1)
        m.fit(X_sc_tr, y_log[tr] if isinstance(y_log, np.ndarray) else y_log.iloc[tr])
        all_oof.setdefault('et_log', np.zeros(len(X_cb)))[va] = np.expm1(m.predict(X_sc_va))
        all_tpred['et_log'] = all_tpred.get('et_log', np.zeros(len(X_test_cb))) + np.expm1(m.predict(X_sc_te)) / 5

        m = ExtraTreesRegressor(n_estimators=500, max_depth=12, min_samples_leaf=10, random_state=seed, n_jobs=-1)
        m.fit(X_sc_tr, y_raw[tr])
        all_oof.setdefault('et_raw', np.zeros(len(X_cb)))[va] = m.predict(X_sc_va)
        all_tpred['et_raw'] = all_tpred.get('et_raw', np.zeros(len(X_test_cb))) + m.predict(X_sc_te) / 5

        m = lgb.LGBMRegressor(objective='regression', metric='rmse', verbose=-1, random_state=seed, n_estimators=3000, **LGB_ET_PARAMS)
        m.fit(X_lgb.iloc[tr], y_log[tr] if isinstance(y_log, np.ndarray) else y_log.iloc[tr],
              eval_set=[(X_lgb.iloc[va], y_log[va] if isinstance(y_log, np.ndarray) else y_log.iloc[va])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        all_oof.setdefault('lgbet_log', np.zeros(len(X_cb)))[va] = np.expm1(m.predict(X_lgb.iloc[va]))
        all_tpred['lgbet_log'] = all_tpred.get('lgbet_log', np.zeros(len(X_test_cb))) + np.expm1(m.predict(X_test_lgb)) / 5

        m = lgb.LGBMRegressor(objective='regression', metric='rmse', verbose=-1, random_state=seed, n_estimators=3000, **LGB_ET_PARAMS)
        m.fit(X_lgb.iloc[tr], y_raw[tr], eval_set=[(X_lgb.iloc[va], y_raw[va])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        all_oof.setdefault('lgbet_raw', np.zeros(len(X_cb)))[va] = m.predict(X_lgb.iloc[va])
        all_tpred['lgbet_raw'] = all_tpred.get('lgbet_raw', np.zeros(len(X_test_cb))) + m.predict(X_test_lgb) / 5

    for k in ['cb_up_log', 'cb_up_raw', 'lgb_up_log', 'lgb_up_raw']:
        all_oof[k] = all_oof[k][:n_orig] * area_tr[:n_orig]
        all_tpred[k] = all_tpred[k] * area_te
    for k in [kk for kk in all_oof if kk not in ['cb_up_log', 'cb_up_raw', 'lgb_up_log', 'lgb_up_raw']]:
        all_oof[k] = all_oof[k][:n_orig]

    return all_oof, all_tpred

def ridge_stack(all_oof, all_tpred, y_true, n_orig, n_test, kf_meta):
    """Best-alpha Ridge stacking. 이제 best_oof(전 구간 OOF)까지 같이 반환."""
    base_12 = ['cb_log', 'cb_raw', 'lgb_log', 'lgb_raw',
               'cb_up_log', 'cb_up_raw', 'lgb_up_log', 'lgb_up_raw',
               'et_log', 'et_raw', 'lgbet_log', 'lgbet_raw']
    st_tr = np.column_stack([all_oof[k] for k in base_12])
    st_te = np.column_stack([all_tpred[k] for k in base_12])
    best_rmse = float('inf')
    best_test = None
    best_oof = None
    for alpha in [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]:
        s_oof = np.zeros(n_orig)
        s_test = np.zeros(n_test)
        for fold, (tr, va) in enumerate(kf_meta.split(st_tr)):
            meta = Ridge(alpha=alpha)
            meta.fit(st_tr[tr], y_true[tr])
            s_oof[va] = meta.predict(st_tr[va])
            s_test += meta.predict(st_te) / kf_meta.n_splits
        rmse = np.sqrt(np.mean((s_oof - y_true) ** 2))
        if rmse < best_rmse:
            best_rmse = rmse
            best_test = s_test.copy()
            best_oof = s_oof.copy()
    return best_test, best_rmse, best_oof

def residual_gbdt(X_cb, X_test_cb, X_lgb, X_test_lgb, cat_idx, y_resid, y_resid_up,
                   area_tr, area_te, n_orig, n_test, label):
    """4시드 GBDT 잔차 학습. (test 예측 평균, OOF 평균) 반환."""
    seed_tests, seed_oofs = [], []
    for seed in [42, 123, 456, 789]:
        print(f"  --- {label} Resid Seed {seed} ---")
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        oof_r, tpred_r = {}, {}
        for fold, (tr, va) in enumerate(kf.split(X_cb)):
            for nm, Xtr, Xva, Xte, y, params, is_cb in [
                ('cb_r', X_cb, X_cb, X_test_cb, y_resid, CB_PARAMS, True),
                ('cb_ru', X_cb, X_cb, X_test_cb, y_resid_up, CB_PARAMS, True),
                ('lgb_r', X_lgb, X_lgb, X_test_lgb, y_resid, LGB_PARAMS, False),
                ('lgb_ru', X_lgb, X_lgb, X_test_lgb, y_resid_up, LGB_PARAMS, False),
            ]:
                if is_cb:
                    m = CatBoostRegressor(loss_function='RMSE', random_seed=seed, verbose=0,
                                          iterations=3000, early_stopping_rounds=100, **params)
                    m.fit(Xtr.iloc[tr], y[tr], eval_set=(Xva.iloc[va], y[va]), cat_features=cat_idx)
                else:
                    m = lgb.LGBMRegressor(objective='regression', metric='rmse', verbose=-1,
                                           random_state=seed, n_estimators=3000, **params)
                    m.fit(Xtr.iloc[tr], y[tr], eval_set=[(Xva.iloc[va], y[va])],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
                oof_r.setdefault(nm, np.zeros(n_orig))[va] = m.predict(Xva.iloc[va])
                tpred_r[nm] = tpred_r.get(nm, np.zeros(n_test)) + m.predict(Xte) / 5

            sc = StandardScaler()
            Xstr = sc.fit_transform(X_cb.iloc[tr]); Xsva = sc.transform(X_cb.iloc[va]); Xste = sc.transform(X_test_cb)
            m = ExtraTreesRegressor(n_estimators=500, max_depth=12, min_samples_leaf=10, random_state=seed, n_jobs=-1)
            m.fit(Xstr, y_resid[tr])
            oof_r.setdefault('et_r', np.zeros(n_orig))[va] = m.predict(Xsva)
            tpred_r['et_r'] = tpred_r.get('et_r', np.zeros(n_test)) + m.predict(Xste) / 5

            m = lgb.LGBMRegressor(objective='regression', metric='rmse', verbose=-1,
                                   random_state=seed, n_estimators=3000, **LGB_ET_PARAMS)
            m.fit(X_lgb.iloc[tr], y_resid[tr], eval_set=[(X_lgb.iloc[va], y_resid[va])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
            oof_r.setdefault('lgbet_r', np.zeros(n_orig))[va] = m.predict(X_lgb.iloc[va])
            tpred_r['lgbet_r'] = tpred_r.get('lgbet_r', np.zeros(n_test)) + m.predict(X_test_lgb) / 5

        for k in ['cb_ru', 'lgb_ru']:
            oof_r[k] *= area_tr; tpred_r[k] *= area_te

        rnames = ['cb_r', 'lgb_r', 'cb_ru', 'lgb_ru', 'et_r', 'lgbet_r']
        st_tr = np.column_stack([oof_r[k] for k in rnames])
        st_te = np.column_stack([tpred_r[k] for k in rnames])
        best_rmse, best_test, best_oof = float('inf'), None, None
        kf_meta = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
        for alpha in [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]:
            s_oof, s_test = np.zeros(n_orig), np.zeros(n_test)
            for tr, va in kf_meta.split(st_tr):
                meta = Ridge(alpha=alpha); meta.fit(st_tr[tr], y_resid[tr])
                s_oof[va] = meta.predict(st_tr[va]); s_test += meta.predict(st_te) / N_SPLITS
            rmse = np.sqrt(np.mean((s_oof - y_resid) ** 2))
            if rmse < best_rmse: best_rmse = rmse; best_test = s_test.copy(); best_oof = s_oof.copy()
        seed_tests.append(best_test)
        seed_oofs.append(best_oof)
    return np.mean(seed_tests, axis=0), np.mean(seed_oofs, axis=0)

# =============================================
# PART 1: 전략 53 (PL2 없음, 4시드)
# =============================================
print("=" * 60)
print("PART 1: 전략 53 (PL2 없음, 4시드)")
print("=" * 60)

y_log = np.log1p(y_true_orig)
y_raw = y_true_orig.copy()
y_up_log = np.log1p(y_true_orig / area_train)
y_up_raw = (y_true_orig / area_train).astype(float)

seed_tests_53, seed_oofs_53 = [], []
for seed in [42, 123, 456, 789]:
    print(f"\n--- Seed {seed} ---")
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    oof, tpred = train_12models(X_cb, X_test_cb, X_lgb, X_test_lgb, cat_idx,
                                 y_log, y_raw, y_up_log, y_up_raw,
                                 area_train, area_test, n_orig, kf, seed, f"S53-{seed}")
    kf_meta = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    test_pred, rmse, oof_pred = ridge_stack(oof, tpred, y_true_orig, n_orig, len(test_orig), kf_meta)
    print(f"  Seed {seed}: OOF {rmse:,.0f}")
    seed_tests_53.append(test_pred)
    seed_oofs_53.append(oof_pred)

pred_53 = np.mean(seed_tests_53, axis=0)
oof_53 = np.mean(seed_oofs_53, axis=0)

# =============================================
# PART 2: 전략 47 (PL2 + 12모델)
# =============================================
print(f"\n{'=' * 60}")
print("PART 2: 전략 47 (PL2 + 12모델)")
print("=" * 60)

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
oof_s1, tpred_s1 = train_12models(X_cb, X_test_cb, X_lgb, X_test_lgb, cat_idx,
                                    y_log, y_raw, y_up_log, y_up_raw,
                                    area_train, area_test, n_orig, kf, 42, "S47-PL")

base4_tpred = np.column_stack([tpred_s1[k] for k in MODELS])
pseudo_labels = base4_tpred.mean(axis=1)
model_disagreement = np.std(base4_tpred, axis=1) / np.mean(base4_tpred, axis=1)
confidence = 1 - model_disagreement
mask = confidence >= np.percentile(confidence, 50)

test_selected = test_orig[mask].copy()
test_selected['Target'] = pseudo_labels[mask]
train_aug = pd.concat([train_orig, test_selected], ignore_index=True)
area_train_aug = train_aug['Exclusive_Area'].values

print(f"\n  PL2: {mask.sum()}건 채택")

train_cb2, test_cb2, train_lgb2, test_lgb2 = prepare_data(train_aug, test_orig)
X_cb2 = train_cb2.drop(columns=['Target'])
X_test_cb2 = test_cb2
cat_idx2 = [X_cb2.columns.get_loc(c) for c in CAT_FEATURES]
X_lgb2 = train_lgb2.drop(columns=['Target'])
X_test_lgb2 = test_lgb2

y_log_aug = np.log1p(train_aug['Target'])
y_raw_aug = train_aug['Target'].values
y_up_log_aug = np.log1p(train_aug['Target'].values / area_train_aug)
y_up_raw_aug = (train_aug['Target'].values / area_train_aug).astype(float)

kf2 = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
oof_47, tpred_47 = train_12models(X_cb2, X_test_cb2, X_lgb2, X_test_lgb2, cat_idx2,
                                    y_log_aug, y_raw_aug, y_up_log_aug, y_up_raw_aug,
                                    area_train_aug, area_test, n_orig, kf2, 42, "S47")

kf_meta = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
pred_47, rmse_47, oof_47_pred = ridge_stack(oof_47, tpred_47, y_true_orig, n_orig, len(test_orig), kf_meta)
print(f"\n  전략47 OOF: {rmse_47:,.0f}")

# =============================================
# 전략 56 예측 (GBDT + GTR)
# =============================================
print(f"\n{'=' * 60}")
print("전략 56: GBDT + GTR")
print("=" * 60)

pred_56 = (pred_53 * BLEND_W53 + pred_47 * (1 - BLEND_W53)) * trend_correction
# OOF는 in-sample(미래 시점 아님)이므로 트렌드 보정 없이 그대로 비교
oof_56 = oof_53 * BLEND_W53 + oof_47_pred * (1 - BLEND_W53)
print(f"  pred_56 mean: {pred_56.mean():,.0f}, OOF RMSE: {np.sqrt(np.mean((oof_56 - y_true_orig) ** 2)):,.0f}")

# =============================================
# 전략 63: Per-Gu Ridge Skeleton + GBDT Residual
# =============================================
print(f"\n{'=' * 60}")
print("전략 63: Per-Gu Ridge Skeleton + GBDT Residual")
print("=" * 60)

train_skel = add_feature_engineering(base_preprocess(train_orig))
test_skel = add_feature_engineering(base_preprocess(test_orig))

gu_train = train_skel['Gu'].values
gu_test = test_skel['Gu'].values
num_cols_skel = [c for c in train_skel.columns if c not in ['Gu', 'Dong', 'Target']]

ohe_dong = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
ohe_dong.fit(pd.concat([train_skel[['Dong']], test_skel[['Dong']]]))
dong_tr = ohe_dong.transform(train_skel[['Dong']])
dong_te = ohe_dong.transform(test_skel[['Dong']])

X_skel_raw_tr = np.hstack([train_skel[num_cols_skel].values.astype(float), dong_tr])
X_skel_raw_te = np.hstack([test_skel[num_cols_skel].values.astype(float), dong_te])

skeleton_oof_pergu = np.zeros(n_orig)
skeleton_test_pergu = np.zeros(len(test_orig))

for gu in np.unique(gu_train):
    tr_mask = gu_train == gu
    te_mask = gu_test == gu
    n_gu = tr_mask.sum()
    n_splits_gu = min(N_SPLITS, n_gu)
    if n_splits_gu < 2:
        continue

    scaler_gu = StandardScaler()
    X_gu = scaler_gu.fit_transform(X_skel_raw_tr[tr_mask])
    y_gu = y_log[tr_mask]
    y_true_gu = y_true_orig[tr_mask]

    kf_gu = KFold(n_splits=n_splits_gu, shuffle=True, random_state=42)
    best_alpha_gu, best_rmse_gu = None, float('inf')
    for alpha in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
        oof_tmp = np.zeros(n_gu)
        for tr, va in kf_gu.split(X_gu):
            m = Ridge(alpha=alpha); m.fit(X_gu[tr], y_gu[tr])
            oof_tmp[va] = m.predict(X_gu[va])
        rmse = np.sqrt(np.mean((np.expm1(oof_tmp) - y_true_gu) ** 2))
        if rmse < best_rmse_gu: best_rmse_gu = rmse; best_alpha_gu = alpha

    idx_tr = np.where(tr_mask)[0]
    for tr, va in kf_gu.split(X_gu):
        m = Ridge(alpha=best_alpha_gu); m.fit(X_gu[tr], y_gu[tr])
        skeleton_oof_pergu[idx_tr[va]] = np.expm1(m.predict(X_gu[va]))

    if te_mask.sum() > 0:
        X_gu_te = scaler_gu.transform(X_skel_raw_te[te_mask])
        m_full = Ridge(alpha=best_alpha_gu); m_full.fit(X_gu, y_gu)
        skeleton_test_pergu[te_mask] = np.expm1(m_full.predict(X_gu_te))

    print(f"  Gu={gu}: n={n_gu}, alpha={best_alpha_gu}, RMSE={best_rmse_gu:,.0f}")

print(f"  Per-Gu Skeleton OOF RMSE: {np.sqrt(np.mean((skeleton_oof_pergu - y_true_orig) ** 2)):,.0f}")

y_resid_pergu = (y_true_orig - skeleton_oof_pergu).astype(float)
y_resid_up_pergu = (y_resid_pergu / area_train).astype(float)

resid_test_pergu, resid_oof_pergu = residual_gbdt(
    X_cb, X_test_cb, X_lgb, X_test_lgb, cat_idx, y_resid_pergu, y_resid_up_pergu,
    area_train, area_test, n_orig, len(test_orig), "PerGu")

pred_63 = skeleton_test_pergu + resid_test_pergu
oof_63 = skeleton_oof_pergu + resid_oof_pergu
print(f"  pred_63 mean: {pred_63.mean():,.0f}, OOF RMSE: {np.sqrt(np.mean((oof_63 - y_true_orig) ** 2)):,.0f}")

# =============================================
# 전략 69: One-Hot Gu/Dong Skeleton + GBDT Residual
# =============================================
print(f"\n{'=' * 60}")
print("전략 69: One-Hot Skeleton + GBDT Residual")
print("=" * 60)

ohe_gd = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
ohe_gd.fit(pd.concat([train_skel[['Gu', 'Dong']], test_skel[['Gu', 'Dong']]]))
X_skel_oh_tr = np.hstack([train_skel[num_cols_skel].values.astype(float), ohe_gd.transform(train_skel[['Gu', 'Dong']])])
X_skel_oh_te = np.hstack([test_skel[num_cols_skel].values.astype(float), ohe_gd.transform(test_skel[['Gu', 'Dong']])])

scaler_skel = StandardScaler()
X_skel_oh_all = scaler_skel.fit_transform(X_skel_oh_tr)
X_skel_oh_test = scaler_skel.transform(X_skel_oh_te)

skeleton_oof_oh = np.zeros(n_orig)
kf_skel = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
best_alpha_skel, best_rmse_skel = None, float('inf')
for alpha in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
    oof_tmp = np.zeros(n_orig)
    for tr, va in kf_skel.split(X_skel_oh_all):
        m = Ridge(alpha=alpha); m.fit(X_skel_oh_all[tr], y_log[tr])
        oof_tmp[va] = m.predict(X_skel_oh_all[va])
    rmse = np.sqrt(np.mean((np.expm1(oof_tmp) - y_true_orig) ** 2))
    if rmse < best_rmse_skel: best_rmse_skel = rmse; best_alpha_skel = alpha

for tr, va in kf_skel.split(X_skel_oh_all):
    m = Ridge(alpha=best_alpha_skel); m.fit(X_skel_oh_all[tr], y_log[tr])
    skeleton_oof_oh[va] = np.expm1(m.predict(X_skel_oh_all[va]))
m_full = Ridge(alpha=best_alpha_skel); m_full.fit(X_skel_oh_all, y_log)
skeleton_test_oh = np.expm1(m_full.predict(X_skel_oh_test))
print(f"  One-Hot Skeleton OOF RMSE: {best_rmse_skel:,.0f} (alpha={best_alpha_skel})")

y_resid_oh = (y_true_orig - skeleton_oof_oh).astype(float)
y_resid_up_oh = (y_resid_oh / area_train).astype(float)

resid_test_oh, resid_oof_oh = residual_gbdt(
    X_cb, X_test_cb, X_lgb, X_test_lgb, cat_idx, y_resid_oh, y_resid_up_oh,
    area_train, area_test, n_orig, len(test_orig), "OneHot")

pred_69 = skeleton_test_oh + resid_test_oh
oof_69 = skeleton_oof_oh + resid_oof_oh
print(f"  pred_69 mean: {pred_69.mean():,.0f}, OOF RMSE: {np.sqrt(np.mean((oof_69 - y_true_orig) ** 2)):,.0f}")

# =============================================
# 신규: UnitPrice(평당가) Per-Gu Skeleton + GBDT Residual
# =============================================
print(f"\n{'=' * 60}")
print("신규: UnitPrice Per-Gu Skeleton + GBDT Residual")
print("=" * 60)

skeleton_oof_up = np.zeros(n_orig)
skeleton_test_up = np.zeros(len(test_orig))

for gu in np.unique(gu_train):
    tr_mask = gu_train == gu
    te_mask = gu_test == gu
    n_gu = tr_mask.sum()
    n_splits_gu = min(N_SPLITS, n_gu)
    if n_splits_gu < 2:
        continue

    scaler_gu = StandardScaler()
    X_gu = scaler_gu.fit_transform(X_skel_raw_tr[tr_mask])
    y_gu = y_up_log[tr_mask]
    y_true_gu = y_true_orig[tr_mask]
    area_gu = area_train[tr_mask]

    kf_gu = KFold(n_splits=n_splits_gu, shuffle=True, random_state=42)
    best_alpha_gu, best_rmse_gu = None, float('inf')
    for alpha in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
        oof_tmp = np.zeros(n_gu)
        for tr, va in kf_gu.split(X_gu):
            m = Ridge(alpha=alpha); m.fit(X_gu[tr], y_gu[tr])
            oof_tmp[va] = m.predict(X_gu[va])
        rmse = np.sqrt(np.mean((np.expm1(oof_tmp) * area_gu - y_true_gu) ** 2))
        if rmse < best_rmse_gu: best_rmse_gu = rmse; best_alpha_gu = alpha

    idx_tr = np.where(tr_mask)[0]
    for tr, va in kf_gu.split(X_gu):
        m = Ridge(alpha=best_alpha_gu); m.fit(X_gu[tr], y_gu[tr])
        skeleton_oof_up[idx_tr[va]] = np.expm1(m.predict(X_gu[va])) * area_gu[va]

    if te_mask.sum() > 0:
        X_gu_te = scaler_gu.transform(X_skel_raw_te[te_mask])
        m_full = Ridge(alpha=best_alpha_gu); m_full.fit(X_gu, y_gu)
        skeleton_test_up[te_mask] = np.expm1(m_full.predict(X_gu_te)) * area_test[te_mask]

print(f"  UnitPrice Skeleton OOF RMSE: {np.sqrt(np.mean((skeleton_oof_up - y_true_orig) ** 2)):,.0f}")

y_resid_up_sk = (y_true_orig - skeleton_oof_up).astype(float)
y_resid_up_up = (y_resid_up_sk / area_train).astype(float)

resid_test_up, resid_oof_up = residual_gbdt(
    X_cb, X_test_cb, X_lgb, X_test_lgb, cat_idx, y_resid_up_sk, y_resid_up_up,
    area_train, area_test, n_orig, len(test_orig), "UnitPrice")

pred_up = skeleton_test_up + resid_test_up
oof_up = skeleton_oof_up + resid_oof_up
print(f"  pred_up mean: {pred_up.mean():,.0f}, OOF RMSE: {np.sqrt(np.mean((oof_up - y_true_orig) ** 2)):,.0f}")

# =============================================
# 4-WAY 블렌딩: 가중치는 OOT(시간분리 holdout) 검증으로 결정
# =============================================
print(f"\n{'=' * 60}")
print("4-way 블렌딩")
print("=" * 60)

for nm_a, nm_b, o_a, o_b in [('56','63',oof_56,oof_63), ('56','69',oof_56,oof_69), ('56','up',oof_56,oof_up),
                              ('63','69',oof_63,oof_69), ('63','up',oof_63,oof_up), ('69','up',oof_69,oof_up)]:
    print(f"  {nm_a}↔{nm_b} 상관: {np.corrcoef(o_a, o_b)[0,1]:.4f}")

# naive in-sample OOF 그리드서치는 90:10:0(69~up 무시)을 골랐으나, 이는 이미 검증된 사실
# (56단독 Public 2,086.6 > 56+63 60:40 Public 2,039.7)과 정면으로 상충 -> 신뢰 불가.
# 대신 oot_split(holdout_months=3)으로 56/63/69/up을 따로 학습해 실제 라벨이 있는 미래 3개월에
# 직접 평가: 56단독 2,631 -> 60:40 2,601 -> 50:30:20(3-way) 2,595.1 -> 40:15:25:20(4-way) 2,587.1
# 순서대로 일관되게 개선되고(사니티체크 통과), 상위 15개 조합도 2,587~2,588로 촘촘해 과적합 아님.
w56, w63, w69, wup = 0.4, 0.15, 0.25, 0.2
print(f"  가중치(OOT 검증): 56×{w56:.0%} + 63×{w63:.0%} + 69×{w69:.0%} + up×{wup:.0%}")

baseline_3way_oof = np.sqrt(np.mean((0.5 * oof_56 + 0.3 * oof_63 + 0.2 * oof_69 - y_true_orig) ** 2))
chosen_oof = np.sqrt(np.mean((w56 * oof_56 + w63 * oof_63 + w69 * oof_69 + wup * oof_up - y_true_orig) ** 2))
print(f"  (참고, 신뢰 안 함) 기존 3-way(50:30:20) OOF RMSE: {baseline_3way_oof:,.0f}")
print(f"  (참고, 신뢰 안 함) 선택 가중치 OOF RMSE: {chosen_oof:,.0f}")

final_pred = w56 * pred_56 + w63 * pred_63 + w69 * pred_69 + wup * pred_up
print(f"  Final mean: {final_pred.mean():,.0f}")

sub = sample_sub.copy()
sub['Target'] = final_pred
sub.to_csv(f'{OUTPUT_DIR}/submission.csv', index=False)
print(f"\n제출 파일 저장 완료: {OUTPUT_DIR}/submission.csv")
