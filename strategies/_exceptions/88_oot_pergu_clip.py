"""
88 OOT PER-GU ADAPTIVE CLIP: 전략84의 전역 std Winsorize(±3.0*std)를 Gu별 std 기반으로 교체.
지금은 전체 잔차 std 하나로 전 구를 동일하게 클립하는데, 성동/용산처럼 이상치가 몰린 구와
나머지 구를 같은 기준으로 자르는 게 부정확할 수 있음. Gu별로 자기 구의 잔차 std*3.0을 기준으로
클립하도록 바꿔서 OOT(holdout 3개월)로 검증.

pred_56은 클립 방식과 무관하므로 1회 계산 후 npz로 캐시(다음 실험에서 재사용, ~500s 절약).
전역 클립 베이스라인(전략86 결과): 63=2,690 / 69=2,640 / 50:30:20 블렌드=2,590.
전략87 실패로 "OOT 마진이 좁으면 신뢰 불가"가 확인됐으므로, 여기서도 마진이 충분히 넓은지 확인 후
제출 여부를 판단할 것 (좁으면 제출 보류).
"""
import os
import sys
import time
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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from utils import oot_split

_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
_SCRATCH = '/private/tmp/claude-501/-Users-leewonseok-playground-scu-5th-competition/ec25c96e-686e-46b1-9358-0de0b8066708/scratchpad'
CACHE_PATH = os.path.join(_SCRATCH, 'oot_cache_pred56.npz')

N_SPLITS = 5
CAT_FEATURES = ['Gu', 'Dong']
MODELS = ['cb_log', 'cb_raw', 'lgb_log', 'lgb_raw']
SEEDS = [42, 123, 456, 789]

t0 = time.time()

train_full = pd.read_csv(f'{_DIR}/seoul_real_estate_train.csv')
tr_idx, va_idx, cutoff_ym = oot_split(train_full, holdout_months=3)
train_orig = train_full.loc[tr_idx].reset_index(drop=True)
test_orig_labeled = train_full.loc[va_idx].reset_index(drop=True)
y_true_orig = train_orig['Target'].values
y_true_val = test_orig_labeled['Target'].values
test_orig = test_orig_labeled.drop(columns=['Target'])
n_orig = len(train_orig)
area_train = train_orig['Exclusive_Area'].values
area_test = test_orig['Exclusive_Area'].values

print(f"OOT 분할: cutoff_ym={cutoff_ym}, train={n_orig}건, val(holdout)={len(test_orig)}건")

last_train_ym = train_orig['Transaction_YearMonth'].max()
last_train_seq = (last_train_ym // 100 - 2024) * 12 + last_train_ym % 100
gu_growth = {}
for gu in train_orig['Gu'].unique():
    monthly = train_orig[train_orig['Gu'] == gu].groupby('Transaction_YearMonth')['Target'].mean()
    gu_growth[gu] = monthly.pct_change().dropna().mean()
test_seq = (test_orig['Transaction_YearMonth'] // 100 - 2024) * 12 + test_orig['Transaction_YearMonth'] % 100
months_ahead = test_seq.values - last_train_seq
trend_correction = np.array([(1 + gu_growth[gu]) ** m for gu, m in zip(test_orig['Gu'], months_ahead)])

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
        print(f"  [{label}] Fold {fold+1}/5  ({time.time()-t0:,.0f}s)")

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
    seed_tests, seed_oofs = [], []
    for seed in SEEDS:
        print(f"  --- {label} Resid Seed {seed}  ({time.time()-t0:,.0f}s) ---")
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

y_log = np.log1p(y_true_orig)
y_raw = y_true_orig.copy()
y_up_log = np.log1p(y_true_orig / area_train)
y_up_raw = (y_true_orig / area_train).astype(float)

# =============================================
# pred_56 (=pred_47*GTR): 클립 방식과 무관 -> 캐시 재사용
# =============================================
if os.path.exists(CACHE_PATH):
    cache = np.load(CACHE_PATH)
    pred_56 = cache['pred_56']
    print(f"\n  pred_56 캐시 로드 완료 ({time.time()-t0:,.0f}s)")
else:
    print(f"\n{'=' * 60}\nPART: 전략47 (PL2 + 12모델, 4시드)  ({time.time()-t0:,.0f}s)\n{'=' * 60}")
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

    print(f"  PL2: {mask.sum()}건 채택")

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

    seed_tests_47 = []
    for seed47 in SEEDS:
        print(f"  --- S47 Seed {seed47}  ({time.time()-t0:,.0f}s) ---")
        kf2 = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed47)
        oof_47_s, tpred_47_s = train_12models(X_cb2, X_test_cb2, X_lgb2, X_test_lgb2, cat_idx2,
                                              y_log_aug, y_raw_aug, y_up_log_aug, y_up_raw_aug,
                                              area_train_aug, area_test, n_orig, kf2, seed47, f"S47-{seed47}")
        kf_meta = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
        pred_47_s, rmse_47_s, _ = ridge_stack(oof_47_s, tpred_47_s, y_true_orig, n_orig, len(test_orig), kf_meta)
        print(f"    Seed {seed47}: OOF {rmse_47_s:,.0f}")
        seed_tests_47.append(pred_47_s)

    pred_47 = np.mean(seed_tests_47, axis=0)
    pred_56 = pred_47 * trend_correction
    os.makedirs(_SCRATCH, exist_ok=True)
    np.savez(CACHE_PATH, pred_56=pred_56, y_true_val=y_true_val)
    print(f"  pred_56 캐시 저장: {CACHE_PATH}")

val_rmse_56 = np.sqrt(np.mean((pred_56 - y_true_val) ** 2))
print(f"  전략56 OOT-val RMSE: {val_rmse_56:,.0f}")

# =============================================
# 전략 63: Per-Gu Ridge Skeleton (클립 방식과 무관, 항상 새로 계산)
# =============================================
print(f"\n{'=' * 60}\n전략63: Per-Gu Skeleton  ({time.time()-t0:,.0f}s)\n{'=' * 60}")

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

print(f"  Per-Gu Skeleton OOF RMSE: {np.sqrt(np.mean((skeleton_oof_pergu - y_true_orig) ** 2)):,.0f}")

y_resid_pergu_raw = (y_true_orig - skeleton_oof_pergu).astype(float)

# === Gu별 적응형 클립 (전역 std 대신 각 구의 std*3.0 사용) ===
y_resid_pergu_adaptive = y_resid_pergu_raw.copy()
gu_stds = {}
for gu in np.unique(gu_train):
    gmask = gu_train == gu
    gu_std = float(np.std(y_resid_pergu_raw[gmask]))
    gu_stds[gu] = gu_std
    y_resid_pergu_adaptive[gmask] = np.clip(y_resid_pergu_raw[gmask], -3.0 * gu_std, 3.0 * gu_std)
print(f"  Gu별 std: {', '.join(f'{g}={s:.0f}' for g, s in sorted(gu_stds.items(), key=lambda x: -x[1])[:5])} ...")
n_clipped_adaptive = np.sum(y_resid_pergu_adaptive != y_resid_pergu_raw)
print(f"  전역 클립 대비 Gu별 적응형 클립: clipped={n_clipped_adaptive}건 (전역 기준은 전략84 참고)")

y_resid_up_pergu_adaptive = (y_resid_pergu_adaptive / area_train).astype(float)

resid_test_pergu_a, _ = residual_gbdt(
    X_cb, X_test_cb, X_lgb, X_test_lgb, cat_idx, y_resid_pergu_adaptive, y_resid_up_pergu_adaptive,
    area_train, area_test, n_orig, len(test_orig), "PerGu-Adaptive")

pred_63_adaptive = skeleton_test_pergu + resid_test_pergu_a
val_rmse_63_adaptive = np.sqrt(np.mean((pred_63_adaptive - y_true_val) ** 2))
print(f"  전략63(Gu별 적응형 클립) OOT-val RMSE: {val_rmse_63_adaptive:,.0f}  (전역클립 베이스라인: 2,690)")

# =============================================
# 전략 69: One-Hot Skeleton (Gu 라벨은 있으니 잔차도 Gu별로 그룹핑해서 클립)
# =============================================
print(f"\n{'=' * 60}\n전략69: One-Hot Skeleton  ({time.time()-t0:,.0f}s)\n{'=' * 60}")

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

y_resid_oh_raw = (y_true_orig - skeleton_oof_oh).astype(float)
y_resid_oh_adaptive = y_resid_oh_raw.copy()
for gu in np.unique(gu_train):
    gmask = gu_train == gu
    gu_std = float(np.std(y_resid_oh_raw[gmask]))
    y_resid_oh_adaptive[gmask] = np.clip(y_resid_oh_raw[gmask], -3.0 * gu_std, 3.0 * gu_std)
n_clipped_oh = np.sum(y_resid_oh_adaptive != y_resid_oh_raw)
print(f"  Gu별 적응형 클립(One-Hot): clipped={n_clipped_oh}건")

y_resid_up_oh_adaptive = (y_resid_oh_adaptive / area_train).astype(float)

resid_test_oh_a, _ = residual_gbdt(
    X_cb, X_test_cb, X_lgb, X_test_lgb, cat_idx, y_resid_oh_adaptive, y_resid_up_oh_adaptive,
    area_train, area_test, n_orig, len(test_orig), "OneHot-Adaptive")

pred_69_adaptive = skeleton_test_oh + resid_test_oh_a
val_rmse_69_adaptive = np.sqrt(np.mean((pred_69_adaptive - y_true_val) ** 2))
print(f"  전략69(Gu별 적응형 클립) OOT-val RMSE: {val_rmse_69_adaptive:,.0f}  (전역클립 베이스라인: 2,640)")

# =============================================
# 50:30:20 고정 비율로 블렌드 비교 (비율 재탐색은 이미 실패로 종료됨)
# =============================================
print(f"\n{'=' * 60}\n50:30:20 블렌드 비교  ({time.time()-t0:,.0f}s)\n{'=' * 60}")

blend_adaptive = 0.5 * pred_56 + 0.3 * pred_63_adaptive + 0.2 * pred_69_adaptive
val_rmse_blend_adaptive = np.sqrt(np.mean((blend_adaptive - y_true_val) ** 2))
print(f"  Gu별 적응형 클립 50:30:20 블렌드 OOT-val RMSE: {val_rmse_blend_adaptive:,.0f}  (전역클립 베이스라인: 2,590)")
print(f"  개선 마진: {2590 - val_rmse_blend_adaptive:+.1f}")

print(f"\n총 소요시간: {time.time()-t0:,.0f}초")
