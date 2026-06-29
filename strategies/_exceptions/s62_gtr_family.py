"""
62 GTR FAMILY TEST
파이프라인 1회 실행 → pred_53, pred_47 저장
→ GTR 5종 × 블렌딩 5종 = 25개 조합 비교 (수초)
"""
import os, numpy as np, pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb
import warnings; warnings.filterwarnings('ignore')

_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
INPUT_DIR = _DIR
OUTPUT_DIR = _DIR
train_orig = pd.read_csv(f'{INPUT_DIR}/seoul_real_estate_train.csv')
test_orig = pd.read_csv(f'{INPUT_DIR}/seoul_real_estate_test.csv')
sample_sub = pd.read_csv(f'{INPUT_DIR}/sample_submission.csv')
y_true_orig = train_orig['Target'].values
n_orig = len(train_orig)
area_train = train_orig['Exclusive_Area'].values
area_test = test_orig['Exclusive_Area'].values
N_SPLITS = 5
CAT_FEATURES = ['Gu', 'Dong']
MODELS = ['cb_log', 'cb_raw', 'lgb_log', 'lgb_raw']

# === 전처리 (56과 동일) ===
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
        le.fit(list(train_df[col].astype(str)) + list(test_df[col].astype(str)))
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
X_cb = train_cb.drop(columns=['Target']); X_test_cb = test_cb
cat_idx = [X_cb.columns.get_loc(c) for c in CAT_FEATURES]
X_lgb = train_lgb.drop(columns=['Target']); X_test_lgb = test_lgb

def train_12models(X_cb, X_test_cb, X_lgb, X_test_lgb, cat_idx,
                   y_log, y_raw, y_up_log, y_up_raw, area_tr, area_te,
                   n_orig, kf, seed, label):
    all_oof, all_tpred = {}, {}
    for fold, (tr, va) in enumerate(kf.split(X_cb)):
        print(f"  [{label}] Fold {fold+1}/5")
        m = CatBoostRegressor(loss_function='RMSE', random_seed=seed, verbose=0, iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
        m.fit(X_cb.iloc[tr], y_log[tr] if isinstance(y_log, np.ndarray) else y_log.iloc[tr], eval_set=(X_cb.iloc[va], y_log[va] if isinstance(y_log, np.ndarray) else y_log.iloc[va]), cat_features=cat_idx)
        all_oof.setdefault('cb_log', np.zeros(len(X_cb)))[va] = np.expm1(m.predict(X_cb.iloc[va]))
        all_tpred['cb_log'] = all_tpred.get('cb_log', np.zeros(len(X_test_cb))) + np.expm1(m.predict(X_test_cb)) / 5
        m = CatBoostRegressor(loss_function='RMSE', random_seed=seed, verbose=0, iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
        m.fit(X_cb.iloc[tr], y_raw[tr], eval_set=(X_cb.iloc[va], y_raw[va]), cat_features=cat_idx)
        all_oof.setdefault('cb_raw', np.zeros(len(X_cb)))[va] = m.predict(X_cb.iloc[va])
        all_tpred['cb_raw'] = all_tpred.get('cb_raw', np.zeros(len(X_test_cb))) + m.predict(X_test_cb) / 5
        m = lgb.LGBMRegressor(objective='regression', metric='rmse', verbose=-1, random_state=seed, n_estimators=3000, **LGB_PARAMS)
        m.fit(X_lgb.iloc[tr], y_log[tr] if isinstance(y_log, np.ndarray) else y_log.iloc[tr], eval_set=[(X_lgb.iloc[va], y_log[va] if isinstance(y_log, np.ndarray) else y_log.iloc[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
        all_oof.setdefault('lgb_log', np.zeros(len(X_cb)))[va] = np.expm1(m.predict(X_lgb.iloc[va]))
        all_tpred['lgb_log'] = all_tpred.get('lgb_log', np.zeros(len(X_test_cb))) + np.expm1(m.predict(X_test_lgb)) / 5
        m = lgb.LGBMRegressor(objective='regression', metric='rmse', verbose=-1, random_state=seed, n_estimators=3000, **LGB_PARAMS)
        m.fit(X_lgb.iloc[tr], y_raw[tr], eval_set=[(X_lgb.iloc[va], y_raw[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
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
        m.fit(X_lgb.iloc[tr], y_up_log[tr], eval_set=[(X_lgb.iloc[va], y_up_log[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
        all_oof.setdefault('lgb_up_log', np.zeros(len(X_cb)))[va] = np.expm1(m.predict(X_lgb.iloc[va]))
        all_tpred['lgb_up_log'] = all_tpred.get('lgb_up_log', np.zeros(len(X_test_cb))) + np.expm1(m.predict(X_test_lgb)) / 5
        m = lgb.LGBMRegressor(objective='regression', metric='rmse', verbose=-1, random_state=seed, n_estimators=3000, **LGB_PARAMS)
        m.fit(X_lgb.iloc[tr], y_up_raw[tr], eval_set=[(X_lgb.iloc[va], y_up_raw[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
        all_oof.setdefault('lgb_up_raw', np.zeros(len(X_cb)))[va] = m.predict(X_lgb.iloc[va])
        all_tpred['lgb_up_raw'] = all_tpred.get('lgb_up_raw', np.zeros(len(X_test_cb))) + m.predict(X_test_lgb) / 5
        scaler = StandardScaler()
        X_sc_tr = scaler.fit_transform(X_cb.iloc[tr]); X_sc_va = scaler.transform(X_cb.iloc[va]); X_sc_te = scaler.transform(X_test_cb)
        m = ExtraTreesRegressor(n_estimators=500, max_depth=12, min_samples_leaf=10, random_state=seed, n_jobs=-1)
        m.fit(X_sc_tr, y_log[tr] if isinstance(y_log, np.ndarray) else y_log.iloc[tr])
        all_oof.setdefault('et_log', np.zeros(len(X_cb)))[va] = np.expm1(m.predict(X_sc_va))
        all_tpred['et_log'] = all_tpred.get('et_log', np.zeros(len(X_test_cb))) + np.expm1(m.predict(X_sc_te)) / 5
        m = ExtraTreesRegressor(n_estimators=500, max_depth=12, min_samples_leaf=10, random_state=seed, n_jobs=-1)
        m.fit(X_sc_tr, y_raw[tr])
        all_oof.setdefault('et_raw', np.zeros(len(X_cb)))[va] = m.predict(X_sc_va)
        all_tpred['et_raw'] = all_tpred.get('et_raw', np.zeros(len(X_test_cb))) + m.predict(X_sc_te) / 5
        m = lgb.LGBMRegressor(objective='regression', metric='rmse', verbose=-1, random_state=seed, n_estimators=3000, **LGB_ET_PARAMS)
        m.fit(X_lgb.iloc[tr], y_log[tr] if isinstance(y_log, np.ndarray) else y_log.iloc[tr], eval_set=[(X_lgb.iloc[va], y_log[va] if isinstance(y_log, np.ndarray) else y_log.iloc[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
        all_oof.setdefault('lgbet_log', np.zeros(len(X_cb)))[va] = np.expm1(m.predict(X_lgb.iloc[va]))
        all_tpred['lgbet_log'] = all_tpred.get('lgbet_log', np.zeros(len(X_test_cb))) + np.expm1(m.predict(X_test_lgb)) / 5
        m = lgb.LGBMRegressor(objective='regression', metric='rmse', verbose=-1, random_state=seed, n_estimators=3000, **LGB_ET_PARAMS)
        m.fit(X_lgb.iloc[tr], y_raw[tr], eval_set=[(X_lgb.iloc[va], y_raw[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
        all_oof.setdefault('lgbet_raw', np.zeros(len(X_cb)))[va] = m.predict(X_lgb.iloc[va])
        all_tpred['lgbet_raw'] = all_tpred.get('lgbet_raw', np.zeros(len(X_test_cb))) + m.predict(X_test_lgb) / 5
    for k in ['cb_up_log', 'cb_up_raw', 'lgb_up_log', 'lgb_up_raw']:
        all_oof[k] = all_oof[k][:n_orig] * area_tr[:n_orig]
        all_tpred[k] = all_tpred[k] * area_te
    for k in [kk for kk in all_oof if kk not in ['cb_up_log', 'cb_up_raw', 'lgb_up_log', 'lgb_up_raw']]:
        all_oof[k] = all_oof[k][:n_orig]
    return all_oof, all_tpred

def ridge_stack(all_oof, all_tpred, y_true, n_orig, n_test, kf_meta):
    base_12 = ['cb_log', 'cb_raw', 'lgb_log', 'lgb_raw', 'cb_up_log', 'cb_up_raw',
               'lgb_up_log', 'lgb_up_raw', 'et_log', 'et_raw', 'lgbet_log', 'lgbet_raw']
    st_tr = np.column_stack([all_oof[k] for k in base_12])
    st_te = np.column_stack([all_tpred[k] for k in base_12])
    best_rmse, best_test = float('inf'), None
    for alpha in [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]:
        s_oof, s_test = np.zeros(n_orig), np.zeros(n_test)
        for tr, va in kf_meta.split(st_tr):
            meta = Ridge(alpha=alpha); meta.fit(st_tr[tr], y_true[tr])
            s_oof[va] = meta.predict(st_tr[va]); s_test += meta.predict(st_te) / kf_meta.n_splits
        rmse = np.sqrt(np.mean((s_oof - y_true) ** 2))
        if rmse < best_rmse: best_rmse = rmse; best_test = s_test.copy()
    return best_test, best_rmse


# =============================================
# STEP 1: 파이프라인 실행 (1회만)
# =============================================
print("=" * 60)
print("STEP 1: 모델 학습 (전략 53 + 47)")
print("=" * 60)

y_log = np.log1p(y_true_orig); y_raw = y_true_orig.copy()
y_up_log = np.log1p(y_true_orig / area_train); y_up_raw = (y_true_orig / area_train).astype(float)

# 전략 53 (no-PL2, 4시드)
seed_tests_53 = []
for seed in [42, 123, 456, 789]:
    print(f"\n--- S53 Seed {seed} ---")
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    oof, tpred = train_12models(X_cb, X_test_cb, X_lgb, X_test_lgb, cat_idx,
                                 y_log, y_raw, y_up_log, y_up_raw,
                                 area_train, area_test, n_orig, kf, seed, f"S53-{seed}")
    kf_meta = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    test_pred, rmse = ridge_stack(oof, tpred, y_true_orig, n_orig, len(test_orig), kf_meta)
    print(f"  OOF {rmse:,.0f}")
    seed_tests_53.append(test_pred)
pred_53 = np.mean(seed_tests_53, axis=0)

# 전략 47 (PL2)
print(f"\n--- S47 PL2 ---")
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
oof_s1, tpred_s1 = train_12models(X_cb, X_test_cb, X_lgb, X_test_lgb, cat_idx,
                                    y_log, y_raw, y_up_log, y_up_raw,
                                    area_train, area_test, n_orig, kf, 42, "S47-PL")
base4_tpred = np.column_stack([tpred_s1[k] for k in MODELS])
pseudo_labels = base4_tpred.mean(axis=1)
confidence = 1 - np.std(base4_tpred, axis=1) / np.mean(base4_tpred, axis=1)
mask = confidence >= np.percentile(confidence, 50)
test_selected = test_orig[mask].copy(); test_selected['Target'] = pseudo_labels[mask]
train_aug = pd.concat([train_orig, test_selected], ignore_index=True)
area_train_aug = train_aug['Exclusive_Area'].values
print(f"  PL2: {mask.sum()}건 채택")

train_cb2, test_cb2, train_lgb2, test_lgb2 = prepare_data(train_aug, test_orig)
X_cb2 = train_cb2.drop(columns=['Target']); X_test_cb2 = test_cb2
cat_idx2 = [X_cb2.columns.get_loc(c) for c in CAT_FEATURES]
X_lgb2 = train_lgb2.drop(columns=['Target']); X_test_lgb2 = test_lgb2
y_log_aug = np.log1p(train_aug['Target']); y_raw_aug = train_aug['Target'].values
y_up_log_aug = np.log1p(train_aug['Target'].values / area_train_aug)
y_up_raw_aug = (train_aug['Target'].values / area_train_aug).astype(float)

kf2 = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
oof_47, tpred_47 = train_12models(X_cb2, X_test_cb2, X_lgb2, X_test_lgb2, cat_idx2,
                                    y_log_aug, y_raw_aug, y_up_log_aug, y_up_raw_aug,
                                    area_train_aug, area_test, n_orig, kf2, 42, "S47")
kf_meta = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
pred_47, rmse_47 = ridge_stack(oof_47, tpred_47, y_true_orig, n_orig, len(test_orig), kf_meta)
print(f"  전략47 OOF: {rmse_47:,.0f}")

print(f"\n  pred_53, pred_47 준비 완료!")


# =============================================
# STEP 2: GTR 5종 × 블렌딩 5종 = 25조합 비교
# =============================================
print(f"\n{'=' * 60}")
print("STEP 2: GTR Family × Blend Ratio 비교")
print("=" * 60)

last_train_ym = train_orig['Transaction_YearMonth'].max()
last_train_seq = (last_train_ym // 100 - 2024) * 12 + last_train_ym % 100
test_seq = (test_orig['Transaction_YearMonth'] // 100 - 2024) * 12 + test_orig['Transaction_YearMonth'] % 100
months_ahead = test_seq.values - last_train_seq

def compute_gtr(name):
    """GTR 변형별 trend_correction 벡터 반환"""
    corrections = np.ones(len(test_orig))

    if name == 'original':
        # 현재 방식: 구별 Target 평균 pct_change
        for gu in train_orig['Gu'].unique():
            monthly = train_orig[train_orig['Gu'] == gu].groupby('Transaction_YearMonth')['Target'].mean()
            g = monthly.pct_change().dropna().mean()
            mask = test_orig['Gu'] == gu
            corrections[mask] = (1 + g) ** months_ahead[mask]

    elif name == 'log_trend':
        # Log trend: exp(mean(diff(log(monthly_mean))) × months)
        for gu in train_orig['Gu'].unique():
            monthly = train_orig[train_orig['Gu'] == gu].groupby('Transaction_YearMonth')['Target'].mean()
            log_monthly = np.log(monthly)
            g = log_monthly.diff().dropna().mean()
            mask = test_orig['Gu'] == gu
            corrections[mask] = np.exp(g * months_ahead[mask])

    elif name == 'unit_price_trend':
        # Unit-price trend: Target/Area 기준 성장률
        train_tmp = train_orig.copy()
        train_tmp['UnitPrice'] = train_tmp['Target'] / train_tmp['Exclusive_Area']
        for gu in train_tmp['Gu'].unique():
            monthly = train_tmp[train_tmp['Gu'] == gu].groupby('Transaction_YearMonth')['UnitPrice'].mean()
            g = monthly.pct_change().dropna().mean()
            mask = test_orig['Gu'] == gu
            corrections[mask] = (1 + g) ** months_ahead[mask]

    elif name == 'robust_median':
        # Robust: 월별 median 기준 pct_change
        for gu in train_orig['Gu'].unique():
            monthly = train_orig[train_orig['Gu'] == gu].groupby('Transaction_YearMonth')['Target'].median()
            g = monthly.pct_change().dropna().mean()
            mask = test_orig['Gu'] == gu
            corrections[mask] = (1 + g) ** months_ahead[mask]

    elif name == 'recent_6m':
        # Recent slope: 최근 6개월만 사용
        all_yms = sorted(train_orig['Transaction_YearMonth'].unique())
        recent_yms = all_yms[-6:] if len(all_yms) >= 6 else all_yms
        recent = train_orig[train_orig['Transaction_YearMonth'].isin(recent_yms)]
        for gu in train_orig['Gu'].unique():
            gu_recent = recent[recent['Gu'] == gu]
            if len(gu_recent) < 10:
                monthly = train_orig[train_orig['Gu'] == gu].groupby('Transaction_YearMonth')['Target'].mean()
            else:
                monthly = gu_recent.groupby('Transaction_YearMonth')['Target'].mean()
            g = monthly.pct_change().dropna().mean()
            mask = test_orig['Gu'] == gu
            corrections[mask] = (1 + g) ** months_ahead[mask]

    elif name == 'global_gu_shrink':
        # Global+Gu shrink: w=0.7 × gu_trend + 0.3 × global_trend
        global_monthly = train_orig.groupby('Transaction_YearMonth')['Target'].mean()
        g_global = global_monthly.pct_change().dropna().mean()
        for gu in train_orig['Gu'].unique():
            monthly = train_orig[train_orig['Gu'] == gu].groupby('Transaction_YearMonth')['Target'].mean()
            g_gu = monthly.pct_change().dropna().mean()
            g = 0.7 * g_gu + 0.3 * g_global
            mask = test_orig['Gu'] == gu
            corrections[mask] = (1 + g) ** months_ahead[mask]

    return corrections

# GTR 종류
GTR_NAMES = ['original', 'log_trend', 'unit_price_trend', 'robust_median', 'recent_6m', 'global_gu_shrink']
BLEND_RATIOS = [0.70, 0.75, 0.80, 0.85, 0.90]

# 구별 GTR 값 출력
print("\n=== 구별 GTR 보정 계수 (2026-01 기준) ===")
print(f"{'Gu':20s}", end="")
for gname in GTR_NAMES:
    print(f" {gname:>16s}", end="")
print()
for gu in sorted(train_orig['Gu'].unique()):
    print(f"{gu:20s}", end="")
    for gname in GTR_NAMES:
        corr = compute_gtr(gname)
        gu_mask = test_orig['Gu'] == gu
        avg_corr = corr[gu_mask].mean()
        print(f" {avg_corr:16.4f}", end="")
    print()

# 25 조합 비교
print(f"\n=== GTR × Blend 비교 (예측 분포 통계) ===")
print(f"{'GTR':20s} {'Blend':>6s}  {'Mean':>10s} {'Std':>10s} {'Min':>10s} {'Max':>10s}")
print("-" * 70)

results = []
for gname in GTR_NAMES:
    gtr = compute_gtr(gname)
    for w53 in BLEND_RATIOS:
        blended = pred_53 * w53 + pred_47 * (1 - w53)
        final = blended * gtr
        results.append({
            'gtr': gname, 'w53': w53,
            'mean': final.mean(), 'std': final.std(),
            'min': final.min(), 'max': final.max(),
            'final': final, 'gtr_vec': gtr,
        })
        print(f"{gname:20s} {w53:6.0%}  {final.mean():10,.0f} {final.std():10,.0f} {final.min():10,.0f} {final.max():10,.0f}")

# 53↔47 상관
corr_53_47 = np.corrcoef(pred_53, pred_47)[0, 1]
print(f"\n  pred_53 ↔ pred_47 상관: {corr_53_47:.6f}")
print(f"  pred_53 mean: {pred_53.mean():,.0f}, pred_47 mean: {pred_47.mean():,.0f}")

# GTR 변형 간 상관
print(f"\n=== GTR 변형 간 최종 예측 상관 (w53=0.80) ===")
gtr_preds = {}
for gname in GTR_NAMES:
    gtr = compute_gtr(gname)
    gtr_preds[gname] = (pred_53 * 0.8 + pred_47 * 0.2) * gtr

for i, g1 in enumerate(GTR_NAMES):
    for g2 in GTR_NAMES[i+1:]:
        c = np.corrcoef(gtr_preds[g1], gtr_preds[g2])[0, 1]
        diff = np.sqrt(np.mean((gtr_preds[g1] - gtr_preds[g2]) ** 2))
        print(f"  {g1:20s} ↔ {g2:20s}: corr={c:.6f}, RMSD={diff:,.1f}")

# =============================================
# STEP 3: 제출 파일 생성
# =============================================
print(f"\n{'=' * 60}")
print("STEP 3: 제출 파일 생성")
print("=" * 60)

sub_dir = os.path.join(_DIR, 'submissions_gtr')
os.makedirs(sub_dir, exist_ok=True)

for r in results:
    gname, w53 = r['gtr'], r['w53']
    sub = sample_sub.copy()
    sub['Target'] = r['final']
    fname = f"sub_{gname}_w{int(w53*100)}.csv"
    sub.to_csv(os.path.join(sub_dir, fname), index=False)

# 현재 최선(original, 80:20)도 별도 저장
sub = sample_sub.copy()
sub['Target'] = (pred_53 * 0.8 + pred_47 * 0.2) * compute_gtr('original')
sub.to_csv(os.path.join(sub_dir, 'sub_baseline_w80.csv'), index=False)

print(f"\n  {len(results)+1}개 제출 파일 → {sub_dir}/")
print(f"  파일 목록:")
for f in sorted(os.listdir(sub_dir)):
    if f.endswith('.csv'):
        print(f"    {f}")

print(f"\n{'=' * 60}")
print("완료! submissions_gtr/ 폴더에서 제출 파일을 선택하세요.")
print("=" * 60)
