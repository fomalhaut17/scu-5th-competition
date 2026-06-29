"""
57 NUCLEAR WIDE STACK - 로컬 실험
Phase A: 20모델 no-PL2 (12기존 vs +4 MAE vs +4 ET-UP)
Phase B: 20모델 + Weighted PL2
"""
import os
import sys
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

_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
INPUT_DIR = _DIR
train_orig = pd.read_csv(f'{INPUT_DIR}/seoul_real_estate_train.csv')
test_orig = pd.read_csv(f'{INPUT_DIR}/seoul_real_estate_test.csv')
y_true = train_orig['Target'].values
n_orig = len(train_orig)
area_train = train_orig['Exclusive_Area'].values
area_test = test_orig['Exclusive_Area'].values

N_SPLITS = 5
CAT_FEATURES = ['Gu', 'Dong']
SEED = 42

# === 전처리 ===
def base_preprocess(df):
    df = df.copy()
    df['Distance_to_Subway'] = df['Distance_to_Subway'].fillna(df['Distance_to_Subway'].median())
    df['Age'] = 2026 - df['Year_Built']
    df['Year'] = df['Transaction_YearMonth'] // 100
    df['Month'] = df['Transaction_YearMonth'] % 100
    df = df.drop(columns=['ID', 'Transaction_YearMonth', 'Year_Built'])
    return df

def add_fe(df):
    df = df.copy()
    df['YearMonth_Seq'] = (df['Year'] - 2024) * 12 + df['Month']
    df['Area_x_Floor'] = df['Exclusive_Area'] * df['Floor']
    df['Floor_per_Area'] = df['Floor'] / df['Exclusive_Area']
    df['Brand_x_Area'] = df['Brand_Apartment'] * df['Exclusive_Area']
    return df

def encode_cat(train_df, test_df, as_category=False):
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
    train_p = add_fe(base_preprocess(train_df))
    test_p = add_fe(base_preprocess(test_df))
    tr_cb, te_cb = encode_cat(train_p, test_p, as_category=False)
    tr_lgb, te_lgb = encode_cat(train_p, test_p, as_category=True)
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

tr_cb, te_cb, tr_lgb, te_lgb = prepare_data(train_orig, test_orig)
X_cb = tr_cb.drop(columns=['Target'])
X_te_cb = te_cb
cat_idx = [X_cb.columns.get_loc(c) for c in CAT_FEATURES]
X_lgb = tr_lgb.drop(columns=['Target'])
X_te_lgb = te_lgb

y_log = np.log1p(y_true)
y_raw = y_true.copy()
y_up_log = np.log1p(y_true / area_train)
y_up_raw = (y_true / area_train).astype(float)


def train_20models(X_cb, X_te_cb, X_lgb, X_te_lgb, cat_idx,
                   y_log, y_raw, y_up_log, y_up_raw, area_tr, area_te,
                   n_orig, kf, seed, sample_weights=None):
    """20모델 학습. sample_weights가 주어지면 Weighted PL2용."""
    oof = {}
    tpred = {}
    n_train = len(X_cb)
    n_test = len(X_te_cb)

    UP_MODELS = ['cb_up_log', 'cb_up_raw', 'lgb_up_log', 'lgb_up_raw',
                 'et_up_log', 'et_up_raw', 'lgbet_up_log', 'lgbet_up_raw']

    for fold, (tr, va) in enumerate(kf.split(X_cb)):
        print(f"  Fold {fold+1}/{N_SPLITS} (seed={seed})")
        w_tr = sample_weights[tr] if sample_weights is not None else None

        # --- CB RMSE log/raw ---
        for name, y, transform in [('cb_log', y_log, 'log'), ('cb_raw', y_raw, 'raw')]:
            m = CatBoostRegressor(loss_function='RMSE', random_seed=seed, verbose=0,
                                  iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
            m.fit(X_cb.iloc[tr], y[tr], eval_set=(X_cb.iloc[va], y[va]),
                  cat_features=cat_idx, sample_weight=w_tr)
            pred_va = np.expm1(m.predict(X_cb.iloc[va])) if transform == 'log' else m.predict(X_cb.iloc[va])
            pred_te = np.expm1(m.predict(X_te_cb)) if transform == 'log' else m.predict(X_te_cb)
            oof.setdefault(name, np.zeros(n_train))[va] = pred_va
            tpred[name] = tpred.get(name, np.zeros(n_test)) + pred_te / N_SPLITS

        # --- CB RMSE unit_price log/raw ---
        for name, y, transform in [('cb_up_log', y_up_log, 'log'), ('cb_up_raw', y_up_raw, 'raw')]:
            m = CatBoostRegressor(loss_function='RMSE', random_seed=seed, verbose=0,
                                  iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
            m.fit(X_cb.iloc[tr], y[tr], eval_set=(X_cb.iloc[va], y[va]),
                  cat_features=cat_idx, sample_weight=w_tr)
            pred_va = np.expm1(m.predict(X_cb.iloc[va])) if transform == 'log' else m.predict(X_cb.iloc[va])
            pred_te = np.expm1(m.predict(X_te_cb)) if transform == 'log' else m.predict(X_te_cb)
            oof.setdefault(name, np.zeros(n_train))[va] = pred_va
            tpred[name] = tpred.get(name, np.zeros(n_test)) + pred_te / N_SPLITS

        # --- CB MAE log/raw (NEW) ---
        for name, y, transform in [('cb_mae_log', y_log, 'log'), ('cb_mae_raw', y_raw, 'raw')]:
            m = CatBoostRegressor(loss_function='MAE', random_seed=seed, verbose=0,
                                  iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
            m.fit(X_cb.iloc[tr], y[tr], eval_set=(X_cb.iloc[va], y[va]),
                  cat_features=cat_idx, sample_weight=w_tr)
            pred_va = np.expm1(m.predict(X_cb.iloc[va])) if transform == 'log' else m.predict(X_cb.iloc[va])
            pred_te = np.expm1(m.predict(X_te_cb)) if transform == 'log' else m.predict(X_te_cb)
            oof.setdefault(name, np.zeros(n_train))[va] = pred_va
            tpred[name] = tpred.get(name, np.zeros(n_test)) + pred_te / N_SPLITS

        # --- LGB RMSE log/raw ---
        for name, y, transform in [('lgb_log', y_log, 'log'), ('lgb_raw', y_raw, 'raw')]:
            m = lgb.LGBMRegressor(objective='regression', metric='rmse', verbose=-1,
                                   random_state=seed, n_estimators=3000, **LGB_PARAMS)
            m.fit(X_lgb.iloc[tr], y[tr], eval_set=[(X_lgb.iloc[va], y[va])],
                  callbacks=[lgb.early_stopping(100, verbose=False)],
                  sample_weight=w_tr)
            pred_va = np.expm1(m.predict(X_lgb.iloc[va])) if transform == 'log' else m.predict(X_lgb.iloc[va])
            pred_te = np.expm1(m.predict(X_te_lgb)) if transform == 'log' else m.predict(X_te_lgb)
            oof.setdefault(name, np.zeros(n_train))[va] = pred_va
            tpred[name] = tpred.get(name, np.zeros(n_test)) + pred_te / N_SPLITS

        # --- LGB RMSE unit_price log/raw ---
        for name, y, transform in [('lgb_up_log', y_up_log, 'log'), ('lgb_up_raw', y_up_raw, 'raw')]:
            m = lgb.LGBMRegressor(objective='regression', metric='rmse', verbose=-1,
                                   random_state=seed, n_estimators=3000, **LGB_PARAMS)
            m.fit(X_lgb.iloc[tr], y[tr], eval_set=[(X_lgb.iloc[va], y[va])],
                  callbacks=[lgb.early_stopping(100, verbose=False)],
                  sample_weight=w_tr)
            pred_va = np.expm1(m.predict(X_lgb.iloc[va])) if transform == 'log' else m.predict(X_lgb.iloc[va])
            pred_te = np.expm1(m.predict(X_te_lgb)) if transform == 'log' else m.predict(X_te_lgb)
            oof.setdefault(name, np.zeros(n_train))[va] = pred_va
            tpred[name] = tpred.get(name, np.zeros(n_test)) + pred_te / N_SPLITS

        # --- LGB MAE log/raw (NEW) ---
        for name, y, transform in [('lgb_mae_log', y_log, 'log'), ('lgb_mae_raw', y_raw, 'raw')]:
            m = lgb.LGBMRegressor(objective='mae', metric='mae', verbose=-1,
                                   random_state=seed, n_estimators=3000, **LGB_PARAMS)
            m.fit(X_lgb.iloc[tr], y[tr], eval_set=[(X_lgb.iloc[va], y[va])],
                  callbacks=[lgb.early_stopping(100, verbose=False)],
                  sample_weight=w_tr)
            pred_va = np.expm1(m.predict(X_lgb.iloc[va])) if transform == 'log' else m.predict(X_lgb.iloc[va])
            pred_te = np.expm1(m.predict(X_te_lgb)) if transform == 'log' else m.predict(X_te_lgb)
            oof.setdefault(name, np.zeros(n_train))[va] = pred_va
            tpred[name] = tpred.get(name, np.zeros(n_test)) + pred_te / N_SPLITS

        # --- ET / LGBET (Scaled features) ---
        scaler = StandardScaler()
        X_sc_tr = scaler.fit_transform(X_cb.iloc[tr])
        X_sc_va = scaler.transform(X_cb.iloc[va])
        X_sc_te = scaler.transform(X_te_cb)

        # ET log/raw (기존)
        for name, y, transform in [('et_log', y_log, 'log'), ('et_raw', y_raw, 'raw')]:
            m = ExtraTreesRegressor(n_estimators=500, max_depth=12, min_samples_leaf=10,
                                     random_state=seed, n_jobs=-1)
            m.fit(X_sc_tr, y[tr], sample_weight=w_tr)
            pred_va = np.expm1(m.predict(X_sc_va)) if transform == 'log' else m.predict(X_sc_va)
            pred_te = np.expm1(m.predict(X_sc_te)) if transform == 'log' else m.predict(X_sc_te)
            oof.setdefault(name, np.zeros(n_train))[va] = pred_va
            tpred[name] = tpred.get(name, np.zeros(n_test)) + pred_te / N_SPLITS

        # ET unit_price log/raw (NEW)
        for name, y, transform in [('et_up_log', y_up_log, 'log'), ('et_up_raw', y_up_raw, 'raw')]:
            m = ExtraTreesRegressor(n_estimators=500, max_depth=12, min_samples_leaf=10,
                                     random_state=seed, n_jobs=-1)
            m.fit(X_sc_tr, y[tr], sample_weight=w_tr)
            pred_va = np.expm1(m.predict(X_sc_va)) if transform == 'log' else m.predict(X_sc_va)
            pred_te = np.expm1(m.predict(X_sc_te)) if transform == 'log' else m.predict(X_sc_te)
            oof.setdefault(name, np.zeros(n_train))[va] = pred_va
            tpred[name] = tpred.get(name, np.zeros(n_test)) + pred_te / N_SPLITS

        # LGBET log/raw (기존)
        for name, y, transform in [('lgbet_log', y_log, 'log'), ('lgbet_raw', y_raw, 'raw')]:
            m = lgb.LGBMRegressor(objective='regression', metric='rmse', verbose=-1,
                                   random_state=seed, n_estimators=3000, **LGB_ET_PARAMS)
            m.fit(X_lgb.iloc[tr], y[tr], eval_set=[(X_lgb.iloc[va], y[va])],
                  callbacks=[lgb.early_stopping(100, verbose=False)],
                  sample_weight=w_tr)
            pred_va = np.expm1(m.predict(X_lgb.iloc[va])) if transform == 'log' else m.predict(X_lgb.iloc[va])
            pred_te = np.expm1(m.predict(X_te_lgb)) if transform == 'log' else m.predict(X_te_lgb)
            oof.setdefault(name, np.zeros(n_train))[va] = pred_va
            tpred[name] = tpred.get(name, np.zeros(n_test)) + pred_te / N_SPLITS

        # LGBET unit_price log/raw (NEW)
        for name, y, transform in [('lgbet_up_log', y_up_log, 'log'), ('lgbet_up_raw', y_up_raw, 'raw')]:
            m = lgb.LGBMRegressor(objective='regression', metric='rmse', verbose=-1,
                                   random_state=seed, n_estimators=3000, **LGB_ET_PARAMS)
            m.fit(X_lgb.iloc[tr], y[tr], eval_set=[(X_lgb.iloc[va], y[va])],
                  callbacks=[lgb.early_stopping(100, verbose=False)],
                  sample_weight=w_tr)
            pred_va = np.expm1(m.predict(X_lgb.iloc[va])) if transform == 'log' else m.predict(X_lgb.iloc[va])
            pred_te = np.expm1(m.predict(X_te_lgb)) if transform == 'log' else m.predict(X_te_lgb)
            oof.setdefault(name, np.zeros(n_train))[va] = pred_va
            tpred[name] = tpred.get(name, np.zeros(n_test)) + pred_te / N_SPLITS

    # unit_price → price 환산
    for k in UP_MODELS:
        oof[k] = oof[k][:n_orig] * area_tr[:n_orig]
        tpred[k] = tpred[k] * area_te
    for k in [kk for kk in oof if kk not in UP_MODELS]:
        oof[k] = oof[k][:n_orig]

    return oof, tpred


def ridge_stack(oof_dict, tpred_dict, y_true, model_names, label=""):
    """주어진 모델 이름들로 Ridge 스태킹, 최적 alpha 탐색."""
    n = len(y_true)
    n_test = len(list(tpred_dict.values())[0])
    st_tr = np.column_stack([oof_dict[k] for k in model_names])
    st_te = np.column_stack([tpred_dict[k] for k in model_names])

    best_rmse = float('inf')
    best_alpha = None
    best_test = None
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    for alpha in [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]:
        s_oof = np.zeros(n)
        s_test = np.zeros(n_test)
        for tr, va in kf.split(st_tr):
            meta = Ridge(alpha=alpha)
            meta.fit(st_tr[tr], y_true[tr])
            s_oof[va] = meta.predict(st_tr[va])
            s_test += meta.predict(st_te) / N_SPLITS
        rmse = np.sqrt(np.mean((s_oof - y_true) ** 2))
        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = alpha
            best_test = s_test.copy()

    print(f"  [{label}] {len(model_names)}모델 Ridge(α={best_alpha}) → OOF RMSE: {best_rmse:,.1f}")
    return best_test, best_rmse, best_alpha


# === 개별 모델 OOF 보기 ===
def show_individual_oof(oof_dict, y_true):
    print("\n=== 개별 모델 OOF RMSE ===")
    results = []
    for name, pred in sorted(oof_dict.items()):
        rmse = np.sqrt(np.mean((pred - y_true) ** 2))
        results.append((name, rmse))
    for name, rmse in sorted(results, key=lambda x: x[1]):
        tag = " ★NEW" if any(t in name for t in ['mae', 'et_up', 'lgbet_up']) else ""
        print(f"  {name:15s}: {rmse:,.0f}{tag}")
    return results


# =============================================
# PHASE A: 20모델 no-PL2 (single seed)
# =============================================
print("=" * 60)
print("PHASE A: 20모델 no-PL2 (seed=42)")
print("=" * 60)

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
oof_all, tpred_all = train_20models(
    X_cb, X_te_cb, X_lgb, X_te_lgb, cat_idx,
    y_log, y_raw, y_up_log, y_up_raw,
    area_train, area_test, n_orig, kf, SEED
)

show_individual_oof(oof_all, y_true)

# 모델 그룹 정의
BASE_12 = ['cb_log', 'cb_raw', 'lgb_log', 'lgb_raw',
           'cb_up_log', 'cb_up_raw', 'lgb_up_log', 'lgb_up_raw',
           'et_log', 'et_raw', 'lgbet_log', 'lgbet_raw']

MAE_4 = ['cb_mae_log', 'cb_mae_raw', 'lgb_mae_log', 'lgb_mae_raw']

ET_UP_4 = ['et_up_log', 'et_up_raw', 'lgbet_up_log', 'lgbet_up_raw']

print("\n=== Ridge 스태킹 비교 ===")
_, rmse_12, _ = ridge_stack(oof_all, tpred_all, y_true, BASE_12, "기존 12모델")
_, rmse_12_mae, _ = ridge_stack(oof_all, tpred_all, y_true, BASE_12 + MAE_4, "+MAE = 16모델")
_, rmse_12_etup, _ = ridge_stack(oof_all, tpred_all, y_true, BASE_12 + ET_UP_4, "+ET-UP = 16모델")
_, rmse_20, _ = ridge_stack(oof_all, tpred_all, y_true, BASE_12 + MAE_4 + ET_UP_4, "전체 20모델")

# MAE만 / ET-UP만 추가 vs 둘 다
print(f"\n  12 → +MAE: {rmse_12 - rmse_12_mae:+.1f}")
print(f"  12 → +ET-UP: {rmse_12 - rmse_12_etup:+.1f}")
print(f"  12 → 20모델: {rmse_12 - rmse_20:+.1f}")

# 상관관계 분석 (다양성 확인)
print("\n=== 신규 모델 상관관계 (기존 cb_log 대비) ===")
ref = oof_all['cb_log']
for name in MAE_4 + ET_UP_4:
    corr = np.corrcoef(ref, oof_all[name])[0, 1]
    print(f"  cb_log ↔ {name:15s}: {corr:.4f}")


# =============================================
# PHASE B: Weighted PL2 + 20모델
# =============================================
print(f"\n{'=' * 60}")
print("PHASE B: Weighted PL2 + best 모델 조합")
print("=" * 60)

# Stage 1 predictions (4 base models)
STAGE1_MODELS = ['cb_log', 'cb_raw', 'lgb_log', 'lgb_raw']
base4_tpred = np.column_stack([tpred_all[k] for k in STAGE1_MODELS])
pseudo_labels = base4_tpred.mean(axis=1)
model_std = np.std(base4_tpred, axis=1)
model_mean = np.mean(base4_tpred, axis=1)
confidence = 1 - (model_std / model_mean)

print(f"\n  Confidence 분포: min={confidence.min():.4f}, median={np.median(confidence):.4f}, max={confidence.max():.4f}")

# Weighted PL2: 모든 테스트 샘플 사용, confidence를 weight로
test_with_pl = test_orig.copy()
test_with_pl['Target'] = pseudo_labels
train_aug = pd.concat([train_orig, test_with_pl], ignore_index=True)
area_aug = train_aug['Exclusive_Area'].values
n_aug = len(train_aug)

# 가중치: 원본=1.0, PL=confidence
weights_full = np.concatenate([np.ones(n_orig), confidence])

# 비교용: Hard cutoff 50% (기존 방식)
mask50 = confidence >= np.percentile(confidence, 50)
test_50 = test_orig[mask50].copy()
test_50['Target'] = pseudo_labels[mask50]
train_50 = pd.concat([train_orig, test_50], ignore_index=True)
area_50 = train_50['Exclusive_Area'].values
n_50 = len(train_50)

tr_cb_aug, te_cb_aug, tr_lgb_aug, te_lgb_aug = prepare_data(train_aug, test_orig)
X_cb_aug = tr_cb_aug.drop(columns=['Target'])
X_te_cb_aug = te_cb_aug
cat_idx_aug = [X_cb_aug.columns.get_loc(c) for c in CAT_FEATURES]
X_lgb_aug = tr_lgb_aug.drop(columns=['Target'])
X_te_lgb_aug = te_lgb_aug

y_log_aug = np.log1p(train_aug['Target'].values)
y_raw_aug = train_aug['Target'].values.astype(float)
y_up_log_aug = np.log1p(train_aug['Target'].values / area_aug)
y_up_raw_aug = (train_aug['Target'].values / area_aug).astype(float)

# --- Weighted PL2 ---
print("\n--- Weighted PL2 (전체 531건, confidence weight) ---")
kf_aug = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
oof_wpl, tpred_wpl = train_20models(
    X_cb_aug, X_te_cb_aug, X_lgb_aug, X_te_lgb_aug, cat_idx_aug,
    y_log_aug, y_raw_aug, y_up_log_aug, y_up_raw_aug,
    area_aug, area_test, n_orig, kf_aug, SEED, sample_weights=weights_full
)

# --- Hard cutoff PL2 (기존 방식, 50%) ---
print("\n--- Hard PL2 (상위 50%, 기존 방식) ---")
tr_cb_50, te_cb_50, tr_lgb_50, te_lgb_50 = prepare_data(train_50, test_orig)
X_cb_50 = tr_cb_50.drop(columns=['Target'])
X_te_cb_50 = te_cb_50
cat_idx_50 = [X_cb_50.columns.get_loc(c) for c in CAT_FEATURES]
X_lgb_50 = tr_lgb_50.drop(columns=['Target'])
X_te_lgb_50 = te_lgb_50
y_log_50 = np.log1p(train_50['Target'].values)
y_raw_50 = train_50['Target'].values.astype(float)
y_up_log_50 = np.log1p(train_50['Target'].values / area_50)
y_up_raw_50 = (train_50['Target'].values / area_50).astype(float)

kf_50 = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
oof_hpl, tpred_hpl = train_20models(
    X_cb_50, X_te_cb_50, X_lgb_50, X_te_lgb_50, cat_idx_50,
    y_log_50, y_raw_50, y_up_log_50, y_up_raw_50,
    area_50, area_test, n_orig, kf_50, SEED
)

# === 최종 비교 ===
print(f"\n{'=' * 60}")
print("=== 최종 비교: 모델 수 × PL 방식 ===")
print("=" * 60)

# 최적 모델 조합 결정 (Phase A에서)
if rmse_20 <= min(rmse_12, rmse_12_mae, rmse_12_etup):
    best_models = BASE_12 + MAE_4 + ET_UP_4
    best_label = "20모델"
elif rmse_12_mae <= min(rmse_12, rmse_12_etup):
    best_models = BASE_12 + MAE_4
    best_label = "16모델(+MAE)"
elif rmse_12_etup <= rmse_12:
    best_models = BASE_12 + ET_UP_4
    best_label = "16모델(+ET-UP)"
else:
    best_models = BASE_12
    best_label = "12모델"

for models, label in [(BASE_12, "12모델"), (best_models, best_label)]:
    print(f"\n--- {label} ---")
    _, r1, _ = ridge_stack(oof_all, tpred_all, y_true, models, f"{label} no-PL2")
    _, r2, _ = ridge_stack(oof_hpl, tpred_hpl, y_true, models, f"{label} Hard-PL2")
    _, r3, _ = ridge_stack(oof_wpl, tpred_wpl, y_true, models, f"{label} Weight-PL2")

print(f"\n참고: 전략 47(12모델 Hard-PL2) OOF = 2,191")
print(f"참고: 전략 53(12모델 no-PL2) OOF = 2,229")
