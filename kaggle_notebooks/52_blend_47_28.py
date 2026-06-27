"""
52 BLEND 47:28 = 80:20
파이프라인: 전략47(12모델) × 80% + 전략28(8모델) × 20% 블렌딩
변경점: 12모델과 8모델의 예측을 블렌딩하여 Private 방어
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
MODELS = ['cb_log', 'cb_raw', 'lgb_log', 'lgb_raw']
BLEND_W47 = 0.8

# === 데이터 로드 ===
train_orig = pd.read_csv(f'{INPUT_DIR}/seoul_real_estate_train.csv')
test_orig = pd.read_csv(f'{INPUT_DIR}/seoul_real_estate_test.csv')
sample_sub = pd.read_csv(f'{INPUT_DIR}/sample_submission.csv')
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
LGB_ET_PARAMS = {
    'extra_trees': True, 'max_depth': 3, 'num_leaves': 31,
    'feature_fraction': 0.6, 'bagging_fraction': 0.7, 'bagging_freq': 1,
    'min_child_samples': 30, 'learning_rate': 0.02,
}

def train_4models(train_cb, test_cb, train_lgb, test_lgb, kf,
                  y_log_override=None, y_raw_override=None,
                  return_fold_preds=False, label=""):
    X_cb = train_cb.drop(columns=['Target'])
    X_test_cb = test_cb
    cat_idx = [X_cb.columns.get_loc(c) for c in CAT_FEATURES]
    X_lgb = train_lgb.drop(columns=['Target'])
    X_test_lgb = test_lgb

    if y_log_override is not None:
        y_log = y_log_override
        y_raw = y_raw_override
    else:
        y_log = np.log1p(train_cb['Target'])
        y_raw = train_cb['Target'].values

    oof = {k: np.zeros(len(X_cb)) for k in MODELS}
    tpred = {k: np.zeros(len(X_test_cb)) for k in MODELS}
    fold_test_preds = {k: [] for k in MODELS} if return_fold_preds else None

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_cb)):
        print(f"  [{label}] Fold {fold+1}/{kf.n_splits}")

        m = CatBoostRegressor(loss_function='RMSE', random_seed=42, verbose=0,
                              iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
        m.fit(X_cb.iloc[tr_idx], y_log.iloc[tr_idx] if hasattr(y_log, 'iloc') else y_log[tr_idx],
              eval_set=(X_cb.iloc[va_idx], y_log.iloc[va_idx] if hasattr(y_log, 'iloc') else y_log[va_idx]),
              cat_features=cat_idx)
        oof['cb_log'][va_idx] = np.expm1(m.predict(X_cb.iloc[va_idx]))
        fp = np.expm1(m.predict(X_test_cb))
        tpred['cb_log'] += fp / kf.n_splits
        if return_fold_preds: fold_test_preds['cb_log'].append(fp)

        m = CatBoostRegressor(loss_function='RMSE', random_seed=42, verbose=0,
                              iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
        m.fit(X_cb.iloc[tr_idx], y_raw[tr_idx],
              eval_set=(X_cb.iloc[va_idx], y_raw[va_idx]), cat_features=cat_idx)
        oof['cb_raw'][va_idx] = m.predict(X_cb.iloc[va_idx])
        fp = m.predict(X_test_cb)
        tpred['cb_raw'] += fp / kf.n_splits
        if return_fold_preds: fold_test_preds['cb_raw'].append(fp)

        m = lgb.LGBMRegressor(objective='regression', metric='rmse',
                              verbose=-1, random_state=42, n_estimators=3000, **LGB_PARAMS)
        m.fit(X_lgb.iloc[tr_idx], y_log.iloc[tr_idx] if hasattr(y_log, 'iloc') else y_log[tr_idx],
              eval_set=[(X_lgb.iloc[va_idx], y_log.iloc[va_idx] if hasattr(y_log, 'iloc') else y_log[va_idx])],
              callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])
        oof['lgb_log'][va_idx] = np.expm1(m.predict(X_lgb.iloc[va_idx]))
        fp = np.expm1(m.predict(X_test_lgb))
        tpred['lgb_log'] += fp / kf.n_splits
        if return_fold_preds: fold_test_preds['lgb_log'].append(fp)

        m = lgb.LGBMRegressor(objective='regression', metric='rmse',
                              verbose=-1, random_state=42, n_estimators=3000, **LGB_PARAMS)
        m.fit(X_lgb.iloc[tr_idx], y_raw[tr_idx],
              eval_set=[(X_lgb.iloc[va_idx], y_raw[va_idx])],
              callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])
        oof['lgb_raw'][va_idx] = m.predict(X_lgb.iloc[va_idx])
        fp = m.predict(X_test_lgb)
        tpred['lgb_raw'] += fp / kf.n_splits
        if return_fold_preds: fold_test_preds['lgb_raw'].append(fp)

    return oof, tpred, fold_test_preds

# ========================================
# Stage 1: PL2 신뢰도
# ========================================
print("\n" + "=" * 60)
print("[Stage 1] 원본 데이터 4모델 학습 + Test 신뢰도 측정")
print("=" * 60)

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
train_cb, test_cb, train_lgb, test_lgb = prepare_data(train_orig, test_orig)

oof_s1, tpred_s1, fold_preds_s1 = train_4models(
    train_cb, test_cb, train_lgb, test_lgb, kf,
    return_fold_preds=True, label="S1")

pseudo_labels = np.mean([tpred_s1[k] for k in MODELS], axis=0)
model_means = np.array([tpred_s1[k] for k in MODELS])
model_disagreement = np.std(model_means, axis=0) / np.mean(model_means, axis=0)
fold_cvs = []
for k in MODELS:
    folds_arr = np.array(fold_preds_s1[k])
    fold_cvs.append(np.std(folds_arr, axis=0) / np.mean(folds_arr, axis=0))
fold_cv = np.mean(fold_cvs, axis=0)
confidence = 1 - (model_disagreement + fold_cv) / 2

threshold = np.percentile(confidence, 50)
mask = confidence >= threshold
n_pseudo = mask.sum()

print(f"\n  PL2: 상위 50% = {n_pseudo}건 채택")

test_selected = test_orig[mask].copy()
test_selected['Target'] = pseudo_labels[mask]
train_aug = pd.concat([train_orig, test_selected], ignore_index=True)
area_train_aug = train_aug['Exclusive_Area'].values

# ========================================
# Stage 2: 기존/평당가 8모델 + ET 2 + LGB-ET 2
# ========================================
kf2 = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
train_cb2, test_cb2, train_lgb2, test_lgb2 = prepare_data(train_aug, test_orig)

print(f"\n{'=' * 60}")
print("[Stage 2] 기존 4모델")
print("=" * 60)
oof_base, tpred_base, _ = train_4models(
    train_cb2, test_cb2, train_lgb2, test_lgb2, kf2, label="기존")
for k in MODELS:
    oof_base[k] = oof_base[k][:n_orig]

print(f"\n--- 평당가 4모델 ---")
y_up_log_aug = np.log1p(train_aug['Target'].values / area_train_aug)
y_up_raw_aug = (train_aug['Target'].values / area_train_aug).astype(float)
oof_unit, tpred_unit, _ = train_4models(
    train_cb2, test_cb2, train_lgb2, test_lgb2, kf2,
    y_log_override=y_up_log_aug, y_raw_override=y_up_raw_aug, label="평당가")
for k in MODELS:
    oof_unit[k] = oof_unit[k][:n_orig] * area_train_orig
    tpred_unit[k] = tpred_unit[k] * area_test

# --- ExtraTrees ---
print(f"\n--- ExtraTrees 2모델 ---")
scaler = StandardScaler()
X_et_full = train_cb2.drop(columns=['Target'])
X_et_test_full = test_cb2
et_oof = {'et_log': np.zeros(len(X_et_full)), 'et_raw': np.zeros(len(X_et_full))}
et_tpred = {'et_log': np.zeros(len(X_et_test_full)), 'et_raw': np.zeros(len(X_et_test_full))}

for fold, (tr_idx, va_idx) in enumerate(kf2.split(X_et_full)):
    print(f"  [ET] Fold {fold+1}/{N_SPLITS}")
    X_tr_sc = scaler.fit_transform(X_et_full.iloc[tr_idx])
    X_va_sc = scaler.transform(X_et_full.iloc[va_idx])
    X_te_sc = scaler.transform(X_et_test_full)
    y_log_f = np.log1p(train_aug['Target'].values)
    y_raw_f = train_aug['Target'].values

    m = ExtraTreesRegressor(n_estimators=500, max_depth=12, min_samples_leaf=10, random_state=42, n_jobs=-1)
    m.fit(X_tr_sc, y_log_f[tr_idx])
    et_oof['et_log'][va_idx] = np.expm1(m.predict(X_va_sc))
    et_tpred['et_log'] += np.expm1(m.predict(X_te_sc)) / N_SPLITS

    m = ExtraTreesRegressor(n_estimators=500, max_depth=12, min_samples_leaf=10, random_state=42, n_jobs=-1)
    m.fit(X_tr_sc, y_raw_f[tr_idx])
    et_oof['et_raw'][va_idx] = m.predict(X_va_sc)
    et_tpred['et_raw'] += m.predict(X_te_sc) / N_SPLITS

for k in ['et_log', 'et_raw']:
    et_oof[k] = et_oof[k][:n_orig]

# --- LGBM extra_trees ---
print(f"\n--- LGBM extra_trees 2모델 ---")
X_lgbet = train_lgb2.drop(columns=['Target'])
X_lgbet_test = test_lgb2
lgbet_oof = {'lgbet_log': np.zeros(len(X_lgbet)), 'lgbet_raw': np.zeros(len(X_lgbet))}
lgbet_tpred = {'lgbet_log': np.zeros(len(X_lgbet_test)), 'lgbet_raw': np.zeros(len(X_lgbet_test))}

for fold, (tr_idx, va_idx) in enumerate(kf2.split(X_lgbet)):
    print(f"  [LGB-ET] Fold {fold+1}/{N_SPLITS}")
    y_log_f = np.log1p(train_aug['Target'].values)
    y_raw_f = train_aug['Target'].values

    m = lgb.LGBMRegressor(objective='regression', metric='rmse', verbose=-1,
                           random_state=42, n_estimators=3000, **LGB_ET_PARAMS)
    m.fit(X_lgbet.iloc[tr_idx], y_log_f[tr_idx],
          eval_set=[(X_lgbet.iloc[va_idx], y_log_f[va_idx])],
          callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])
    lgbet_oof['lgbet_log'][va_idx] = np.expm1(m.predict(X_lgbet.iloc[va_idx]))
    lgbet_tpred['lgbet_log'] += np.expm1(m.predict(X_lgbet_test)) / N_SPLITS

    m = lgb.LGBMRegressor(objective='regression', metric='rmse', verbose=-1,
                           random_state=42, n_estimators=3000, **LGB_ET_PARAMS)
    m.fit(X_lgbet.iloc[tr_idx], y_raw_f[tr_idx],
          eval_set=[(X_lgbet.iloc[va_idx], y_raw_f[va_idx])],
          callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])
    lgbet_oof['lgbet_raw'][va_idx] = m.predict(X_lgbet.iloc[va_idx])
    lgbet_tpred['lgbet_raw'] += m.predict(X_lgbet_test) / N_SPLITS

for k in ['lgbet_log', 'lgbet_raw']:
    lgbet_oof[k] = lgbet_oof[k][:n_orig]

# ========================================
# Ridge 스태킹: 12모델(=47) + 8모델(=28) → 블렌딩
# ========================================
print(f"\n{'=' * 60}")
print("Ridge 스태킹 + 블렌딩")
print("=" * 60)

# 12모델 스택 (전략 47)
stack_train_12 = np.column_stack(
    [oof_base[k] for k in MODELS] + [oof_unit[k] for k in MODELS] +
    [et_oof['et_log'], et_oof['et_raw']] +
    [lgbet_oof['lgbet_log'], lgbet_oof['lgbet_raw']])
stack_test_12 = np.column_stack(
    [tpred_base[k] for k in MODELS] + [tpred_unit[k] for k in MODELS] +
    [et_tpred['et_log'], et_tpred['et_raw']] +
    [lgbet_tpred['lgbet_log'], lgbet_tpred['lgbet_raw']])

# 8모델 스택 (전략 28)
stack_train_8 = np.column_stack(
    [oof_base[k] for k in MODELS] + [oof_unit[k] for k in MODELS])
stack_test_8 = np.column_stack(
    [tpred_base[k] for k in MODELS] + [tpred_unit[k] for k in MODELS])

def ridge_stack(st_tr, st_te, alphas):
    best_rmse = float('inf')
    best_test = None
    best_alpha = None
    for alpha in alphas:
        s_oof = np.zeros(n_orig)
        s_test = np.zeros(len(test_orig))
        for fold, (tr_idx, va_idx) in enumerate(kf.split(st_tr)):
            meta = Ridge(alpha=alpha)
            meta.fit(st_tr[tr_idx], y_true_orig[tr_idx])
            s_oof[va_idx] = meta.predict(st_tr[va_idx])
            s_test += meta.predict(st_te) / N_SPLITS
        rmse = np.sqrt(np.mean((s_oof - y_true_orig) ** 2))
        if rmse < best_rmse:
            best_rmse = rmse
            best_test = s_test.copy()
            best_alpha = alpha
    return best_test, best_rmse, best_alpha

alphas = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]

pred_47, rmse_47, alpha_47 = ridge_stack(stack_train_12, stack_test_12, alphas)
print(f"  전략47 (12모델): Ridge(α={alpha_47}) OOF {rmse_47:,.0f}")

pred_28, rmse_28, alpha_28 = ridge_stack(stack_train_8, stack_test_8, alphas)
print(f"  전략28 (8모델):  Ridge(α={alpha_28}) OOF {rmse_28:,.0f}")

# 블렌딩
w47 = BLEND_W47
w28 = 1 - w47
blended = pred_47 * w47 + pred_28 * w28
final_pred = blended * trend_correction

print(f"\n  블렌딩: 47×{w47:.0%} + 28×{w28:.0%}")

sub = sample_sub.copy()
sub['Target'] = final_pred
sub.to_csv(f'{OUTPUT_DIR}/submission.csv', index=False)
print(f"\n제출 파일 저장 완료: {OUTPUT_DIR}/submission.csv")
