"""
38 PL2 UNIT PRICE ALPHA 0.7
파이프라인: 전략28 동일, 트렌드 보정 강도만 α=0.7로 약화
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
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

print("=== 구별 월성장률 ===")
for gu, g in sorted(gu_growth.items(), key=lambda x: -x[1]):
    print(f"  {gu:15s}: {g*100:+.2f}%")

test_seq = (test_orig['Transaction_YearMonth'] // 100 - 2024) * 12 + test_orig['Transaction_YearMonth'] % 100
months_ahead = test_seq.values - last_train_seq
TREND_ALPHA = 0.7
trend_correction = np.array([(1 + TREND_ALPHA * gu_growth[gu]) ** m for gu, m in zip(test_orig['Gu'], months_ahead)])
print(f"\n보정 범위: +{(trend_correction.min()-1)*100:.1f}% ~ +{(trend_correction.max()-1)*100:.1f}%")

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
# Stage 1: 원본 데이터 4모델 → PL 신뢰도 측정
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

print(f"\n  신뢰도: mean={confidence.mean():.4f}, min={confidence.min():.4f}")

avg_oof_s1 = np.mean([oof_s1[k] for k in MODELS], axis=0)
rmse_s1 = np.sqrt(np.mean((avg_oof_s1 - y_true_orig) ** 2))
print(f"  Stage 1 4모델 평균 OOF RMSE: {rmse_s1:,.0f}")

# ========================================
# Stage 2: PL(상위 50%) 추가 → 8모델 학습
# ========================================
threshold = np.percentile(confidence, 50)
mask = confidence >= threshold
n_pseudo = mask.sum()

print(f"\n{'=' * 60}")
print(f"[Stage 2] PL2 + 기존/평당가 8모델 (상위 50% = {n_pseudo}건)")
print(f"{'=' * 60}")

test_selected = test_orig[mask].copy()
test_selected['Target'] = pseudo_labels[mask]
train_aug = pd.concat([train_orig, test_selected], ignore_index=True)

area_train_aug = train_aug['Exclusive_Area'].values

kf2 = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
train_cb2, test_cb2, train_lgb2, test_lgb2 = prepare_data(train_aug, test_orig)

# --- 기존 4모델 ---
print("\n--- 기존 4모델 ---")
oof_base, tpred_base, _ = train_4models(
    train_cb2, test_cb2, train_lgb2, test_lgb2, kf2, label="기존")

for k in MODELS:
    oof_base[k] = oof_base[k][:n_orig]

# --- 평당가 4모델 ---
print("\n--- 평당가 4모델 ---")
y_up_log_aug = np.log1p(train_aug['Target'].values / area_train_aug)
y_up_raw_aug = (train_aug['Target'].values / area_train_aug).astype(float)

oof_unit, tpred_unit, _ = train_4models(
    train_cb2, test_cb2, train_lgb2, test_lgb2, kf2,
    y_log_override=y_up_log_aug, y_raw_override=y_up_raw_aug, label="평당가")

for k in MODELS:
    oof_unit[k] = oof_unit[k][:n_orig] * area_train_orig
    tpred_unit[k] = tpred_unit[k] * area_test

# ========================================
# 블렌딩 비교
# ========================================
print(f"\n{'=' * 60}")
print("블렌딩 비교")
print("=" * 60)

base_w_oof = 0.20*oof_base['cb_log'] + 0.30*oof_base['cb_raw'] + 0.20*oof_base['lgb_log'] + 0.30*oof_base['lgb_raw']
rmse_base_w = np.sqrt(np.mean((base_w_oof - y_true_orig) ** 2))

all_oof = np.mean([oof_base[k] for k in MODELS] + [oof_unit[k] for k in MODELS], axis=0)
all_test = np.mean([tpred_base[k] for k in MODELS] + [tpred_unit[k] for k in MODELS], axis=0)
rmse_8avg = np.sqrt(np.mean((all_oof - y_true_orig) ** 2))

# Ridge 스태킹 (8모델)
stack_train = np.column_stack([oof_base[k] for k in MODELS] + [oof_unit[k] for k in MODELS])
stack_test_arr = np.column_stack([tpred_base[k] for k in MODELS] + [tpred_unit[k] for k in MODELS])

best_ridge_rmse = float('inf')
best_ridge_alpha = None
best_ridge_test = None

for alpha in [0.1, 0.5, 1.0, 5.0, 10.0]:
    s_oof = np.zeros(n_orig)
    s_test = np.zeros(len(test_orig))
    for fold, (tr_idx, va_idx) in enumerate(kf.split(stack_train)):
        meta = Ridge(alpha=alpha)
        meta.fit(stack_train[tr_idx], y_true_orig[tr_idx])
        s_oof[va_idx] = meta.predict(stack_train[va_idx])
        s_test += meta.predict(stack_test_arr) / N_SPLITS
    ridge_rmse = np.sqrt(np.mean((s_oof - y_true_orig) ** 2))
    print(f"  Ridge(α={alpha:5.1f}) OOF RMSE: {ridge_rmse:,.0f}")
    if ridge_rmse < best_ridge_rmse:
        best_ridge_rmse = ridge_rmse
        best_ridge_alpha = alpha
        best_ridge_test = s_test.copy()

print(f"\n  PL2 기존 4모델 가중(60%)  : {rmse_base_w:,.0f}")
print(f"  PL2 8모델 단순평균        : {rmse_8avg:,.0f}")
print(f"  PL2 8모델 Ridge(α={best_ridge_alpha})   : {best_ridge_rmse:,.0f}")
print(f"  ─────────────────────────────")
print(f"  전략 26 (PL2+Scale)       : 2,215 / Public 2,149.6")
print(f"  전략 27 (평당가 크로스)    : 2,217 / Public 2,152.8")

# 최선 선택
options = {
    'base_w': (0.20*tpred_base['cb_log'] + 0.30*tpred_base['cb_raw'] + 0.20*tpred_base['lgb_log'] + 0.30*tpred_base['lgb_raw'],
               rmse_base_w, 'PL2 기존 가중(60%)'),
    '8avg': (all_test, rmse_8avg, 'PL2 8모델 단순평균'),
    '8ridge': (best_ridge_test, best_ridge_rmse, f'PL2 8모델 Ridge(α={best_ridge_alpha})'),
}

best_key = min(options, key=lambda k: options[k][1])
best_test_pred, best_rmse, best_desc = options[best_key]

final_pred = best_test_pred * trend_correction

print(f"\n  ★ 최선: [{best_key}] {best_desc} → OOF {best_rmse:,.0f}")

print(f"\n{'=' * 60}")
print(f"최종: {best_desc} OOF RMSE {best_rmse:,.0f}")
print(f"{'=' * 60}")

print(f"\n트렌드 보정 전 평균: {best_test_pred.mean():,.0f}")
print(f"트렌드 보정 후 평균: {final_pred.mean():,.0f}")

sub = sample_sub.copy()
sub['Target'] = final_pred
sub.to_csv(f'{OUTPUT_DIR}/submission.csv', index=False)
print(f"\n제출 파일 저장 완료: {OUTPUT_DIR}/submission.csv")
