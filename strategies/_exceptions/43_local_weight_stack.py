"""
[L4-43] 구별 메타 가중치 + 신뢰도 계층화
──────────────────────────
축약명  : LOCAL WEIGHT
주요 전략: 전략28의 8모델 OOF에 Gu/Area 정보를 추가하여
          "이 구에서는 어떤 모델을 더 믿을지" 학습 + 신뢰도별 다른 전략
출처    : Gemini(Local Weighting) + opencode(신뢰도 계층화)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
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

# === 전처리 & 모델 학습 (전략 28 동일) ===
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

def train_4models(train_cb, test_cb, train_lgb, test_lgb, kf,
                  y_log_override=None, y_raw_override=None,
                  return_fold_preds=False, label=""):
    X_cb = train_cb.drop(columns=['Target'])
    X_test_cb = test_cb
    cat_idx = [X_cb.columns.get_loc(c) for c in CAT_FEATURES]
    X_lgb = train_lgb.drop(columns=['Target'])
    X_test_lgb = test_lgb

    if y_log_override is not None:
        y_log, y_raw = y_log_override, y_raw_override
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
# Stage 1 + PL2 + 8모델 (전략 28 동일)
# ========================================
print("=" * 60)
print("[Stage 1] 원본 4모델 + PL2 신뢰도")
print("=" * 60)

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
train_cb, test_cb, train_lgb, test_lgb = prepare_data(train_orig, test_orig)

oof_s1, tpred_s1, fold_preds_s1 = train_4models(
    train_cb, test_cb, train_lgb, test_lgb, kf,
    return_fold_preds=True, label="S1")

pseudo_labels = np.mean([tpred_s1[k] for k in MODELS], axis=0)
model_means = np.array([tpred_s1[k] for k in MODELS])
model_disagreement = np.std(model_means, axis=0) / np.mean(model_means, axis=0)
fold_cvs = [np.std(np.array(fold_preds_s1[k]), axis=0) / np.mean(np.array(fold_preds_s1[k]), axis=0) for k in MODELS]
test_confidence = 1 - (model_disagreement + np.mean(fold_cvs, axis=0)) / 2

threshold = np.percentile(test_confidence, 50)
mask = test_confidence >= threshold

print(f"\n{'=' * 60}")
print(f"[Stage 2] PL2 + 8모델 학습")
print(f"{'=' * 60}")

test_selected = test_orig[mask].copy()
test_selected['Target'] = pseudo_labels[mask]
train_aug = pd.concat([train_orig, test_selected], ignore_index=True)
area_train_aug = train_aug['Exclusive_Area'].values

kf2 = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
train_cb2, test_cb2, train_lgb2, test_lgb2 = prepare_data(train_aug, test_orig)

print("\n--- 기존 4모델 ---")
oof_base, tpred_base, _ = train_4models(train_cb2, test_cb2, train_lgb2, test_lgb2, kf2, label="기존")
for k in MODELS: oof_base[k] = oof_base[k][:n_orig]

print("\n--- 평당가 4모델 ---")
y_up_log = np.log1p(train_aug['Target'].values / area_train_aug)
y_up_raw = (train_aug['Target'].values / area_train_aug).astype(float)
oof_unit, tpred_unit, _ = train_4models(
    train_cb2, test_cb2, train_lgb2, test_lgb2, kf2,
    y_log_override=y_up_log, y_raw_override=y_up_raw, label="평당가")
for k in MODELS:
    oof_unit[k] = oof_unit[k][:n_orig] * area_train_orig
    tpred_unit[k] = tpred_unit[k] * area_test

# ========================================
# 8모델 OOF + 메타 피처 준비
# ========================================
stack_train = np.column_stack([oof_base[k] for k in MODELS] + [oof_unit[k] for k in MODELS])
stack_test = np.column_stack([tpred_base[k] for k in MODELS] + [tpred_unit[k] for k in MODELS])

# Gu dummy (train_orig 기준)
gu_le = LabelEncoder()
gu_le.fit(train_orig['Gu'])
gu_train = gu_le.transform(train_orig['Gu'])
gu_test = gu_le.transform(test_orig['Gu'])
gu_dummies_train = np.eye(len(gu_le.classes_))[gu_train]
gu_dummies_test = np.eye(len(gu_le.classes_))[gu_test]

# Area (정규화)
area_train_norm = (area_train_orig - area_train_orig.mean()) / area_train_orig.std()
area_test_norm = (area_test - area_train_orig.mean()) / area_train_orig.std()

print(f"\n{'=' * 60}")
print("메타 가중치 실험")
print("=" * 60)

def meta_ridge(stack_tr, stack_te, y, kf_m, alpha=10.0):
    m_oof = np.zeros(len(y))
    m_test = np.zeros(len(stack_te))
    for fold, (tr, va) in enumerate(kf_m.split(stack_tr)):
        m = Ridge(alpha=alpha)
        m.fit(stack_tr[tr], y[tr])
        m_oof[va] = m.predict(stack_tr[va])
        m_test += m.predict(stack_te) / kf_m.n_splits
    rmse = np.sqrt(np.mean((m_oof - y) ** 2))
    return m_oof, m_test, rmse

# --- A. 기준: 8모델만 (전략 28) ---
_, _, rmse_base = meta_ridge(stack_train, stack_test, y_true_orig, kf, alpha=10.0)
print(f"\n  [A] 기준 (8모델 Ridge α=10):        OOF {rmse_base:,.0f}")

# --- B. 8모델 + Gu dummy ---
stack_gu_train = np.column_stack([stack_train, gu_dummies_train])
stack_gu_test = np.column_stack([stack_test, gu_dummies_test])

for alpha in [10, 50, 100, 500]:
    _, _, r = meta_ridge(stack_gu_train, stack_gu_test, y_true_orig, kf, alpha=alpha)
    print(f"  [B] +Gu dummy (α={alpha:4d}):              OOF {r:,.0f} ({r-rmse_base:+,.0f})")

# --- C. 8모델 + Area ---
stack_area_train = np.column_stack([stack_train, area_train_norm])
stack_area_test = np.column_stack([stack_test, area_test_norm])

for alpha in [10, 50, 100]:
    _, _, r = meta_ridge(stack_area_train, stack_area_test, y_true_orig, kf, alpha=alpha)
    print(f"  [C] +Area (α={alpha:4d}):                  OOF {r:,.0f} ({r-rmse_base:+,.0f})")

# --- D. 8모델 + Gu + Area ---
stack_all_train = np.column_stack([stack_train, gu_dummies_train, area_train_norm])
stack_all_test = np.column_stack([stack_test, gu_dummies_test, area_test_norm])

for alpha in [50, 100, 500]:
    _, _, r = meta_ridge(stack_all_train, stack_all_test, y_true_orig, kf, alpha=alpha)
    print(f"  [D] +Gu+Area (α={alpha:4d}):               OOF {r:,.0f} ({r-rmse_base:+,.0f})")

# --- E. Gu × 모델 인터랙션 (Gu별로 모델 가중치 학습) ---
# 각 모델 예측 × 각 Gu dummy = 8모델 × 8구 = 64 피처
interactions_train = []
interactions_test = []
for i in range(8):
    for j in range(gu_dummies_train.shape[1]):
        interactions_train.append(stack_train[:, i] * gu_dummies_train[:, j])
        interactions_test.append(stack_test[:, i] * gu_dummies_test[:, j])

stack_interact_train = np.column_stack([stack_train] + interactions_train)
stack_interact_test = np.column_stack([stack_test] + interactions_test)

for alpha in [100, 500, 1000, 5000]:
    _, _, r = meta_ridge(stack_interact_train, stack_interact_test, y_true_orig, kf, alpha=alpha)
    print(f"  [E] +Gu×모델 (α={alpha:5d}):              OOF {r:,.0f} ({r-rmse_base:+,.0f})")

# ========================================
# 신뢰도 계층화 (opencode 축 2)
# ========================================
print(f"\n{'=' * 60}")
print("신뢰도 계층화 (Test 상/하위 50% 다른 전략)")
print("=" * 60)

# Train의 confidence (OOF 기반 모델 불일치)
oof_means_s2 = np.array([oof_base[k] for k in MODELS])
train_disagreement = np.std(oof_means_s2, axis=0) / np.mean(oof_means_s2, axis=0)
train_conf_median = np.median(train_disagreement)

high_train = train_disagreement <= train_conf_median
low_train = ~high_train

# Test confidence
test_disagreement = np.std([tpred_base[k] for k in MODELS] + [tpred_unit[k] for k in MODELS], axis=0) / \
                    np.mean([tpred_base[k] for k in MODELS] + [tpred_unit[k] for k in MODELS], axis=0)
test_conf_median = np.median(test_disagreement)
high_test = test_disagreement <= test_conf_median
low_test = ~high_test

print(f"  Train: high={high_train.sum()}, low={low_train.sum()}")
print(f"  Test:  high={high_test.sum()}, low={low_test.sum()}")

# F. 전체 Ridge (기준)
ridge_oof, ridge_test, rmse_all = meta_ridge(stack_train, stack_test, y_true_orig, kf, alpha=10.0)
print(f"\n  [F] 전체 Ridge:              OOF {rmse_all:,.0f}")

# G. 상위 50% Ridge + 하위 50% 단순평균
avg_test = np.mean([tpred_base[k] for k in MODELS] + [tpred_unit[k] for k in MODELS], axis=0)
pred_stratified = np.zeros(len(test_orig))
pred_stratified[high_test] = ridge_test[high_test]
pred_stratified[low_test] = avg_test[low_test]

# OOF에서도 같은 방식으로 계산
avg_oof = np.mean([oof_base[k] for k in MODELS] + [oof_unit[k] for k in MODELS], axis=0)
oof_stratified = np.zeros(n_orig)
oof_stratified[high_train] = ridge_oof[high_train]
oof_stratified[low_train] = avg_oof[low_train]
rmse_strat = np.sqrt(np.mean((oof_stratified - y_true_orig) ** 2))
print(f"  [G] high=Ridge, low=단순평균: OOF {rmse_strat:,.0f} ({rmse_strat-rmse_all:+,.0f})")

# H. 상위 Ridge(α=10) + 하위 Ridge(α=100, 더 정규화)
for lo_alpha in [50, 100, 500]:
    _, lo_test, _ = meta_ridge(stack_train, stack_test, y_true_orig, kf, alpha=lo_alpha)
    pred_h = np.zeros(len(test_orig))
    pred_h[high_test] = ridge_test[high_test]
    pred_h[low_test] = lo_test[low_test]

    oof_h = np.zeros(n_orig)
    _, lo_oof_full, _ = meta_ridge(stack_train, stack_test, y_true_orig, kf, alpha=lo_alpha)
    # 재계산 필요 - 위에서 oof도 받아야 함
    lo_oof, _, _ = meta_ridge(stack_train, stack_test, y_true_orig, kf, alpha=lo_alpha)
    oof_h[high_train] = ridge_oof[high_train]
    oof_h[low_train] = lo_oof[low_train]
    r = np.sqrt(np.mean((oof_h - y_true_orig) ** 2))
    print(f"  [H] high=Ridge(10), low=Ridge({lo_alpha}): OOF {r:,.0f} ({r-rmse_all:+,.0f})")

# ========================================
# 최종 비교
# ========================================
print(f"\n{'=' * 60}")
print("최종 비교")
print("=" * 60)
print(f"  전략 28 기준 (8모델 Ridge α=10): OOF {rmse_base:,.0f}")
print(f"  어떤 변형이든 이보다 나아야 의미 있음")

record_result('L4', 43, 'LOCAL WEIGHT',
              '구별 메타 가중치 + 신뢰도 계층화 실험', rmse_base, 'tested')
