"""
[L4-42] Confidence를 Feature로 활용
──────────────────────────
축약명  : CONF FEAT
주요 전략: Stage 1의 모델 불일치/fold 변동을 피처로 추가하여 Stage 2 학습
차별점  : "이 샘플이 얼마나 예측하기 어려운가"를 모델이 알게 됨
출처    : opencode 조언 (2026-06-26)
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
# Stage 1: 원본 4모델 → confidence 피처 생성
# ========================================
print("=" * 60)
print("[Stage 1] 원본 데이터 4모델 → confidence 피처 생성")
print("=" * 60)

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
train_cb, test_cb, train_lgb, test_lgb = prepare_data(train_orig, test_orig)

oof_s1, tpred_s1, fold_preds_s1 = train_4models(
    train_cb, test_cb, train_lgb, test_lgb, kf,
    return_fold_preds=True, label="S1")

# --- Train용 confidence (K-Fold OOF 기반) ---
# 모델 간 불일치
oof_means = np.array([oof_s1[k] for k in MODELS])
train_model_disagreement = np.std(oof_means, axis=0) / np.mean(oof_means, axis=0)

# Fold 간 변동은 train에서는 직접 계산 불가 (각 샘플은 1개 fold에만 예측됨)
# 대신 모델 간 불일치만 사용
train_confidence = 1 - train_model_disagreement

# --- Test용 confidence ---
test_model_means = np.array([tpred_s1[k] for k in MODELS])
test_model_disagreement = np.std(test_model_means, axis=0) / np.mean(test_model_means, axis=0)

fold_cvs = []
for k in MODELS:
    folds_arr = np.array(fold_preds_s1[k])
    fold_cvs.append(np.std(folds_arr, axis=0) / np.mean(folds_arr, axis=0))
test_fold_cv = np.mean(fold_cvs, axis=0)

test_confidence = 1 - (test_model_disagreement + test_fold_cv) / 2

print(f"\n  Train confidence: mean={train_confidence.mean():.4f}, std={train_confidence.std():.4f}")
print(f"  Test confidence:  mean={test_confidence.mean():.4f}, std={test_confidence.std():.4f}")

# OOF 예측 평균 (예측값 자체도 피처로 활용)
train_pred_avg = np.mean(oof_means, axis=0)
test_pred_avg = np.mean(test_model_means, axis=0)

# PL pseudo labels
pseudo_labels = test_pred_avg.copy()

# ========================================
# Stage 2: PL + confidence 피처 추가 8모델 학습
# ========================================
threshold = np.percentile(test_confidence, 50)
mask = test_confidence >= threshold
n_pseudo = mask.sum()

print(f"\n{'=' * 60}")
print(f"[Stage 2] PL2 + confidence 피처 추가 8모델")
print(f"{'=' * 60}")

test_selected = test_orig[mask].copy()
test_selected['Target'] = pseudo_labels[mask]
train_aug = pd.concat([train_orig, test_selected], ignore_index=True)

# confidence 피처를 원본 데이터에 추가
# Train: OOF 기반 confidence
# PL 샘플: test confidence 중 선택된 것
# Test: test confidence
train_conf_aug = np.concatenate([train_confidence, test_confidence[mask]])
test_conf_full = test_confidence

# model disagreement도 별도 피처로
train_disagree_aug = np.concatenate([train_model_disagreement, test_model_disagreement[mask]])
test_disagree_full = test_model_disagreement

area_train_aug = train_aug['Exclusive_Area'].values

kf2 = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

def prepare_data_with_conf(train_df, test_df, train_conf, test_conf, train_disagree, test_disagree):
    train_p = add_feature_engineering(base_preprocess(train_df))
    test_p = add_feature_engineering(base_preprocess(test_df))
    train_p['confidence'] = train_conf
    test_p['confidence'] = test_conf
    train_p['model_disagreement'] = train_disagree
    test_p['model_disagreement'] = test_disagree
    train_cb, test_cb = encode_categoricals(train_p, test_p, as_category=False)
    train_lgb, test_lgb = encode_categoricals(train_p, test_p, as_category=True)
    return train_cb, test_cb, train_lgb, test_lgb

# --- A. confidence 피처 포함 ---
print("\n--- [A] confidence 피처 포함 8모델 ---")
train_cb2, test_cb2, train_lgb2, test_lgb2 = prepare_data_with_conf(
    train_aug, test_orig, train_conf_aug, test_conf_full,
    train_disagree_aug, test_disagree_full)

# 기존 4모델
oof_conf, tpred_conf, _ = train_4models(
    train_cb2, test_cb2, train_lgb2, test_lgb2, kf2, label="conf기존")

for k in MODELS:
    oof_conf[k] = oof_conf[k][:n_orig]

# 평당가 4모델
y_up_log = np.log1p(train_aug['Target'].values / area_train_aug)
y_up_raw = (train_aug['Target'].values / area_train_aug).astype(float)

oof_conf_unit, tpred_conf_unit, _ = train_4models(
    train_cb2, test_cb2, train_lgb2, test_lgb2, kf2,
    y_log_override=y_up_log, y_raw_override=y_up_raw, label="conf평당가")

for k in MODELS:
    oof_conf_unit[k] = oof_conf_unit[k][:n_orig] * area_train_orig
    tpred_conf_unit[k] = tpred_conf_unit[k] * area_test

# Ridge 스태킹
stack_conf = np.column_stack(
    [oof_conf[k] for k in MODELS] + [oof_conf_unit[k] for k in MODELS])
stack_conf_test = np.column_stack(
    [tpred_conf[k] for k in MODELS] + [tpred_conf_unit[k] for k in MODELS])

best_conf_rmse = float('inf')
best_conf_test = None
best_conf_alpha = None

for alpha in [0.1, 1.0, 5.0, 10.0, 50.0]:
    s_oof = np.zeros(n_orig)
    s_test = np.zeros(len(test_orig))
    for fold, (tr, va) in enumerate(kf.split(stack_conf)):
        meta = Ridge(alpha=alpha)
        meta.fit(stack_conf[tr], y_true_orig[tr])
        s_oof[va] = meta.predict(stack_conf[va])
        s_test += meta.predict(stack_conf_test) / N_SPLITS
    r = np.sqrt(np.mean((s_oof - y_true_orig) ** 2))
    if r < best_conf_rmse:
        best_conf_rmse = r
        best_conf_test = s_test.copy()
        best_conf_alpha = alpha

print(f"  [A] confidence 피처 포함 Ridge(α={best_conf_alpha}): OOF {best_conf_rmse:,.0f}")

# --- B. 기준: confidence 피처 없이 (전략 28 재현) ---
print("\n--- [B] 기준: 전략 28 재현 (confidence 없음) ---")
train_cb3, test_cb3, train_lgb3, test_lgb3 = prepare_data(train_aug, test_orig)

oof_base, tpred_base, _ = train_4models(
    train_cb3, test_cb3, train_lgb3, test_lgb3, kf2, label="기준기존")

for k in MODELS:
    oof_base[k] = oof_base[k][:n_orig]

oof_unit, tpred_unit, _ = train_4models(
    train_cb3, test_cb3, train_lgb3, test_lgb3, kf2,
    y_log_override=y_up_log, y_raw_override=y_up_raw, label="기준평당가")

for k in MODELS:
    oof_unit[k] = oof_unit[k][:n_orig] * area_train_orig
    tpred_unit[k] = tpred_unit[k] * area_test

stack_base = np.column_stack(
    [oof_base[k] for k in MODELS] + [oof_unit[k] for k in MODELS])
stack_base_test = np.column_stack(
    [tpred_base[k] for k in MODELS] + [tpred_unit[k] for k in MODELS])

best_base_rmse = float('inf')
best_base_test = None

for alpha in [0.1, 1.0, 5.0, 10.0, 50.0]:
    s_oof = np.zeros(n_orig)
    s_test = np.zeros(len(test_orig))
    for fold, (tr, va) in enumerate(kf.split(stack_base)):
        meta = Ridge(alpha=alpha)
        meta.fit(stack_base[tr], y_true_orig[tr])
        s_oof[va] = meta.predict(stack_base[va])
        s_test += meta.predict(stack_base_test) / N_SPLITS
    r = np.sqrt(np.mean((s_oof - y_true_orig) ** 2))
    if r < best_base_rmse:
        best_base_rmse = r
        best_base_test = s_test.copy()

print(f"  [B] 기준 (전략 28): OOF {best_base_rmse:,.0f}")

# ========================================
# 결과 비교
# ========================================
print(f"\n{'=' * 60}")
print("결과 비교")
print("=" * 60)
diff = best_conf_rmse - best_base_rmse
print(f"  [A] confidence 피처: OOF {best_conf_rmse:,.0f}")
print(f"  [B] 기준 (전략 28):  OOF {best_base_rmse:,.0f}")
print(f"  차이: {diff:+,.0f} ({'개선' if diff < 0 else '악화'})")

if best_conf_rmse < best_base_rmse:
    final_pred = best_conf_test * trend_correction
    final_rmse = best_conf_rmse
    desc = f'confidence 피처 Ridge(α={best_conf_alpha})'
else:
    final_pred = best_base_test * trend_correction
    final_rmse = best_base_rmse
    desc = '기준 (전략 28)'

sub = sample_sub.copy()
sub['Target'] = final_pred
sub.to_csv(os.path.join(os.path.dirname(__file__), '..', '..', 'submission_l4_42_conf_feat.csv'), index=False)
print(f"\n제출 파일 생성 완료: submission_l4_42_conf_feat.csv ({desc})")

record_result('L4', 42, 'CONF FEAT',
              f'confidence 피처 ({desc}) + GTR', final_rmse, 'tested')
