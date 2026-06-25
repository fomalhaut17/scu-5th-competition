"""
[L4-33] Meta 입력에 예측 요약 통계 추가
──────────────────────────
축약명  : META SUMMARY
주요 전략: 8모델 예측 + 요약 통계 5개 = 13개 입력 Ridge
차별점  : 모델 disagreement 정보를 메타에 전달
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, ElasticNet
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


def build_summary_features(base_preds, unit_preds):
    """8모델 예측에서 요약 통계 5개 생성"""
    all_8 = np.column_stack([base_preds[k] for k in MODELS] + [unit_preds[k] for k in MODELS])
    base_4 = np.column_stack([base_preds[k] for k in MODELS])
    unit_4 = np.column_stack([unit_preds[k] for k in MODELS])
    raw_2 = np.column_stack([base_preds['cb_raw'], base_preds['lgb_raw']])
    log_2 = np.column_stack([base_preds['cb_log'], base_preds['lgb_log']])

    feat_mean = np.mean(all_8, axis=1)
    feat_std = np.std(all_8, axis=1)
    feat_base_unit_diff = np.mean(base_4, axis=1) - np.mean(unit_4, axis=1)
    feat_raw_log_diff = np.mean(raw_2, axis=1) - np.mean(log_2, axis=1)
    feat_unit_base_ratio = np.mean(unit_4, axis=1) / (np.mean(base_4, axis=1) + 1e-8)

    return np.column_stack([feat_mean, feat_std, feat_base_unit_diff, feat_raw_log_diff, feat_unit_base_ratio])


# ========================================
# Stage 1: PL 신뢰도 측정
# ========================================
print("=" * 60)
print("[Stage 1] 원본 4모델 → PL 신뢰도")
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
print(f"  신뢰도: mean={confidence.mean():.4f}")

# ========================================
# Stage 2: PL(50%) → 8모델
# ========================================
threshold = np.percentile(confidence, 50)
mask = confidence >= threshold
n_pseudo = mask.sum()

print(f"\n{'=' * 60}")
print(f"[Stage 2] PL2 + 8모델 ({n_pseudo}건 PL)")
print("=" * 60)

test_selected = test_orig[mask].copy()
test_selected['Target'] = pseudo_labels[mask]
train_aug = pd.concat([train_orig, test_selected], ignore_index=True)
area_train_aug = train_aug['Exclusive_Area'].values

kf2 = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
train_cb2, test_cb2, train_lgb2, test_lgb2 = prepare_data(train_aug, test_orig)

# 기존 4모델
print("\n--- 기존 4모델 ---")
oof_base, tpred_base, _ = train_4models(
    train_cb2, test_cb2, train_lgb2, test_lgb2, kf2, label="기존")
for k in MODELS:
    oof_base[k] = oof_base[k][:n_orig]

# 평당가 4모델
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
# 메타 스태킹 비교
# ========================================
print(f"\n{'=' * 60}")
print("메타 스태킹 비교")
print("=" * 60)

# 기본 8개 입력
stack_8_train = np.column_stack([oof_base[k] for k in MODELS] + [oof_unit[k] for k in MODELS])
stack_8_test = np.column_stack([tpred_base[k] for k in MODELS] + [tpred_unit[k] for k in MODELS])

# 요약 통계 5개 추가 → 13개 입력
summary_train = build_summary_features(oof_base, oof_unit)
summary_test = build_summary_features(tpred_base, tpred_unit)

stack_13_train = np.column_stack([stack_8_train, summary_train])
stack_13_test = np.column_stack([stack_8_test, summary_test])

kf_meta = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# --- A: 기존 8입력 Ridge (전략 28 재현) ---
print("\n[A] 기존 8입력 Ridge")
best_8_rmse = float('inf')
best_8_test = None
for alpha in [1.0, 5.0, 10.0, 50.0]:
    s_oof = np.zeros(n_orig)
    s_test = np.zeros(len(test_orig))
    for fold, (tr_idx, va_idx) in enumerate(kf_meta.split(stack_8_train)):
        meta = Ridge(alpha=alpha)
        meta.fit(stack_8_train[tr_idx], y_true_orig[tr_idx])
        s_oof[va_idx] = meta.predict(stack_8_train[va_idx])
        s_test += meta.predict(stack_8_test) / N_SPLITS
    rmse = np.sqrt(np.mean((s_oof - y_true_orig) ** 2))
    print(f"  Ridge(α={alpha:5.1f}): {rmse:,.0f}")
    if rmse < best_8_rmse:
        best_8_rmse = rmse
        best_8_test = s_test.copy()

# --- B: 13입력 Ridge ---
print("\n[B] 13입력 Ridge (8예측 + 5요약)")
best_13_rmse = float('inf')
best_13_test = None
best_13_alpha = None
for alpha in [1.0, 5.0, 10.0, 50.0, 100.0]:
    s_oof = np.zeros(n_orig)
    s_test = np.zeros(len(test_orig))
    for fold, (tr_idx, va_idx) in enumerate(kf_meta.split(stack_13_train)):
        meta = Ridge(alpha=alpha)
        meta.fit(stack_13_train[tr_idx], y_true_orig[tr_idx])
        s_oof[va_idx] = meta.predict(stack_13_train[va_idx])
        s_test += meta.predict(stack_13_test) / N_SPLITS
    rmse = np.sqrt(np.mean((s_oof - y_true_orig) ** 2))
    print(f"  Ridge(α={alpha:5.1f}): {rmse:,.0f}")
    if rmse < best_13_rmse:
        best_13_rmse = rmse
        best_13_test = s_test.copy()
        best_13_alpha = alpha

# --- C: 13입력 ElasticNet ---
print("\n[C] 13입력 ElasticNet")
best_en_rmse = float('inf')
best_en_test = None
for alpha in [0.1, 1.0, 10.0]:
    for l1 in [0.1, 0.3, 0.5]:
        s_oof = np.zeros(n_orig)
        s_test = np.zeros(len(test_orig))
        for fold, (tr_idx, va_idx) in enumerate(kf_meta.split(stack_13_train)):
            meta = ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=10000)
            meta.fit(stack_13_train[tr_idx], y_true_orig[tr_idx])
            s_oof[va_idx] = meta.predict(stack_13_train[va_idx])
            s_test += meta.predict(stack_13_test) / N_SPLITS
        rmse = np.sqrt(np.mean((s_oof - y_true_orig) ** 2))
        if rmse < best_en_rmse:
            best_en_rmse = rmse
            best_en_test = s_test.copy()

print(f"  Best ElasticNet: {best_en_rmse:,.0f}")

# --- D: 요약 통계만 사용하는 서브셋 테스트 ---
print("\n[D] 요약 통계 개별 기여도")
summary_names = ['mean', 'std', 'base-unit diff', 'raw-log diff', 'unit/base ratio']
for i, name in enumerate(summary_names):
    stack_9 = np.column_stack([stack_8_train, summary_train[:, i]])
    stack_9_te = np.column_stack([stack_8_test, summary_test[:, i]])
    s_oof = np.zeros(n_orig)
    for fold, (tr_idx, va_idx) in enumerate(kf_meta.split(stack_9)):
        meta = Ridge(alpha=10.0)
        meta.fit(stack_9[tr_idx], y_true_orig[tr_idx])
        s_oof[va_idx] = meta.predict(stack_9[va_idx])
    rmse = np.sqrt(np.mean((s_oof - y_true_orig) ** 2))
    diff = rmse - best_8_rmse
    print(f"  8+{name:18s}: {rmse:,.0f} ({diff:+.0f})")

# ========================================
# 최종 비교
# ========================================
print(f"\n{'=' * 60}")
print("최종 비교")
print("=" * 60)

avg_oof = np.mean([oof_base[k] for k in MODELS] + [oof_unit[k] for k in MODELS], axis=0)
avg_test = np.mean([tpred_base[k] for k in MODELS] + [tpred_unit[k] for k in MODELS], axis=0)
rmse_avg = np.sqrt(np.mean((avg_oof - y_true_orig) ** 2))

print(f"  8모델 단순평균         : {rmse_avg:,.0f}")
print(f"  [A] 8입력 Ridge       : {best_8_rmse:,.0f}  (전략 28)")
print(f"  [B] 13입력 Ridge(α={best_13_alpha}): {best_13_rmse:,.0f}")
print(f"  [C] 13입력 ElasticNet : {best_en_rmse:,.0f}")
print(f"  ─────────────────────────────")
print(f"  전략 28               : OOF 2,196 / Public 2,096.8")

# 최선 선택
options = {
    '8avg': (avg_test, rmse_avg, '8모델 단순평균'),
    '8ridge': (best_8_test, best_8_rmse, '8입력 Ridge'),
    '13ridge': (best_13_test, best_13_rmse, f'13입력 Ridge(α={best_13_alpha})'),
    '13en': (best_en_test, best_en_rmse, '13입력 ElasticNet'),
}

best_key = min(options, key=lambda k: options[k][1])
best_test_pred, best_rmse, best_desc = options[best_key]

final_pred = best_test_pred * trend_correction

print(f"\n  ★ 최선: [{best_key}] {best_desc} → OOF {best_rmse:,.0f}")

# === 제출 파일 ===
sub = sample_sub.copy()
sub['Target'] = final_pred
sub.to_csv(os.path.join(os.path.dirname(__file__), '..', '..', 'submission_l4_33_meta_summary.csv'), index=False)
print(f"제출 파일 생성 완료: submission_l4_33_meta_summary.csv")

record_result('L4', 33, 'META SUMMARY',
              f'Meta 요약 통계 ({best_desc}) + GTR', best_rmse, 'tested')
