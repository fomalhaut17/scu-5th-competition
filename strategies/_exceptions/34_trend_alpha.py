"""
[L4-34] 트렌드 보정 강도 최적화
──────────────────────────
축약명  : TREND ALPHA
주요 전략: 전략 28 구조 유지, 보정 계수 alpha 스윕
차별점  : (1+alpha*g)^m 에서 alpha를 0.5~1.5 탐색 + 단리 비교
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

# === 구별 성장률 계산 ===
last_train_ym = train_orig['Transaction_YearMonth'].max()
last_train_seq = (last_train_ym // 100 - 2024) * 12 + last_train_ym % 100

gu_growth = {}
for gu in train_orig['Gu'].unique():
    monthly = train_orig[train_orig['Gu'] == gu].groupby('Transaction_YearMonth')['Target'].mean()
    gu_growth[gu] = monthly.pct_change().dropna().mean()

test_seq = (test_orig['Transaction_YearMonth'] // 100 - 2024) * 12 + test_orig['Transaction_YearMonth'] % 100
months_ahead = test_seq.values - last_train_seq
test_gu = test_orig['Gu'].values

print("=== 구별 월성장률 ===")
for gu, g in sorted(gu_growth.items(), key=lambda x: -x[1]):
    print(f"  {gu:15s}: {g*100:+.2f}%")

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
                  y_log_override=None, y_raw_override=None, label=""):
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

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_cb)):
        print(f"  [{label}] Fold {fold+1}/{kf.n_splits}")

        m = CatBoostRegressor(loss_function='RMSE', random_seed=42, verbose=0,
                              iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
        m.fit(X_cb.iloc[tr_idx], y_log.iloc[tr_idx] if hasattr(y_log, 'iloc') else y_log[tr_idx],
              eval_set=(X_cb.iloc[va_idx], y_log.iloc[va_idx] if hasattr(y_log, 'iloc') else y_log[va_idx]),
              cat_features=cat_idx)
        oof['cb_log'][va_idx] = np.expm1(m.predict(X_cb.iloc[va_idx]))
        tpred['cb_log'] += np.expm1(m.predict(X_test_cb)) / kf.n_splits

        m = CatBoostRegressor(loss_function='RMSE', random_seed=42, verbose=0,
                              iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
        m.fit(X_cb.iloc[tr_idx], y_raw[tr_idx],
              eval_set=(X_cb.iloc[va_idx], y_raw[va_idx]), cat_features=cat_idx)
        oof['cb_raw'][va_idx] = m.predict(X_cb.iloc[va_idx])
        tpred['cb_raw'] += m.predict(X_test_cb) / kf.n_splits

        m = lgb.LGBMRegressor(objective='regression', metric='rmse',
                              verbose=-1, random_state=42, n_estimators=3000, **LGB_PARAMS)
        m.fit(X_lgb.iloc[tr_idx], y_log.iloc[tr_idx] if hasattr(y_log, 'iloc') else y_log[tr_idx],
              eval_set=[(X_lgb.iloc[va_idx], y_log.iloc[va_idx] if hasattr(y_log, 'iloc') else y_log[va_idx])],
              callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])
        oof['lgb_log'][va_idx] = np.expm1(m.predict(X_lgb.iloc[va_idx]))
        tpred['lgb_log'] += np.expm1(m.predict(X_test_lgb)) / kf.n_splits

        m = lgb.LGBMRegressor(objective='regression', metric='rmse',
                              verbose=-1, random_state=42, n_estimators=3000, **LGB_PARAMS)
        m.fit(X_lgb.iloc[tr_idx], y_raw[tr_idx],
              eval_set=[(X_lgb.iloc[va_idx], y_raw[va_idx])],
              callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])
        oof['lgb_raw'][va_idx] = m.predict(X_lgb.iloc[va_idx])
        tpred['lgb_raw'] += m.predict(X_test_lgb) / kf.n_splits

    return oof, tpred


# ========================================
# 전략 28 파이프라인 실행 (보정 전까지)
# ========================================
print("=" * 60)
print("[Stage 1] PL 신뢰도 측정")
print("=" * 60)

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
train_cb, test_cb, train_lgb, test_lgb = prepare_data(train_orig, test_orig)

oof_s1, tpred_s1 = train_4models(train_cb, test_cb, train_lgb, test_lgb, kf, label="S1")

pseudo_labels = np.mean([tpred_s1[k] for k in MODELS], axis=0)
model_means = np.array([tpred_s1[k] for k in MODELS])
model_disagreement = np.std(model_means, axis=0) / np.mean(model_means, axis=0)
fold_cvs = []
kf_temp = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
# fold preds를 다시 계산하지 않고 confidence는 model disagreement만으로 근사
confidence = 1 - model_disagreement

threshold = np.percentile(confidence, 50)
mask = confidence >= threshold
n_pseudo = mask.sum()

print(f"\n{'=' * 60}")
print(f"[Stage 2] PL2 + 8모델 ({n_pseudo}건)")
print("=" * 60)

test_selected = test_orig[mask].copy()
test_selected['Target'] = pseudo_labels[mask]
train_aug = pd.concat([train_orig, test_selected], ignore_index=True)
area_train_aug = train_aug['Exclusive_Area'].values

kf2 = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
train_cb2, test_cb2, train_lgb2, test_lgb2 = prepare_data(train_aug, test_orig)

oof_base, tpred_base = train_4models(train_cb2, test_cb2, train_lgb2, test_lgb2, kf2, label="기존")
for k in MODELS:
    oof_base[k] = oof_base[k][:n_orig]

y_up_log = np.log1p(train_aug['Target'].values / area_train_aug)
y_up_raw = (train_aug['Target'].values / area_train_aug).astype(float)

oof_unit, tpred_unit = train_4models(
    train_cb2, test_cb2, train_lgb2, test_lgb2, kf2,
    y_log_override=y_up_log, y_raw_override=y_up_raw, label="평당가")
for k in MODELS:
    oof_unit[k] = oof_unit[k][:n_orig] * area_train_orig
    tpred_unit[k] = tpred_unit[k] * area_test

# Ridge 스태킹 (보정 전 예측)
kf_meta = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
stack_train = np.column_stack([oof_base[k] for k in MODELS] + [oof_unit[k] for k in MODELS])
stack_test = np.column_stack([tpred_base[k] for k in MODELS] + [tpred_unit[k] for k in MODELS])

best_ridge_test = None
best_ridge_rmse = float('inf')
for alpha in [1.0, 10.0, 50.0]:
    s_oof = np.zeros(n_orig)
    s_test = np.zeros(len(test_orig))
    for fold, (tr_idx, va_idx) in enumerate(kf_meta.split(stack_train)):
        meta = Ridge(alpha=alpha)
        meta.fit(stack_train[tr_idx], y_true_orig[tr_idx])
        s_oof[va_idx] = meta.predict(stack_train[va_idx])
        s_test += meta.predict(stack_test) / N_SPLITS
    rmse = np.sqrt(np.mean((s_oof - y_true_orig) ** 2))
    if rmse < best_ridge_rmse:
        best_ridge_rmse = rmse
        best_ridge_test = s_test.copy()

# 단순 평균도 비교
avg_test = np.mean([tpred_base[k] for k in MODELS] + [tpred_unit[k] for k in MODELS], axis=0)
avg_oof = np.mean([oof_base[k] for k in MODELS] + [oof_unit[k] for k in MODELS], axis=0)
rmse_avg = np.sqrt(np.mean((avg_oof - y_true_orig) ** 2))

if rmse_avg < best_ridge_rmse:
    raw_pred = avg_test
    base_rmse = rmse_avg
    base_method = "단순평균"
else:
    raw_pred = best_ridge_test
    base_rmse = best_ridge_rmse
    base_method = "Ridge"

print(f"\n  보정 전 ({base_method}) OOF: {base_rmse:,.0f}")

# ========================================
# 트렌드 보정 강도 스윕
# ========================================
print(f"\n{'=' * 60}")
print("트렌드 보정 강도 스윕")
print("=" * 60)

# 보정은 Test에만 적용되므로 OOF로 직접 비교 불가.
# 대신 보정 후 예측의 통계를 비교.

print(f"\n  [복리] (1 + alpha*g)^m")
for alpha in [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0]:
    correction = np.array([(1 + alpha * gu_growth[gu]) ** m for gu, m in zip(test_gu, months_ahead)])
    pred = raw_pred * correction
    print(f"    α={alpha:.1f}: 보정범위 +{(correction.min()-1)*100:.1f}%~+{(correction.max()-1)*100:.1f}%, 평균 {pred.mean():,.0f}, std {pred.std():,.0f}")

print(f"\n  [단리] 1 + alpha*g*m")
for alpha in [0.0, 0.5, 0.7, 1.0, 1.3, 1.5]:
    correction = np.array([1 + alpha * gu_growth[gu] * m for gu, m in zip(test_gu, months_ahead)])
    pred = raw_pred * correction
    print(f"    α={alpha:.1f}: 보정범위 +{(correction.min()-1)*100:.1f}%~+{(correction.max()-1)*100:.1f}%, 평균 {pred.mean():,.0f}, std {pred.std():,.0f}")

# ========================================
# 제출 파일 생성 (여러 alpha)
# ========================================
print(f"\n{'=' * 60}")
print("제출 파일 생성")
print("=" * 60)

# 기존 전략 28: alpha=1.0 복리
candidates = [
    (0.7, 'compound', '복리 α=0.7'),
    (0.8, 'compound', '복리 α=0.8'),
    (1.0, 'compound', '복리 α=1.0 (전략 28 동일)'),
    (1.2, 'compound', '복리 α=1.2'),
    (1.0, 'simple', '단리 α=1.0'),
]

for alpha, method, label in candidates:
    if method == 'compound':
        correction = np.array([(1 + alpha * gu_growth[gu]) ** m for gu, m in zip(test_gu, months_ahead)])
    else:
        correction = np.array([1 + alpha * gu_growth[gu] * m for gu, m in zip(test_gu, months_ahead)])

    pred = raw_pred * correction
    fname = f'submission_l4_34_trend_a{alpha:.1f}_{method}.csv'
    sub = sample_sub.copy()
    sub['Target'] = pred
    sub.to_csv(os.path.join(os.path.dirname(__file__), '..', '..', fname), index=False)
    print(f"  {label}: {fname} (평균 {pred.mean():,.0f})")

print(f"\n  전략 28 (α=1.0 복리) 참고: Public 2,096.8")
print(f"  보정은 OOF에 반영되지 않으므로 Public 제출로만 비교 가능")

record_result('L4', 34, 'TREND ALPHA',
              '트렌드 보정 강도 최적화 (alpha sweep)', base_rmse, 'tested')
