"""
[L4-27] 평당가 예측 → 면적 환산 (Unit Price Modeling)
──────────────────────────
축약명  : UNIT PRICE
주요 전략: Target/Area를 예측 후 Area 곱해 최종 가격 산출
차별점  : 면적 스케일 분리로 고가 과소예측 완화
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

train, test, sample_sub = load_data()
y_true = train['Target'].values
area_train = train['Exclusive_Area'].values
area_test = test['Exclusive_Area'].values

# === 구별 트렌드 보정 계수 ===
last_train_ym = train['Transaction_YearMonth'].max()
last_train_seq = (last_train_ym // 100 - 2024) * 12 + last_train_ym % 100

gu_growth = {}
for gu in train['Gu'].unique():
    monthly = train[train['Gu'] == gu].groupby('Transaction_YearMonth')['Target'].mean()
    gu_growth[gu] = monthly.pct_change().dropna().mean()

test_seq = (test['Transaction_YearMonth'] // 100 - 2024) * 12 + test['Transaction_YearMonth'] % 100
months_ahead = test_seq.values - last_train_seq
trend_correction = np.array([(1 + gu_growth[gu]) ** m for gu, m in zip(test['Gu'], months_ahead)])

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

# === 전처리 실행 ===
train_p = add_feature_engineering(base_preprocess(train))
test_p = add_feature_engineering(base_preprocess(test))

train_cb, test_cb = encode_categoricals(train_p, test_p, as_category=False)
X_cb = train_cb.drop(columns=['Target'])
X_test_cb = test_cb
cat_indices = [X_cb.columns.get_loc(c) for c in CAT_FEATURES]

train_lgb, test_lgb = encode_categoricals(train_p, test_p, as_category=True)
X_lgb = train_lgb.drop(columns=['Target'])
X_test_lgb = test_lgb

# === 타겟 정의 ===
# 기존: y = Target (or log1p(Target))
# 신규: y = Target / Exclusive_Area (평당가)
unit_price = y_true / area_train
y_up_log = np.log1p(unit_price)
y_up_raw = unit_price

# 기존 타겟도 비교용
y_log = np.log1p(y_true)
y_raw = y_true.copy()

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
MODELS = ['cb_log', 'cb_raw', 'lgb_log', 'lgb_raw']

def run_experiment(y_log_target, y_raw_target, area_for_oof, area_for_test, label):
    """4모델 학습. area가 None이면 expm1만, 있으면 expm1 후 area 곱."""
    oof = {k: np.zeros(len(X_cb)) for k in MODELS}
    tpred = {k: np.zeros(len(X_test_cb)) for k in MODELS}

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_cb)):
        print(f"\n  Fold {fold+1}/{N_SPLITS}")

        # CB log
        m = CatBoostRegressor(loss_function='RMSE', random_seed=42, verbose=0,
                              iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
        m.fit(X_cb.iloc[tr_idx], y_log_target[tr_idx],
              eval_set=(X_cb.iloc[va_idx], y_log_target[va_idx]), cat_features=cat_indices)
        oof['cb_log'][va_idx] = np.expm1(m.predict(X_cb.iloc[va_idx]))
        tpred['cb_log'] += np.expm1(m.predict(X_test_cb)) / N_SPLITS

        # CB raw
        m = CatBoostRegressor(loss_function='RMSE', random_seed=42, verbose=0,
                              iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
        m.fit(X_cb.iloc[tr_idx], y_raw_target[tr_idx],
              eval_set=(X_cb.iloc[va_idx], y_raw_target[va_idx]), cat_features=cat_indices)
        oof['cb_raw'][va_idx] = m.predict(X_cb.iloc[va_idx])
        tpred['cb_raw'] += m.predict(X_test_cb) / N_SPLITS

        # LGB log
        m = lgb.LGBMRegressor(objective='regression', metric='rmse',
                              verbose=-1, random_state=42, n_estimators=3000, **LGB_PARAMS)
        m.fit(X_lgb.iloc[tr_idx], y_log_target[tr_idx],
              eval_set=[(X_lgb.iloc[va_idx], y_log_target[va_idx])],
              callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])
        oof['lgb_log'][va_idx] = np.expm1(m.predict(X_lgb.iloc[va_idx]))
        tpred['lgb_log'] += np.expm1(m.predict(X_test_lgb)) / N_SPLITS

        # LGB raw
        m = lgb.LGBMRegressor(objective='regression', metric='rmse',
                              verbose=-1, random_state=42, n_estimators=3000, **LGB_PARAMS)
        m.fit(X_lgb.iloc[tr_idx], y_raw_target[tr_idx],
              eval_set=[(X_lgb.iloc[va_idx], y_raw_target[va_idx])],
              callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])
        oof['lgb_raw'][va_idx] = m.predict(X_lgb.iloc[va_idx])
        tpred['lgb_raw'] += m.predict(X_test_lgb) / N_SPLITS

    # 면적 환산 (평당가 → 총가격)
    if area_for_oof is not None:
        for k in MODELS:
            oof[k] = oof[k] * area_for_oof
            tpred[k] = tpred[k] * area_for_test

    # 개별 모델 RMSE
    print(f"\n{'─' * 50}")
    print(f"  [{label}] 개별 모델 OOF RMSE (총가격 기준)")
    for k in MODELS:
        rmse = np.sqrt(np.mean((oof[k] - y_true) ** 2))
        print(f"    {k:10s}: {rmse:,.0f}")

    return oof, tpred


# ========================================
# 실험 1: 기존 방식 (Target 직접 예측) — baseline 재현
# ========================================
print("=" * 60)
print("[실험 1] 기존 방식: Target 직접 예측")
print("=" * 60)

oof_base, tpred_base = run_experiment(
    y_log, y_raw.astype(float), area_for_oof=None, area_for_test=None, label="기존")

# ========================================
# 실험 2: 평당가 예측 → 면적 환산
# ========================================
print(f"\n{'=' * 60}")
print("[실험 2] 평당가 예측: Target/Area → ×Area")
print("=" * 60)

oof_unit, tpred_unit = run_experiment(
    y_up_log, y_up_raw, area_for_oof=area_train, area_for_test=area_test, label="평당가")


# ========================================
# 블렌딩 비교
# ========================================
def eval_blend(oof_dict, label):
    """여러 블렌딩 방법의 OOF RMSE 비교, 최선 반환"""
    results = {}

    # 단순 평균
    avg = np.mean([oof_dict[k] for k in MODELS], axis=0)
    rmse_avg = np.sqrt(np.mean((avg - y_true) ** 2))
    results['avg'] = rmse_avg

    # 가중 평균 (원본 60%)
    w = 0.20*oof_dict['cb_log'] + 0.30*oof_dict['cb_raw'] + 0.20*oof_dict['lgb_log'] + 0.30*oof_dict['lgb_raw']
    rmse_w = np.sqrt(np.mean((w - y_true) ** 2))
    results['w60'] = rmse_w

    print(f"\n  [{label}] 블렌딩 OOF RMSE")
    print(f"    단순 평균  : {rmse_avg:,.0f}")
    print(f"    가중(원본60%): {rmse_w:,.0f}")

    return results

print(f"\n{'=' * 60}")
print("블렌딩 비교")
print("=" * 60)

res_base = eval_blend(oof_base, "기존")
res_unit = eval_blend(oof_unit, "평당가")

# ========================================
# 실험 3: 기존 + 평당가 크로스 블렌딩 (8모델)
# ========================================
print(f"\n{'=' * 60}")
print("[실험 3] 크로스 블렌딩: 기존 4모델 + 평당가 4모델 = 8모델")
print("=" * 60)

# 8모델 단순 평균
all_oof = np.mean([oof_base[k] for k in MODELS] + [oof_unit[k] for k in MODELS], axis=0)
rmse_all = np.sqrt(np.mean((all_oof - y_true) ** 2))
print(f"  8모델 단순 평균 OOF RMSE: {rmse_all:,.0f}")

# Ridge 스태킹 (8모델)
stack_train = np.column_stack([oof_base[k] for k in MODELS] + [oof_unit[k] for k in MODELS])
stack_test = np.column_stack([tpred_base[k] for k in MODELS] + [tpred_unit[k] for k in MODELS])

best_ridge_rmse = float('inf')
best_ridge_alpha = None
best_ridge_oof = None
best_ridge_test = None

for alpha in [0.1, 0.5, 1.0, 5.0, 10.0]:
    s_oof = np.zeros(len(y_true))
    s_test = np.zeros(len(X_test_cb))

    for fold, (tr_idx, va_idx) in enumerate(kf.split(stack_train)):
        meta = Ridge(alpha=alpha)
        meta.fit(stack_train[tr_idx], y_true[tr_idx])
        s_oof[va_idx] = meta.predict(stack_train[va_idx])
        s_test += meta.predict(stack_test) / N_SPLITS

    ridge_rmse = np.sqrt(np.mean((s_oof - y_true) ** 2))
    print(f"  Ridge(α={alpha:5.1f}) OOF RMSE: {ridge_rmse:,.0f}")

    if ridge_rmse < best_ridge_rmse:
        best_ridge_rmse = ridge_rmse
        best_ridge_alpha = alpha
        best_ridge_oof = s_oof.copy()
        best_ridge_test = s_test.copy()

# ========================================
# 최종 비교
# ========================================
print(f"\n{'=' * 60}")
print("최종 비교")
print("=" * 60)
print(f"  기존 4모델 단순평균     : {res_base['avg']:,.0f}")
print(f"  기존 4모델 가중(원본60%): {res_base['w60']:,.0f}")
print(f"  평당가 4모델 단순평균   : {res_unit['avg']:,.0f}")
print(f"  평당가 4모델 가중(원본60%): {res_unit['w60']:,.0f}")
print(f"  8모델 단순 평균         : {rmse_all:,.0f}")
print(f"  8모델 Ridge(α={best_ridge_alpha})    : {best_ridge_rmse:,.0f}")
print(f"  ────────────────────────────────")
print(f"  전략 25 (기존 최선)     : 2,226")
print(f"  전략 08 (Public 최선)   : 2,234")

# 최선 선택
options = {
    'base_avg': (np.mean([tpred_base[k] for k in MODELS], axis=0), res_base['avg'], '기존 단순평균'),
    'unit_avg': (np.mean([tpred_unit[k] for k in MODELS], axis=0), res_unit['avg'], '평당가 단순평균'),
    'cross_avg': (np.mean([tpred_base[k] for k in MODELS] + [tpred_unit[k] for k in MODELS], axis=0), rmse_all, '8모델 단순평균'),
    'cross_ridge': (best_ridge_test, best_ridge_rmse, f'8모델 Ridge(α={best_ridge_alpha})'),
}

best_key = min(options, key=lambda k: options[k][1])
best_test_pred, best_rmse, best_desc = options[best_key]

final_pred = best_test_pred * trend_correction

print(f"\n  ★ 최선: [{best_key}] {best_desc} → OOF {best_rmse:,.0f}")

# === 제출 파일 ===
sub = sample_sub.copy()
sub['Target'] = final_pred
sub.to_csv(os.path.join(os.path.dirname(__file__), '..', '..', 'submission_l4_27_unit_price.csv'), index=False)
print("제출 파일 생성 완료: submission_l4_27_unit_price.csv")

record_result('L4', 27, 'UNIT PRICE',
              f'평당가 예측 ({best_desc}) + 구별 트렌드', best_rmse, 'tested')
