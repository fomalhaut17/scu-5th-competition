"""
[L4-29] 타겟 변환 다양성 실험
──────────────────────────
축약명  : TARGET TRANSFORMS
주요 전략: 기존 8모델(기존+평당가) + 추가 타겟 변환 모델로 다양성 확장
실험: Target/Floor(층당가), sqrt(Target), Target/(Area×Floor) 각각 테스트
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

train, test, sample_sub = load_data()
y_true = train['Target'].values
area_train = train['Exclusive_Area'].values
area_test = test['Exclusive_Area'].values
floor_train = train['Floor'].values.astype(float)
floor_test = test['Floor'].values.astype(float)

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

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

def train_4models(y_log_t, y_raw_t, label):
    oof = {k: np.zeros(len(X_cb)) for k in MODELS}
    tpred = {k: np.zeros(len(X_test_cb)) for k in MODELS}

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_cb)):
        print(f"  [{label}] Fold {fold+1}/{N_SPLITS}")

        m = CatBoostRegressor(loss_function='RMSE', random_seed=42, verbose=0,
                              iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
        m.fit(X_cb.iloc[tr_idx], y_log_t[tr_idx],
              eval_set=(X_cb.iloc[va_idx], y_log_t[va_idx]), cat_features=cat_indices)
        oof['cb_log'][va_idx] = np.expm1(m.predict(X_cb.iloc[va_idx]))
        tpred['cb_log'] += np.expm1(m.predict(X_test_cb)) / N_SPLITS

        m = CatBoostRegressor(loss_function='RMSE', random_seed=42, verbose=0,
                              iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
        m.fit(X_cb.iloc[tr_idx], y_raw_t[tr_idx],
              eval_set=(X_cb.iloc[va_idx], y_raw_t[va_idx]), cat_features=cat_indices)
        oof['cb_raw'][va_idx] = m.predict(X_cb.iloc[va_idx])
        tpred['cb_raw'] += m.predict(X_test_cb) / N_SPLITS

        m = lgb.LGBMRegressor(objective='regression', metric='rmse',
                              verbose=-1, random_state=42, n_estimators=3000, **LGB_PARAMS)
        m.fit(X_lgb.iloc[tr_idx], y_log_t[tr_idx],
              eval_set=[(X_lgb.iloc[va_idx], y_log_t[va_idx])],
              callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])
        oof['lgb_log'][va_idx] = np.expm1(m.predict(X_lgb.iloc[va_idx]))
        tpred['lgb_log'] += np.expm1(m.predict(X_test_lgb)) / N_SPLITS

        m = lgb.LGBMRegressor(objective='regression', metric='rmse',
                              verbose=-1, random_state=42, n_estimators=3000, **LGB_PARAMS)
        m.fit(X_lgb.iloc[tr_idx], y_raw_t[tr_idx],
              eval_set=[(X_lgb.iloc[va_idx], y_raw_t[va_idx])],
              callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])
        oof['lgb_raw'][va_idx] = m.predict(X_lgb.iloc[va_idx])
        tpred['lgb_raw'] += m.predict(X_test_lgb) / N_SPLITS

    return oof, tpred

# ========================================
# 기존 8모델 (기존 + 평당가)
# ========================================
print("=" * 60)
print("[Part 1] 기존 4모델")
print("=" * 60)
y_log = np.log1p(y_true)
y_raw = y_true.astype(float)
oof_base, tpred_base = train_4models(y_log, y_raw, "기존")

print(f"\n{'=' * 60}")
print("[Part 2] 평당가 4모델")
print("=" * 60)
up = y_true / area_train
oof_up, tpred_up = train_4models(np.log1p(up), up.astype(float), "평당가")
for k in MODELS:
    oof_up[k] = oof_up[k] * area_train
    tpred_up[k] = tpred_up[k] * area_test

# ========================================
# 추가 타겟 변환 4모델씩
# ========================================

# T1: 층당가 (Target / Floor)
print(f"\n{'=' * 60}")
print("[Part 3] 층당가 4모델 (Target / Floor)")
print("=" * 60)
pf = y_true / floor_train
oof_pf, tpred_pf = train_4models(np.log1p(pf), pf.astype(float), "층당가")
for k in MODELS:
    oof_pf[k] = oof_pf[k] * floor_train
    tpred_pf[k] = tpred_pf[k] * floor_test

# T2: sqrt(Target)
print(f"\n{'=' * 60}")
print("[Part 4] sqrt 4모델 (sqrt(Target))")
print("=" * 60)
sq = np.sqrt(y_true)
oof_sq, tpred_sq = train_4models(np.log1p(sq), sq.astype(float), "sqrt")
for k in MODELS:
    oof_sq[k] = oof_sq[k] ** 2
    tpred_sq[k] = tpred_sq[k] ** 2

# T3: Target / (Area × Floor)
print(f"\n{'=' * 60}")
print("[Part 5] 면적층가 4모델 (Target / (Area × Floor))")
print("=" * 60)
af = y_true / (area_train * floor_train)
oof_af, tpred_af = train_4models(np.log1p(af), af.astype(float), "면적층가")
for k in MODELS:
    oof_af[k] = oof_af[k] * (area_train * floor_train)
    tpred_af[k] = tpred_af[k] * (area_test * floor_test)

# ========================================
# 개별 변환 성능 확인
# ========================================
print(f"\n{'=' * 60}")
print("개별 변환 4모델 단순평균 OOF RMSE")
print("=" * 60)

groups = {
    '기존': oof_base,
    '평당가': oof_up,
    '층당가': oof_pf,
    'sqrt': oof_sq,
    '면적층가': oof_af,
}

for name, oof in groups.items():
    avg = np.mean([oof[k] for k in MODELS], axis=0)
    rmse = np.sqrt(np.mean((avg - y_true) ** 2))
    print(f"  {name:8s}: {rmse:,.0f}")

# ========================================
# 크로스 블렌딩 실험: 8모델 + 각 변환 4모델 추가
# ========================================
print(f"\n{'=' * 60}")
print("크로스 블렌딩: 기존 8모델 + 추가 변환")
print("=" * 60)

base_8_oof = [oof_base[k] for k in MODELS] + [oof_up[k] for k in MODELS]
base_8_test = [tpred_base[k] for k in MODELS] + [tpred_up[k] for k in MODELS]

base_8_avg = np.mean(base_8_oof, axis=0)
rmse_8 = np.sqrt(np.mean((base_8_avg - y_true) ** 2))
print(f"  기존 8모델 단순평균: {rmse_8:,.0f}")

extra_groups = {
    '층당가': (oof_pf, tpred_pf),
    'sqrt': (oof_sq, tpred_sq),
    '면적층가': (oof_af, tpred_af),
}

results = {}

for name, (oof_extra, tpred_extra) in extra_groups.items():
    # 8 + 4 = 12모델
    all_oof = base_8_oof + [oof_extra[k] for k in MODELS]
    all_test = base_8_test + [tpred_extra[k] for k in MODELS]

    avg_oof = np.mean(all_oof, axis=0)
    rmse_avg = np.sqrt(np.mean((avg_oof - y_true) ** 2))

    # Ridge
    stack_tr = np.column_stack(all_oof)
    stack_te = np.column_stack(all_test)
    best_r_rmse = float('inf')
    best_r_test = None
    for alpha in [0.1, 1.0, 10.0]:
        s_oof = np.zeros(len(y_true))
        s_test = np.zeros(len(X_test_cb))
        for fold, (tr_idx, va_idx) in enumerate(kf.split(stack_tr)):
            meta = Ridge(alpha=alpha)
            meta.fit(stack_tr[tr_idx], y_true[tr_idx])
            s_oof[va_idx] = meta.predict(stack_tr[va_idx])
            s_test += meta.predict(stack_te) / N_SPLITS
        r_rmse = np.sqrt(np.mean((s_oof - y_true) ** 2))
        if r_rmse < best_r_rmse:
            best_r_rmse = r_rmse
            best_r_test = s_test.copy()

    print(f"  +{name:6s} (12모델): 평균 {rmse_avg:,.0f} / Ridge {best_r_rmse:,.0f}")
    results[name] = {'avg': rmse_avg, 'ridge': best_r_rmse, 'test': best_r_test}

# 전부 합치기: 8 + 4 + 4 + 4 = 20모델
print(f"\n{'=' * 60}")
print("전체 합산: 20모델 (기존+평당가+층당가+sqrt+면적층가)")
print("=" * 60)

all_20_oof = base_8_oof
all_20_test = base_8_test
for name, (oof_extra, tpred_extra) in extra_groups.items():
    all_20_oof = all_20_oof + [oof_extra[k] for k in MODELS]
    all_20_test = all_20_test + [tpred_extra[k] for k in MODELS]

avg_20_oof = np.mean(all_20_oof, axis=0)
rmse_20_avg = np.sqrt(np.mean((avg_20_oof - y_true) ** 2))

stack_20_tr = np.column_stack(all_20_oof)
stack_20_te = np.column_stack(all_20_test)

best_20_rmse = float('inf')
best_20_alpha = None
best_20_test = None
for alpha in [0.1, 0.5, 1.0, 5.0, 10.0, 50.0]:
    s_oof = np.zeros(len(y_true))
    s_test = np.zeros(len(X_test_cb))
    for fold, (tr_idx, va_idx) in enumerate(kf.split(stack_20_tr)):
        meta = Ridge(alpha=alpha)
        meta.fit(stack_20_tr[tr_idx], y_true[tr_idx])
        s_oof[va_idx] = meta.predict(stack_20_tr[va_idx])
        s_test += meta.predict(stack_20_te) / N_SPLITS
    r_rmse = np.sqrt(np.mean((s_oof - y_true) ** 2))
    print(f"  Ridge(α={alpha:5.1f}) OOF RMSE: {r_rmse:,.0f}")
    if r_rmse < best_20_rmse:
        best_20_rmse = r_rmse
        best_20_alpha = alpha
        best_20_test = s_test.copy()

print(f"\n  20모델 단순평균 OOF: {rmse_20_avg:,.0f}")
print(f"  20모델 Ridge(α={best_20_alpha}) OOF: {best_20_rmse:,.0f}")
print(f"  ─────────────────────────────")
print(f"  기존 8모델 단순평균: {rmse_8:,.0f}")
print(f"  전략 28 (PL2+8모델): OOF 2,196 / Public 2,096.8")

# 유용한 변환만 선별: 8 + 유효한 것들
print(f"\n{'=' * 60}")
print("선별 조합 테스트")
print("=" * 60)

from itertools import combinations

best_combo_rmse = float('inf')
best_combo_name = None
best_combo_test = None

for r in range(1, 4):
    for combo in combinations(extra_groups.keys(), r):
        combo_oof = list(base_8_oof)
        combo_test_list = list(base_8_test)
        for name in combo:
            oof_e, tpred_e = extra_groups[name]
            combo_oof += [oof_e[k] for k in MODELS]
            combo_test_list += [tpred_e[k] for k in MODELS]

        stack_tr = np.column_stack(combo_oof)
        stack_te = np.column_stack(combo_test_list)

        for alpha in [1.0, 10.0]:
            s_oof = np.zeros(len(y_true))
            s_test = np.zeros(len(X_test_cb))
            for fold, (tr_idx, va_idx) in enumerate(kf.split(stack_tr)):
                meta = Ridge(alpha=alpha)
                meta.fit(stack_tr[tr_idx], y_true[tr_idx])
                s_oof[va_idx] = meta.predict(stack_tr[va_idx])
                s_test += meta.predict(stack_te) / N_SPLITS
            r_rmse = np.sqrt(np.mean((s_oof - y_true) ** 2))

            n_models = len(combo_oof)
            label = '+'.join(combo)
            print(f"  8+{label:20s} ({n_models:2d}모델, α={alpha:5.1f}): {r_rmse:,.0f}")

            if r_rmse < best_combo_rmse:
                best_combo_rmse = r_rmse
                best_combo_name = f"8+{label} (α={alpha})"
                best_combo_test = s_test.copy()

print(f"\n  ★ 최선 조합: {best_combo_name} → OOF {best_combo_rmse:,.0f}")

record_result('L4', 29, 'TGT TRANSFORMS',
              f'타겟 변환 다양성 ({best_combo_name})', best_combo_rmse, 'tested')
