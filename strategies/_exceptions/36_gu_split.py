"""
[L4-36] 구별 독립 파이프라인
──────────────────────────
축약명  : GU SPLIT
주요 전략: 고가/중가/저가 구 그룹별로 완전히 독립적인 8모델+PL2 파이프라인
차별점  : 전체 학습 vs 그룹별 학습 비교
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

# === 구별 통계 ===
print("=== 구별 Train 통계 ===")
for gu in sorted(train_orig['Gu'].unique()):
    mask = train_orig['Gu'] == gu
    n = mask.sum()
    mean = train_orig.loc[mask, 'Target'].mean()
    print(f"  {gu:15s}: {n:4d}건, 평균 {mean:,.0f}")

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


def run_full_pipeline(train_df, test_df, y_true, area_train, area_test, label):
    """전략 28 파이프라인: PL2 + 8모델(기존+평당가) → Ridge/평균"""
    n = len(train_df)

    train_p = add_feature_engineering(base_preprocess(train_df))
    test_p = add_feature_engineering(base_preprocess(test_df))

    train_cb, test_cb = encode_categoricals(train_p, test_p, as_category=False)
    X_cb = train_cb.drop(columns=['Target'])
    X_test_cb = test_cb
    cat_idx = [X_cb.columns.get_loc(c) for c in CAT_FEATURES if c in X_cb.columns]

    train_lgb, test_lgb = encode_categoricals(train_p, test_p, as_category=True)
    X_lgb = train_lgb.drop(columns=['Target'])
    X_test_lgb = test_lgb

    y_log = np.log1p(y_true)
    y_raw = y_true.astype(float)

    n_splits = min(N_SPLITS, n // 10)
    if n_splits < 3:
        n_splits = 3
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    def train_4(X_c, X_tc, X_l, X_tl, ci, yl, yr, tag):
        oof = {k: np.zeros(n) for k in MODELS}
        tpred = {k: np.zeros(len(X_tc)) for k in MODELS}
        fpreds = {k: [] for k in MODELS}

        for fold, (tr_idx, va_idx) in enumerate(kf.split(X_c)):
            m = CatBoostRegressor(loss_function='RMSE', random_seed=42, verbose=0,
                                  iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
            m.fit(X_c.iloc[tr_idx], yl[tr_idx], eval_set=(X_c.iloc[va_idx], yl[va_idx]), cat_features=ci)
            oof['cb_log'][va_idx] = np.expm1(m.predict(X_c.iloc[va_idx]))
            fp = np.expm1(m.predict(X_tc)); tpred['cb_log'] += fp / kf.n_splits; fpreds['cb_log'].append(fp)

            m = CatBoostRegressor(loss_function='RMSE', random_seed=42, verbose=0,
                                  iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
            m.fit(X_c.iloc[tr_idx], yr[tr_idx], eval_set=(X_c.iloc[va_idx], yr[va_idx]), cat_features=ci)
            oof['cb_raw'][va_idx] = m.predict(X_c.iloc[va_idx])
            fp = m.predict(X_tc); tpred['cb_raw'] += fp / kf.n_splits; fpreds['cb_raw'].append(fp)

            m = lgb.LGBMRegressor(objective='regression', metric='rmse', verbose=-1,
                                  random_state=42, n_estimators=3000, **LGB_PARAMS)
            m.fit(X_l.iloc[tr_idx], yl[tr_idx], eval_set=[(X_l.iloc[va_idx], yl[va_idx])],
                  callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])
            oof['lgb_log'][va_idx] = np.expm1(m.predict(X_l.iloc[va_idx]))
            fp = np.expm1(m.predict(X_tl)); tpred['lgb_log'] += fp / kf.n_splits; fpreds['lgb_log'].append(fp)

            m = lgb.LGBMRegressor(objective='regression', metric='rmse', verbose=-1,
                                  random_state=42, n_estimators=3000, **LGB_PARAMS)
            m.fit(X_l.iloc[tr_idx], yr[tr_idx], eval_set=[(X_l.iloc[va_idx], yr[va_idx])],
                  callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])
            oof['lgb_raw'][va_idx] = m.predict(X_l.iloc[va_idx])
            fp = m.predict(X_tl); tpred['lgb_raw'] += fp / kf.n_splits; fpreds['lgb_raw'].append(fp)

        return oof, tpred, fpreds

    # Stage 1: 4모델 → PL 신뢰도
    oof_s1, tpred_s1, fpreds_s1 = train_4(X_cb, X_test_cb, X_lgb, X_test_lgb, cat_idx, y_log, y_raw, "S1")

    pseudo_labels = np.mean([tpred_s1[k] for k in MODELS], axis=0)
    model_means = np.array([tpred_s1[k] for k in MODELS])
    model_dis = np.std(model_means, axis=0) / np.mean(model_means, axis=0)
    fold_cvs = []
    for k in MODELS:
        fa = np.array(fpreds_s1[k])
        fold_cvs.append(np.std(fa, axis=0) / np.mean(fa, axis=0))
    fold_cv = np.mean(fold_cvs, axis=0)
    confidence = 1 - (model_dis + fold_cv) / 2

    # Stage 2: PL(50%) → 8모델
    threshold = np.percentile(confidence, 50)
    mask = confidence >= threshold
    n_pseudo = mask.sum()

    test_selected = test_df[mask].copy()
    test_selected['Target'] = pseudo_labels[mask]
    train_aug = pd.concat([train_df, test_selected], ignore_index=True)
    area_aug = train_aug['Exclusive_Area'].values

    train_p2 = add_feature_engineering(base_preprocess(train_aug))
    test_p2 = add_feature_engineering(base_preprocess(test_df))
    train_cb2, test_cb2 = encode_categoricals(train_p2, test_p2, as_category=False)
    X_cb2 = train_cb2.drop(columns=['Target'])
    X_test_cb2 = test_cb2
    cat_idx2 = [X_cb2.columns.get_loc(c) for c in CAT_FEATURES if c in X_cb2.columns]
    train_lgb2, test_lgb2 = encode_categoricals(train_p2, test_p2, as_category=True)
    X_lgb2 = train_lgb2.drop(columns=['Target'])
    X_test_lgb2 = test_lgb2

    y_log2 = np.log1p(train_aug['Target'].values)
    y_raw2 = train_aug['Target'].values.astype(float)

    n_aug = len(train_aug)
    n_splits2 = min(N_SPLITS, n_aug // 10)
    if n_splits2 < 3:
        n_splits2 = 3
    kf2 = KFold(n_splits=n_splits2, shuffle=True, random_state=42)

    # 기존 4모델
    oof_b = {k: np.zeros(n_aug) for k in MODELS}
    tpred_b = {k: np.zeros(len(test_df)) for k in MODELS}
    for fold, (tr_idx, va_idx) in enumerate(kf2.split(X_cb2)):
        m = CatBoostRegressor(loss_function='RMSE', random_seed=42, verbose=0,
                              iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
        m.fit(X_cb2.iloc[tr_idx], y_log2[tr_idx], eval_set=(X_cb2.iloc[va_idx], y_log2[va_idx]), cat_features=cat_idx2)
        oof_b['cb_log'][va_idx] = np.expm1(m.predict(X_cb2.iloc[va_idx]))
        tpred_b['cb_log'] += np.expm1(m.predict(X_test_cb2)) / kf2.n_splits

        m = CatBoostRegressor(loss_function='RMSE', random_seed=42, verbose=0,
                              iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
        m.fit(X_cb2.iloc[tr_idx], y_raw2[tr_idx], eval_set=(X_cb2.iloc[va_idx], y_raw2[va_idx]), cat_features=cat_idx2)
        oof_b['cb_raw'][va_idx] = m.predict(X_cb2.iloc[va_idx])
        tpred_b['cb_raw'] += m.predict(X_test_cb2) / kf2.n_splits

        m = lgb.LGBMRegressor(objective='regression', metric='rmse', verbose=-1,
                              random_state=42, n_estimators=3000, **LGB_PARAMS)
        m.fit(X_lgb2.iloc[tr_idx], y_log2[tr_idx], eval_set=[(X_lgb2.iloc[va_idx], y_log2[va_idx])],
              callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])
        oof_b['lgb_log'][va_idx] = np.expm1(m.predict(X_lgb2.iloc[va_idx]))
        tpred_b['lgb_log'] += np.expm1(m.predict(X_test_lgb2)) / kf2.n_splits

        m = lgb.LGBMRegressor(objective='regression', metric='rmse', verbose=-1,
                              random_state=42, n_estimators=3000, **LGB_PARAMS)
        m.fit(X_lgb2.iloc[tr_idx], y_raw2[tr_idx], eval_set=[(X_lgb2.iloc[va_idx], y_raw2[va_idx])],
              callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])
        oof_b['lgb_raw'][va_idx] = m.predict(X_lgb2.iloc[va_idx])
        tpred_b['lgb_raw'] += m.predict(X_test_lgb2) / kf2.n_splits

    for k in MODELS:
        oof_b[k] = oof_b[k][:n]

    # 평당가 4모델
    y_up_log = np.log1p(train_aug['Target'].values / area_aug)
    y_up_raw = (train_aug['Target'].values / area_aug).astype(float)

    oof_u = {k: np.zeros(n_aug) for k in MODELS}
    tpred_u = {k: np.zeros(len(test_df)) for k in MODELS}
    for fold, (tr_idx, va_idx) in enumerate(kf2.split(X_cb2)):
        m = CatBoostRegressor(loss_function='RMSE', random_seed=42, verbose=0,
                              iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
        m.fit(X_cb2.iloc[tr_idx], y_up_log[tr_idx], eval_set=(X_cb2.iloc[va_idx], y_up_log[va_idx]), cat_features=cat_idx2)
        oof_u['cb_log'][va_idx] = np.expm1(m.predict(X_cb2.iloc[va_idx]))
        tpred_u['cb_log'] += np.expm1(m.predict(X_test_cb2)) / kf2.n_splits

        m = CatBoostRegressor(loss_function='RMSE', random_seed=42, verbose=0,
                              iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
        m.fit(X_cb2.iloc[tr_idx], y_up_raw[tr_idx], eval_set=(X_cb2.iloc[va_idx], y_up_raw[va_idx]), cat_features=cat_idx2)
        oof_u['cb_raw'][va_idx] = m.predict(X_cb2.iloc[va_idx])
        tpred_u['cb_raw'] += m.predict(X_test_cb2) / kf2.n_splits

        m = lgb.LGBMRegressor(objective='regression', metric='rmse', verbose=-1,
                              random_state=42, n_estimators=3000, **LGB_PARAMS)
        m.fit(X_lgb2.iloc[tr_idx], y_up_log[tr_idx], eval_set=[(X_lgb2.iloc[va_idx], y_up_log[va_idx])],
              callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])
        oof_u['lgb_log'][va_idx] = np.expm1(m.predict(X_lgb2.iloc[va_idx]))
        tpred_u['lgb_log'] += np.expm1(m.predict(X_test_lgb2)) / kf2.n_splits

        m = lgb.LGBMRegressor(objective='regression', metric='rmse', verbose=-1,
                              random_state=42, n_estimators=3000, **LGB_PARAMS)
        m.fit(X_lgb2.iloc[tr_idx], y_up_raw[tr_idx], eval_set=[(X_lgb2.iloc[va_idx], y_up_raw[va_idx])],
              callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])
        oof_u['lgb_raw'][va_idx] = m.predict(X_lgb2.iloc[va_idx])
        tpred_u['lgb_raw'] += m.predict(X_test_lgb2) / kf2.n_splits

    for k in MODELS:
        oof_u[k] = oof_u[k][:n] * area_train
        tpred_u[k] = tpred_u[k] * area_test

    # 8모델 단순평균
    avg_oof = np.mean([oof_b[k] for k in MODELS] + [oof_u[k] for k in MODELS], axis=0)
    avg_test = np.mean([tpred_b[k] for k in MODELS] + [tpred_u[k] for k in MODELS], axis=0)
    rmse = np.sqrt(np.mean((avg_oof - y_true) ** 2))

    print(f"  [{label}] n={n}, PL={n_pseudo}, OOF RMSE: {rmse:,.0f}")

    return avg_oof, avg_test, rmse


# ========================================
# 실험 1: 전체 학습 (전략 28 재현)
# ========================================
print(f"\n{'=' * 60}")
print("[실험 1] 전체 학습 (baseline)")
print("=" * 60)

oof_all, tpred_all, rmse_all = run_full_pipeline(
    train_orig, test_orig, y_true_orig,
    train_orig['Exclusive_Area'].values, test_orig['Exclusive_Area'].values,
    "전체")

# ========================================
# 실험 2: 구 그룹별 독립 학습
# ========================================
GU_GROUPS = {
    '고가': ['Gangnam-gu', 'Seocho-gu', 'Yongsan-gu', 'Seongdong-gu'],
    '중가': ['Songpa-gu', 'Mapo-gu'],
    '저가': ['Eunpyeong-gu', 'Nowon-gu'],
}

print(f"\n{'=' * 60}")
print("[실험 2] 구 그룹별 독립 학습")
print("=" * 60)

oof_split = np.zeros(n_orig)
tpred_split = np.zeros(len(test_orig))

for group_name, gu_list in GU_GROUPS.items():
    print(f"\n  --- {group_name} ({', '.join(gu_list)}) ---")

    train_mask = train_orig['Gu'].isin(gu_list)
    test_mask = test_orig['Gu'].isin(gu_list)

    train_sub = train_orig[train_mask].reset_index(drop=True)
    test_sub = test_orig[test_mask].reset_index(drop=True)

    oof_g, tpred_g, rmse_g = run_full_pipeline(
        train_sub, test_sub,
        train_sub['Target'].values,
        train_sub['Exclusive_Area'].values,
        test_sub['Exclusive_Area'].values,
        group_name)

    oof_split[train_mask.values] = oof_g
    tpred_split[test_mask.values] = tpred_g

rmse_split = np.sqrt(np.mean((oof_split - y_true_orig) ** 2))

# ========================================
# 실험 3: 전체 + 그룹 블렌딩
# ========================================
print(f"\n{'=' * 60}")
print("블렌딩 비교")
print("=" * 60)

print(f"  전체 학습 OOF         : {rmse_all:,.0f}")
print(f"  그룹 독립 OOF         : {rmse_split:,.0f}")

for w in [0.3, 0.5, 0.7]:
    blend = w * oof_all + (1 - w) * oof_split
    rmse_b = np.sqrt(np.mean((blend - y_true_orig) ** 2))
    print(f"  전체{int(w*100)}:그룹{int((1-w)*100)} 블렌드: {rmse_b:,.0f}")

# 최선 선택
options = {
    '전체': (tpred_all, rmse_all),
    '그룹독립': (tpred_split, rmse_split),
}
for w in [0.3, 0.5, 0.7]:
    blend_t = w * tpred_all + (1 - w) * tpred_split
    blend_o = w * oof_all + (1 - w) * oof_split
    rmse_b = np.sqrt(np.mean((blend_o - y_true_orig) ** 2))
    options[f'blend_{int(w*100)}_{int((1-w)*100)}'] = (blend_t, rmse_b)

best_key = min(options, key=lambda k: options[k][1])
best_test, best_rmse = options[best_key]

print(f"\n  ★ 최선: {best_key} → OOF {best_rmse:,.0f}")
print(f"  전략 28: OOF 2,196 / Public 2,096.8")

# === 제출 파일 ===
final_pred = best_test * trend_correction

sub = sample_sub.copy()
sub['Target'] = final_pred
sub.to_csv(os.path.join(os.path.dirname(__file__), '..', '..', 'submission_l4_36_gu_split.csv'), index=False)
print(f"\n제출 파일 생성: submission_l4_36_gu_split.csv")

record_result('L4', 36, 'GU SPLIT',
              f'구별 독립 파이프라인 ({best_key})', best_rmse, 'tested')
