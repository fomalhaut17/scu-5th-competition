"""
[L4-54] 새로운 피처 관점
──────────────────────────
축약명  : NEW FEAT
주요 전략: 전략47 파이프라인에 새 피처 추가 → 트리 모델 재학습
  1. LOO Dong 평균가 (동별 타겟 인코딩, leave-one-out)
  2. 구 내 면적/층수 백분위 (상대적 위치 정보)
  3. kNN 평균가 (유사 아파트 가격)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
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

# === GTR ===
last_train_ym = train_orig['Transaction_YearMonth'].max()
last_train_seq = (last_train_ym // 100 - 2024) * 12 + last_train_ym % 100
gu_growth = {}
for gu in train_orig['Gu'].unique():
    monthly = train_orig[train_orig['Gu'] == gu].groupby('Transaction_YearMonth')['Target'].mean()
    gu_growth[gu] = monthly.pct_change().dropna().mean()
test_seq = (test_orig['Transaction_YearMonth'] // 100 - 2024) * 12 + test_orig['Transaction_YearMonth'] % 100
months_ahead = test_seq.values - last_train_seq
trend_correction = np.array([(1 + gu_growth[gu]) ** m for gu, m in zip(test_orig['Gu'], months_ahead)])

# ========================================
# 새 피처 생성 함수
# ========================================

def add_loo_dong_mean(train_df, test_df, target):
    """Leave-one-out Dong 평균가"""
    train_out = train_df.copy()
    test_out = test_df.copy()

    dong_sum = train_df.groupby('Dong')[target.name if hasattr(target, 'name') else 'Target'].transform('sum')
    dong_count = train_df.groupby('Dong')[target.name if hasattr(target, 'name') else 'Target'].transform('count')

    # 실제로는 target을 Series로 받으므로 직접 계산
    t = pd.Series(target, index=train_df.index)
    dong_groups = train_df['Dong']
    group_sum = dong_groups.map(t.groupby(dong_groups).sum())
    group_count = dong_groups.map(t.groupby(dong_groups).count())
    train_out['dong_loo_mean'] = (group_sum - t) / (group_count - 1)

    dong_mean = t.groupby(dong_groups).mean()
    test_out['dong_loo_mean'] = test_df['Dong'].map(dong_mean)
    test_out['dong_loo_mean'] = test_out['dong_loo_mean'].fillna(t.mean())

    return train_out, test_out

def add_percentile_features(train_df, test_df):
    """구 내 면적/층수 백분위"""
    train_out = train_df.copy()
    test_out = test_df.copy()
    combined = pd.concat([train_df, test_df], ignore_index=True)

    for col in ['Exclusive_Area', 'Floor']:
        pct_col = f'{col}_pct_in_gu'
        combined[pct_col] = combined.groupby('Gu')[col].rank(pct=True)
        train_out[pct_col] = combined[pct_col].iloc[:len(train_df)].values
        test_out[pct_col] = combined[pct_col].iloc[len(train_df):].values

    return train_out, test_out

def add_knn_price(train_df, test_df, target, k=10):
    """kNN 평균가 (유사 아파트)"""
    feat_cols = ['Exclusive_Area', 'Floor', 'Distance_to_Subway', 'Nearby_Parks', 'Brand_Apartment']

    train_out = train_df.copy()
    test_out = test_df.copy()

    train_feat = train_df[feat_cols].copy()
    train_feat['Distance_to_Subway'] = train_feat['Distance_to_Subway'].fillna(train_feat['Distance_to_Subway'].median())
    test_feat = test_df[feat_cols].copy()
    test_feat['Distance_to_Subway'] = test_feat['Distance_to_Subway'].fillna(train_feat['Distance_to_Subway'].median())

    # Gu를 label encode해서 포함
    le = LabelEncoder()
    combined_gu = list(train_df['Gu']) + list(test_df['Gu'])
    le.fit(combined_gu)
    train_feat['Gu_enc'] = le.transform(train_df['Gu'])
    test_feat['Gu_enc'] = le.transform(test_df['Gu'])

    scaler = StandardScaler()
    train_sc = scaler.fit_transform(train_feat)
    test_sc = scaler.transform(test_feat)

    # LOO kNN for train
    train_knn = np.zeros(len(train_df))
    for i in range(len(train_df)):
        dists = np.sqrt(np.sum((train_sc - train_sc[i])**2, axis=1))
        dists[i] = np.inf
        nn_idx = np.argsort(dists)[:k]
        train_knn[i] = target[nn_idx].mean()
    train_out[f'knn{k}_price'] = train_knn

    # kNN for test
    knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
    knn.fit(train_sc, target)
    test_out[f'knn{k}_price'] = knn.predict(test_sc)

    return train_out, test_out

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
        if col not in train_df.columns:
            continue
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

# ========================================
# 피처 조합별 12모델 파이프라인 테스트
# ========================================
FEATURE_CONFIGS = {
    "기준(전략47)": [],
    "+LOO_Dong": ["dong_loo_mean"],
    "+백분위": ["Exclusive_Area_pct_in_gu", "Floor_pct_in_gu"],
    "+kNN10": ["knn10_price"],
    "+kNN20": ["knn20_price"],
    "+LOO+백분위": ["dong_loo_mean", "Exclusive_Area_pct_in_gu", "Floor_pct_in_gu"],
    "+전부": ["dong_loo_mean", "Exclusive_Area_pct_in_gu", "Floor_pct_in_gu", "knn10_price"],
}

# 새 피처 미리 생성
print("새 피처 생성 중...")
train_feat, test_feat = add_loo_dong_mean(train_orig, test_orig, y_true_orig)
train_feat, test_feat = add_percentile_features(train_feat, test_feat)
train_feat, test_feat = add_knn_price(train_feat, test_feat, y_true_orig, k=10)
train_feat2, test_feat2 = add_knn_price(train_orig, test_orig, y_true_orig, k=20)
train_feat['knn20_price'] = train_feat2['knn20_price']
test_feat['knn20_price'] = test_feat2['knn20_price']
print("  완료.")

# PL2 (기준과 동일하게 1회만)
print(f"\n{'=' * 60}")
print("[Stage 1] PL2 (기준 피처)")
print("=" * 60)

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# Stage 1은 기준 피처로 PL2 생성
train_p = add_feature_engineering(base_preprocess(train_orig))
test_p = add_feature_engineering(base_preprocess(test_orig))
train_cb_s1, test_cb_s1 = encode_categoricals(train_p, test_p, as_category=False)
train_lgb_s1, test_lgb_s1 = encode_categoricals(train_p, test_p, as_category=True)

X_cb_s1 = train_cb_s1.drop(columns=['Target'])
X_test_cb_s1 = test_cb_s1
cat_idx_s1 = [X_cb_s1.columns.get_loc(c) for c in CAT_FEATURES]
X_lgb_s1 = train_lgb_s1.drop(columns=['Target'])
X_test_lgb_s1 = test_lgb_s1

oof_s1 = {k: np.zeros(n_orig) for k in MODELS}
tpred_s1 = {k: np.zeros(len(test_orig)) for k in MODELS}
fold_preds_s1 = {k: [] for k in MODELS}

for fold, (tr, va) in enumerate(kf.split(X_cb_s1)):
    print(f"  [S1] Fold {fold+1}/5")
    y_log = np.log1p(train_cb_s1['Target']); y_raw = train_cb_s1['Target'].values

    m = CatBoostRegressor(loss_function='RMSE', random_seed=42, verbose=0, iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
    m.fit(X_cb_s1.iloc[tr], y_log.iloc[tr], eval_set=(X_cb_s1.iloc[va], y_log.iloc[va]), cat_features=cat_idx_s1)
    oof_s1['cb_log'][va] = np.expm1(m.predict(X_cb_s1.iloc[va]))
    fp = np.expm1(m.predict(X_test_cb_s1)); tpred_s1['cb_log'] += fp/5; fold_preds_s1['cb_log'].append(fp)

    m = CatBoostRegressor(loss_function='RMSE', random_seed=42, verbose=0, iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
    m.fit(X_cb_s1.iloc[tr], y_raw[tr], eval_set=(X_cb_s1.iloc[va], y_raw[va]), cat_features=cat_idx_s1)
    oof_s1['cb_raw'][va] = m.predict(X_cb_s1.iloc[va])
    fp = m.predict(X_test_cb_s1); tpred_s1['cb_raw'] += fp/5; fold_preds_s1['cb_raw'].append(fp)

    m = lgb.LGBMRegressor(objective='regression', metric='rmse', verbose=-1, random_state=42, n_estimators=3000, **LGB_PARAMS)
    m.fit(X_lgb_s1.iloc[tr], y_log.iloc[tr], eval_set=[(X_lgb_s1.iloc[va], y_log.iloc[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_s1['lgb_log'][va] = np.expm1(m.predict(X_lgb_s1.iloc[va]))
    fp = np.expm1(m.predict(X_test_lgb_s1)); tpred_s1['lgb_log'] += fp/5; fold_preds_s1['lgb_log'].append(fp)

    m = lgb.LGBMRegressor(objective='regression', metric='rmse', verbose=-1, random_state=42, n_estimators=3000, **LGB_PARAMS)
    m.fit(X_lgb_s1.iloc[tr], y_raw[tr], eval_set=[(X_lgb_s1.iloc[va], y_raw[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_s1['lgb_raw'][va] = m.predict(X_lgb_s1.iloc[va])
    fp = m.predict(X_test_lgb_s1); tpred_s1['lgb_raw'] += fp/5; fold_preds_s1['lgb_raw'].append(fp)

pseudo_labels = np.mean([tpred_s1[k] for k in MODELS], axis=0)
model_means = np.array([tpred_s1[k] for k in MODELS])
disagree = np.std(model_means, axis=0) / np.mean(model_means, axis=0)
fcvs = [np.std(np.array(fold_preds_s1[k]), axis=0) / np.mean(np.array(fold_preds_s1[k]), axis=0) for k in MODELS]
confidence = 1 - (disagree + np.mean(fcvs, axis=0)) / 2
mask_pl = confidence >= np.percentile(confidence, 50)

# ========================================
# 피처 조합별 테스트
# ========================================
results = {}

for config_name, extra_cols in FEATURE_CONFIGS.items():
    print(f"\n{'=' * 60}")
    print(f"[{config_name}] 12모델 테스트")
    print("=" * 60)

    # PL2 증강
    test_selected = test_orig[mask_pl].copy()
    test_selected['Target'] = pseudo_labels[mask_pl]

    if extra_cols:
        # 새 피처 추가
        tr_with_feat = train_feat.copy()
        te_with_feat = test_feat.copy()
        te_sel_with_feat = te_with_feat[mask_pl].copy()
        te_sel_with_feat['Target'] = pseudo_labels[mask_pl]
        train_aug = pd.concat([tr_with_feat, te_sel_with_feat], ignore_index=True)
    else:
        train_aug = pd.concat([train_orig, test_selected], ignore_index=True)

    area_train_aug = train_aug['Exclusive_Area'].values

    # 전처리
    train_p2 = add_feature_engineering(base_preprocess(train_aug))
    test_p2 = add_feature_engineering(base_preprocess(test_orig if not extra_cols else test_feat))

    # 새 피처 추가
    if extra_cols:
        for col in extra_cols:
            if col in train_aug.columns:
                train_p2[col] = train_aug[col].values
            if col in (test_feat if extra_cols else test_orig).columns:
                test_p2[col] = (test_feat if extra_cols else test_orig)[col].values

    train_cb2, test_cb2 = encode_categoricals(train_p2, test_p2, as_category=False)
    train_lgb2, test_lgb2 = encode_categoricals(train_p2, test_p2, as_category=True)

    X_cb2 = train_cb2.drop(columns=['Target'])
    X_test_cb2 = test_cb2
    cat_idx2 = [X_cb2.columns.get_loc(c) for c in CAT_FEATURES if c in X_cb2.columns]
    X_lgb2 = train_lgb2.drop(columns=['Target'])
    X_test_lgb2 = test_lgb2

    y_log_aug = np.log1p(train_aug['Target'])
    y_raw_aug = train_aug['Target'].values
    y_up_log = np.log1p(train_aug['Target'].values / area_train_aug)
    y_up_raw = (train_aug['Target'].values / area_train_aug).astype(float)

    kf2 = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    all_oof = {}
    all_tpred = {}

    for fold, (tr, va) in enumerate(kf2.split(X_cb2)):
        print(f"  Fold {fold+1}/5")

        m = CatBoostRegressor(loss_function='RMSE', random_seed=42, verbose=0, iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
        m.fit(X_cb2.iloc[tr], y_log_aug.iloc[tr], eval_set=(X_cb2.iloc[va], y_log_aug.iloc[va]), cat_features=cat_idx2)
        all_oof.setdefault('cb_log', np.zeros(len(X_cb2)))[va] = np.expm1(m.predict(X_cb2.iloc[va]))
        all_tpred['cb_log'] = all_tpred.get('cb_log', np.zeros(len(X_test_cb2))) + np.expm1(m.predict(X_test_cb2)) / 5

        m = CatBoostRegressor(loss_function='RMSE', random_seed=42, verbose=0, iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
        m.fit(X_cb2.iloc[tr], y_raw_aug[tr], eval_set=(X_cb2.iloc[va], y_raw_aug[va]), cat_features=cat_idx2)
        all_oof.setdefault('cb_raw', np.zeros(len(X_cb2)))[va] = m.predict(X_cb2.iloc[va])
        all_tpred['cb_raw'] = all_tpred.get('cb_raw', np.zeros(len(X_test_cb2))) + m.predict(X_test_cb2) / 5

        m = lgb.LGBMRegressor(objective='regression', metric='rmse', verbose=-1, random_state=42, n_estimators=3000, **LGB_PARAMS)
        m.fit(X_lgb2.iloc[tr], y_log_aug.iloc[tr], eval_set=[(X_lgb2.iloc[va], y_log_aug.iloc[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
        all_oof.setdefault('lgb_log', np.zeros(len(X_cb2)))[va] = np.expm1(m.predict(X_lgb2.iloc[va]))
        all_tpred['lgb_log'] = all_tpred.get('lgb_log', np.zeros(len(X_test_lgb2))) + np.expm1(m.predict(X_test_lgb2)) / 5

        m = lgb.LGBMRegressor(objective='regression', metric='rmse', verbose=-1, random_state=42, n_estimators=3000, **LGB_PARAMS)
        m.fit(X_lgb2.iloc[tr], y_raw_aug[tr], eval_set=[(X_lgb2.iloc[va], y_raw_aug[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
        all_oof.setdefault('lgb_raw', np.zeros(len(X_cb2)))[va] = m.predict(X_lgb2.iloc[va])
        all_tpred['lgb_raw'] = all_tpred.get('lgb_raw', np.zeros(len(X_test_lgb2))) + m.predict(X_test_lgb2) / 5

        m = CatBoostRegressor(loss_function='RMSE', random_seed=42, verbose=0, iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
        m.fit(X_cb2.iloc[tr], y_up_log[tr], eval_set=(X_cb2.iloc[va], y_up_log[va]), cat_features=cat_idx2)
        all_oof.setdefault('cb_up_log', np.zeros(len(X_cb2)))[va] = np.expm1(m.predict(X_cb2.iloc[va]))
        all_tpred['cb_up_log'] = all_tpred.get('cb_up_log', np.zeros(len(X_test_cb2))) + np.expm1(m.predict(X_test_cb2)) / 5

        m = CatBoostRegressor(loss_function='RMSE', random_seed=42, verbose=0, iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
        m.fit(X_cb2.iloc[tr], y_up_raw[tr], eval_set=(X_cb2.iloc[va], y_up_raw[va]), cat_features=cat_idx2)
        all_oof.setdefault('cb_up_raw', np.zeros(len(X_cb2)))[va] = m.predict(X_cb2.iloc[va])
        all_tpred['cb_up_raw'] = all_tpred.get('cb_up_raw', np.zeros(len(X_test_cb2))) + m.predict(X_test_cb2) / 5

        m = lgb.LGBMRegressor(objective='regression', metric='rmse', verbose=-1, random_state=42, n_estimators=3000, **LGB_PARAMS)
        m.fit(X_lgb2.iloc[tr], y_up_log[tr], eval_set=[(X_lgb2.iloc[va], y_up_log[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
        all_oof.setdefault('lgb_up_log', np.zeros(len(X_cb2)))[va] = np.expm1(m.predict(X_lgb2.iloc[va]))
        all_tpred['lgb_up_log'] = all_tpred.get('lgb_up_log', np.zeros(len(X_test_lgb2))) + np.expm1(m.predict(X_test_lgb2)) / 5

        m = lgb.LGBMRegressor(objective='regression', metric='rmse', verbose=-1, random_state=42, n_estimators=3000, **LGB_PARAMS)
        m.fit(X_lgb2.iloc[tr], y_up_raw[tr], eval_set=[(X_lgb2.iloc[va], y_up_raw[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
        all_oof.setdefault('lgb_up_raw', np.zeros(len(X_cb2)))[va] = m.predict(X_lgb2.iloc[va])
        all_tpred['lgb_up_raw'] = all_tpred.get('lgb_up_raw', np.zeros(len(X_test_lgb2))) + m.predict(X_test_lgb2) / 5

        scaler = StandardScaler()
        X_sc_tr = scaler.fit_transform(X_cb2.iloc[tr])
        X_sc_va = scaler.transform(X_cb2.iloc[va])
        X_sc_te = scaler.transform(X_test_cb2)

        m = ExtraTreesRegressor(n_estimators=500, max_depth=12, min_samples_leaf=10, random_state=42, n_jobs=-1)
        m.fit(X_sc_tr, y_log_aug.iloc[tr])
        all_oof.setdefault('et_log', np.zeros(len(X_cb2)))[va] = np.expm1(m.predict(X_sc_va))
        all_tpred['et_log'] = all_tpred.get('et_log', np.zeros(len(X_test_cb2))) + np.expm1(m.predict(X_sc_te)) / 5

        m = ExtraTreesRegressor(n_estimators=500, max_depth=12, min_samples_leaf=10, random_state=42, n_jobs=-1)
        m.fit(X_sc_tr, y_raw_aug[tr])
        all_oof.setdefault('et_raw', np.zeros(len(X_cb2)))[va] = m.predict(X_sc_va)
        all_tpred['et_raw'] = all_tpred.get('et_raw', np.zeros(len(X_test_cb2))) + m.predict(X_sc_te) / 5

        m = lgb.LGBMRegressor(objective='regression', metric='rmse', verbose=-1, random_state=42, n_estimators=3000, **LGB_ET_PARAMS)
        m.fit(X_lgb2.iloc[tr], y_log_aug.iloc[tr], eval_set=[(X_lgb2.iloc[va], y_log_aug.iloc[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
        all_oof.setdefault('lgbet_log', np.zeros(len(X_cb2)))[va] = np.expm1(m.predict(X_lgb2.iloc[va]))
        all_tpred['lgbet_log'] = all_tpred.get('lgbet_log', np.zeros(len(X_test_lgb2))) + np.expm1(m.predict(X_test_lgb2)) / 5

        m = lgb.LGBMRegressor(objective='regression', metric='rmse', verbose=-1, random_state=42, n_estimators=3000, **LGB_ET_PARAMS)
        m.fit(X_lgb2.iloc[tr], y_raw_aug[tr], eval_set=[(X_lgb2.iloc[va], y_raw_aug[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
        all_oof.setdefault('lgbet_raw', np.zeros(len(X_cb2)))[va] = m.predict(X_lgb2.iloc[va])
        all_tpred['lgbet_raw'] = all_tpred.get('lgbet_raw', np.zeros(len(X_test_lgb2))) + m.predict(X_test_lgb2) / 5

    for k in ['cb_up_log', 'cb_up_raw', 'lgb_up_log', 'lgb_up_raw']:
        all_oof[k] = all_oof[k][:n_orig] * area_train_orig
        all_tpred[k] = all_tpred[k] * area_test
    for k in [kk for kk in all_oof if kk not in ['cb_up_log', 'cb_up_raw', 'lgb_up_log', 'lgb_up_raw']]:
        all_oof[k] = all_oof[k][:n_orig]

    base_12 = ['cb_log', 'cb_raw', 'lgb_log', 'lgb_raw',
               'cb_up_log', 'cb_up_raw', 'lgb_up_log', 'lgb_up_raw',
               'et_log', 'et_raw', 'lgbet_log', 'lgbet_raw']

    st_tr = np.column_stack([all_oof[k] for k in base_12])
    st_te = np.column_stack([all_tpred[k] for k in base_12])

    best_rmse = float('inf')
    best_test = None
    for alpha in [0.1, 0.5, 1.0, 5.0, 10.0, 50.0]:
        s_oof = np.zeros(n_orig)
        s_test = np.zeros(len(test_orig))
        for fold, (tr, va) in enumerate(kf.split(st_tr)):
            meta = Ridge(alpha=alpha)
            meta.fit(st_tr[tr], y_true_orig[tr])
            s_oof[va] = meta.predict(st_tr[va])
            s_test += meta.predict(st_te) / N_SPLITS
        r = np.sqrt(np.mean((s_oof - y_true_orig)**2))
        if r < best_rmse:
            best_rmse = r
            best_test = s_test.copy()

    results[config_name] = best_rmse
    print(f"  → OOF {best_rmse:,.0f}")

# ========================================
# 결과 요약
# ========================================
print(f"\n{'=' * 60}")
print("피처 조합 비교")
print("=" * 60)

baseline = results["기준(전략47)"]
for name, rmse in results.items():
    diff = rmse - baseline
    marker = " ★" if diff < 0 else ""
    print(f"  {name:20s}: OOF {rmse:,.0f} ({diff:+,.0f}){marker}")

record_result('L4', 54, 'NEW FEAT', f'새 피처 테스트 (최선: {min(results, key=results.get)})', min(results.values()), 'tested')
