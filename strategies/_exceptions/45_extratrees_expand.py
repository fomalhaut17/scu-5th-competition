"""
[L4-45] ExtraTrees 확장 앙상블
──────────────────────────
축약명  : ET EXPAND
주요 전략: ExtraTrees × log/raw × 기존/평당가 = 4모델 추가 → 기존 8 + ET 4 = 12모델 Ridge
차별점  : 전략 44에서 ExtraTrees만 유일하게 -4 개선 → 같은 패턴(log/raw/평당가)으로 확장
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb
from utils import load_data, record_result
import warnings
warnings.filterwarnings('ignore')

N_SPLITS = 5
CAT_FEATURES = ['Gu', 'Dong']
MODELS = ['cb_log', 'cb_raw', 'lgb_log', 'lgb_raw']
ET_MODELS = ['et_log', 'et_raw']

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

# ========================================
# PL2 (전략 28 동일)
# ========================================
print("=" * 60)
print("[Stage 1] PL2")
print("=" * 60)

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
train_cb, test_cb, train_lgb, test_lgb = prepare_data(train_orig, test_orig)

X_cb = train_cb.drop(columns=['Target'])
X_test_cb = test_cb
cat_idx = [X_cb.columns.get_loc(c) for c in CAT_FEATURES]
X_lgb = train_lgb.drop(columns=['Target'])
X_test_lgb = test_lgb

oof_s1 = {k: np.zeros(n_orig) for k in MODELS}
tpred_s1 = {k: np.zeros(len(test_orig)) for k in MODELS}
fold_preds_s1 = {k: [] for k in MODELS}

for fold, (tr, va) in enumerate(kf.split(X_cb)):
    print(f"  [S1] Fold {fold+1}/5")
    y_log = np.log1p(train_cb['Target'])
    y_raw = train_cb['Target'].values

    m = CatBoostRegressor(loss_function='RMSE', random_seed=42, verbose=0, iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
    m.fit(X_cb.iloc[tr], y_log.iloc[tr], eval_set=(X_cb.iloc[va], y_log.iloc[va]), cat_features=cat_idx)
    oof_s1['cb_log'][va] = np.expm1(m.predict(X_cb.iloc[va]))
    fp = np.expm1(m.predict(X_test_cb)); tpred_s1['cb_log'] += fp/5; fold_preds_s1['cb_log'].append(fp)

    m = CatBoostRegressor(loss_function='RMSE', random_seed=42, verbose=0, iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
    m.fit(X_cb.iloc[tr], y_raw[tr], eval_set=(X_cb.iloc[va], y_raw[va]), cat_features=cat_idx)
    oof_s1['cb_raw'][va] = m.predict(X_cb.iloc[va])
    fp = m.predict(X_test_cb); tpred_s1['cb_raw'] += fp/5; fold_preds_s1['cb_raw'].append(fp)

    m = lgb.LGBMRegressor(objective='regression', metric='rmse', verbose=-1, random_state=42, n_estimators=3000, **LGB_PARAMS)
    m.fit(X_lgb.iloc[tr], y_log.iloc[tr], eval_set=[(X_lgb.iloc[va], y_log.iloc[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_s1['lgb_log'][va] = np.expm1(m.predict(X_lgb.iloc[va]))
    fp = np.expm1(m.predict(X_test_lgb)); tpred_s1['lgb_log'] += fp/5; fold_preds_s1['lgb_log'].append(fp)

    m = lgb.LGBMRegressor(objective='regression', metric='rmse', verbose=-1, random_state=42, n_estimators=3000, **LGB_PARAMS)
    m.fit(X_lgb.iloc[tr], y_raw[tr], eval_set=[(X_lgb.iloc[va], y_raw[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_s1['lgb_raw'][va] = m.predict(X_lgb.iloc[va])
    fp = m.predict(X_test_lgb); tpred_s1['lgb_raw'] += fp/5; fold_preds_s1['lgb_raw'].append(fp)

pseudo_labels = np.mean([tpred_s1[k] for k in MODELS], axis=0)
model_means = np.array([tpred_s1[k] for k in MODELS])
disagree = np.std(model_means, axis=0) / np.mean(model_means, axis=0)
fcvs = [np.std(np.array(fold_preds_s1[k]), axis=0) / np.mean(np.array(fold_preds_s1[k]), axis=0) for k in MODELS]
confidence = 1 - (disagree + np.mean(fcvs, axis=0)) / 2
mask = confidence >= np.percentile(confidence, 50)

test_selected = test_orig[mask].copy()
test_selected['Target'] = pseudo_labels[mask]
train_aug = pd.concat([train_orig, test_selected], ignore_index=True)
area_train_aug = train_aug['Exclusive_Area'].values

# ========================================
# Stage 2: 기존 8모델 (CB/LGB × log/raw × 기존/평당가)
# ========================================
print(f"\n{'=' * 60}")
print("[Stage 2] 8 + ExtraTrees 확장")
print("=" * 60)

kf2 = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
train_cb2, test_cb2, train_lgb2, test_lgb2 = prepare_data(train_aug, test_orig)

X_cb2 = train_cb2.drop(columns=['Target'])
X_test_cb2 = test_cb2
cat_idx2 = [X_cb2.columns.get_loc(c) for c in CAT_FEATURES]
X_lgb2 = train_lgb2.drop(columns=['Target'])
X_test_lgb2 = test_lgb2

y_log_aug = np.log1p(train_aug['Target'])
y_raw_aug = train_aug['Target'].values
y_up_log = np.log1p(train_aug['Target'].values / area_train_aug)
y_up_raw = (train_aug['Target'].values / area_train_aug).astype(float)

all_oof = {}
all_tpred = {}

for fold, (tr, va) in enumerate(kf2.split(X_cb2)):
    print(f"  Fold {fold+1}/5")

    # CB log
    m = CatBoostRegressor(loss_function='RMSE', random_seed=42, verbose=0, iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
    m.fit(X_cb2.iloc[tr], y_log_aug.iloc[tr], eval_set=(X_cb2.iloc[va], y_log_aug.iloc[va]), cat_features=cat_idx2)
    all_oof.setdefault('cb_log', np.zeros(len(X_cb2)))[va] = np.expm1(m.predict(X_cb2.iloc[va]))
    all_tpred['cb_log'] = all_tpred.get('cb_log', np.zeros(len(X_test_cb2))) + np.expm1(m.predict(X_test_cb2)) / 5

    # CB raw
    m = CatBoostRegressor(loss_function='RMSE', random_seed=42, verbose=0, iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
    m.fit(X_cb2.iloc[tr], y_raw_aug[tr], eval_set=(X_cb2.iloc[va], y_raw_aug[va]), cat_features=cat_idx2)
    all_oof.setdefault('cb_raw', np.zeros(len(X_cb2)))[va] = m.predict(X_cb2.iloc[va])
    all_tpred['cb_raw'] = all_tpred.get('cb_raw', np.zeros(len(X_test_cb2))) + m.predict(X_test_cb2) / 5

    # LGB log
    m = lgb.LGBMRegressor(objective='regression', metric='rmse', verbose=-1, random_state=42, n_estimators=3000, **LGB_PARAMS)
    m.fit(X_lgb2.iloc[tr], y_log_aug.iloc[tr], eval_set=[(X_lgb2.iloc[va], y_log_aug.iloc[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
    all_oof.setdefault('lgb_log', np.zeros(len(X_cb2)))[va] = np.expm1(m.predict(X_lgb2.iloc[va]))
    all_tpred['lgb_log'] = all_tpred.get('lgb_log', np.zeros(len(X_test_lgb2))) + np.expm1(m.predict(X_test_lgb2)) / 5

    # LGB raw
    m = lgb.LGBMRegressor(objective='regression', metric='rmse', verbose=-1, random_state=42, n_estimators=3000, **LGB_PARAMS)
    m.fit(X_lgb2.iloc[tr], y_raw_aug[tr], eval_set=[(X_lgb2.iloc[va], y_raw_aug[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
    all_oof.setdefault('lgb_raw', np.zeros(len(X_cb2)))[va] = m.predict(X_lgb2.iloc[va])
    all_tpred['lgb_raw'] = all_tpred.get('lgb_raw', np.zeros(len(X_test_lgb2))) + m.predict(X_test_lgb2) / 5

    # CB 평당가 log
    m = CatBoostRegressor(loss_function='RMSE', random_seed=42, verbose=0, iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
    m.fit(X_cb2.iloc[tr], y_up_log[tr], eval_set=(X_cb2.iloc[va], y_up_log[va]), cat_features=cat_idx2)
    all_oof.setdefault('cb_up_log', np.zeros(len(X_cb2)))[va] = np.expm1(m.predict(X_cb2.iloc[va]))
    all_tpred['cb_up_log'] = all_tpred.get('cb_up_log', np.zeros(len(X_test_cb2))) + np.expm1(m.predict(X_test_cb2)) / 5

    # CB 평당가 raw
    m = CatBoostRegressor(loss_function='RMSE', random_seed=42, verbose=0, iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
    m.fit(X_cb2.iloc[tr], y_up_raw[tr], eval_set=(X_cb2.iloc[va], y_up_raw[va]), cat_features=cat_idx2)
    all_oof.setdefault('cb_up_raw', np.zeros(len(X_cb2)))[va] = m.predict(X_cb2.iloc[va])
    all_tpred['cb_up_raw'] = all_tpred.get('cb_up_raw', np.zeros(len(X_test_cb2))) + m.predict(X_test_cb2) / 5

    # LGB 평당가 log
    m = lgb.LGBMRegressor(objective='regression', metric='rmse', verbose=-1, random_state=42, n_estimators=3000, **LGB_PARAMS)
    m.fit(X_lgb2.iloc[tr], y_up_log[tr], eval_set=[(X_lgb2.iloc[va], y_up_log[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
    all_oof.setdefault('lgb_up_log', np.zeros(len(X_cb2)))[va] = np.expm1(m.predict(X_lgb2.iloc[va]))
    all_tpred['lgb_up_log'] = all_tpred.get('lgb_up_log', np.zeros(len(X_test_lgb2))) + np.expm1(m.predict(X_test_lgb2)) / 5

    # LGB 평당가 raw
    m = lgb.LGBMRegressor(objective='regression', metric='rmse', verbose=-1, random_state=42, n_estimators=3000, **LGB_PARAMS)
    m.fit(X_lgb2.iloc[tr], y_up_raw[tr], eval_set=[(X_lgb2.iloc[va], y_up_raw[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
    all_oof.setdefault('lgb_up_raw', np.zeros(len(X_cb2)))[va] = m.predict(X_lgb2.iloc[va])
    all_tpred['lgb_up_raw'] = all_tpred.get('lgb_up_raw', np.zeros(len(X_test_lgb2))) + m.predict(X_test_lgb2) / 5

    # --- ExtraTrees (스케일링 필요) ---
    scaler = StandardScaler()
    X_et_tr = scaler.fit_transform(X_cb2.iloc[tr])
    X_et_va = scaler.transform(X_cb2.iloc[va])
    X_et_test = scaler.transform(X_test_cb2)

    # ET log
    m = ExtraTreesRegressor(n_estimators=500, max_depth=12, min_samples_leaf=10, random_state=42, n_jobs=-1)
    m.fit(X_et_tr, y_log_aug.iloc[tr])
    all_oof.setdefault('et_log', np.zeros(len(X_cb2)))[va] = np.expm1(m.predict(X_et_va))
    all_tpred['et_log'] = all_tpred.get('et_log', np.zeros(len(X_test_cb2))) + np.expm1(m.predict(X_et_test)) / 5

    # ET raw
    m = ExtraTreesRegressor(n_estimators=500, max_depth=12, min_samples_leaf=10, random_state=42, n_jobs=-1)
    m.fit(X_et_tr, y_raw_aug[tr])
    all_oof.setdefault('et_raw', np.zeros(len(X_cb2)))[va] = m.predict(X_et_va)
    all_tpred['et_raw'] = all_tpred.get('et_raw', np.zeros(len(X_test_cb2))) + m.predict(X_et_test) / 5

    # ET 평당가 log
    m = ExtraTreesRegressor(n_estimators=500, max_depth=12, min_samples_leaf=10, random_state=42, n_jobs=-1)
    m.fit(X_et_tr, y_up_log[tr])
    all_oof.setdefault('et_up_log', np.zeros(len(X_cb2)))[va] = np.expm1(m.predict(X_et_va))
    all_tpred['et_up_log'] = all_tpred.get('et_up_log', np.zeros(len(X_test_cb2))) + np.expm1(m.predict(X_et_test)) / 5

    # ET 평당가 raw
    m = ExtraTreesRegressor(n_estimators=500, max_depth=12, min_samples_leaf=10, random_state=42, n_jobs=-1)
    m.fit(X_et_tr, y_up_raw[tr])
    all_oof.setdefault('et_up_raw', np.zeros(len(X_cb2)))[va] = m.predict(X_et_va)
    all_tpred['et_up_raw'] = all_tpred.get('et_up_raw', np.zeros(len(X_test_cb2))) + m.predict(X_et_test) / 5

# 평당가 → 원래 스케일 환산
for k in ['cb_up_log', 'cb_up_raw', 'lgb_up_log', 'lgb_up_raw', 'et_up_log', 'et_up_raw']:
    all_oof[k] = all_oof[k][:n_orig] * area_train_orig
    all_tpred[k] = all_tpred[k] * area_test

for k in ['cb_log', 'cb_raw', 'lgb_log', 'lgb_raw', 'et_log', 'et_raw']:
    all_oof[k] = all_oof[k][:n_orig]

# ========================================
# 앙상블 비교
# ========================================
print(f"\n{'=' * 60}")
print("앙상블 비교")
print("=" * 60)

model_8 = ['cb_log', 'cb_raw', 'lgb_log', 'lgb_raw', 'cb_up_log', 'cb_up_raw', 'lgb_up_log', 'lgb_up_raw']
model_et2 = ['et_log', 'et_raw']
model_et4 = ['et_log', 'et_raw', 'et_up_log', 'et_up_raw']

def ridge_stack(model_keys, alpha, label):
    stack_tr = np.column_stack([all_oof[k] for k in model_keys])
    stack_te = np.column_stack([all_tpred[k] for k in model_keys])
    s_oof = np.zeros(n_orig)
    s_test = np.zeros(len(test_orig))
    for fold, (tr, va) in enumerate(kf.split(stack_tr)):
        meta = Ridge(alpha=alpha)
        meta.fit(stack_tr[tr], y_true_orig[tr])
        s_oof[va] = meta.predict(stack_tr[va])
        s_test += meta.predict(stack_te) / N_SPLITS
    r = np.sqrt(np.mean((s_oof - y_true_orig)**2))
    return s_oof, s_test, r

# 개별 모델 OOF
print("\n--- 개별 모델 OOF ---")
for k in sorted(all_oof.keys()):
    r = np.sqrt(np.mean((all_oof[k] - y_true_orig)**2))
    print(f"  {k:15s}: OOF {r:,.0f}")

# 앙상블 조합
print("\n--- Ridge 스태킹 조합 ---")
configs = [
    ("8모델 (전략28 기준)", model_8),
    ("8모델 + ET×log/raw", model_8 + model_et2),
    ("8모델 + ET×4", model_8 + model_et4),
    ("ET 4모델만", model_et4),
]

best_rmse = float('inf')
best_test = None
best_name = None

for name, keys in configs:
    for alpha in [1.0, 5.0, 10.0, 50.0]:
        _, s_test, r = ridge_stack(keys, alpha, name)
        diff_str = ""
        if name != configs[0][0]:
            _, _, r_base = ridge_stack(model_8, alpha, "base")
            diff_str = f" ({r - r_base:+,.0f} vs 8모델)"
        print(f"  {name:25s} α={alpha:5.1f}: OOF {r:,.0f}{diff_str}")
        if r < best_rmse:
            best_rmse = r
            best_test = s_test.copy()
            best_name = f"{name} α={alpha}"

print(f"\n  ★ 최선: {best_name} → OOF {best_rmse:,.0f}")

# 제출 파일
final_pred = best_test * trend_correction
sub = sample_sub.copy()
sub['Target'] = final_pred
sub.to_csv(os.path.join(os.path.dirname(__file__), '..', '..', 'submission_l4_45_et_expand.csv'), index=False)
print(f"\n제출 파일 생성: submission_l4_45_et_expand.csv")

record_result('L4', 45, 'ET EXPAND', f'ExtraTrees 확장 ({best_name}) + GTR', best_rmse, 'tested')
