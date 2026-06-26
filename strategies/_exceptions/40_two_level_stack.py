"""
[L4-40] 2단 스태킹
──────────────────────────
축약명  : 2LV STACK
주요 전략: 전략28의 8모델 OOF → Ridge/ElasticNet/LGB 3개 메타 → 최종 블렌딩
차별점  : 메타 모델에도 다양성을 줌 (현재는 Ridge 1개만 사용)
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
print("=" * 60)
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

print("\n--- 기존 4모델 ---")
oof_base, tpred_base, _ = train_4models(
    train_cb2, test_cb2, train_lgb2, test_lgb2, kf2, label="기존")

for k in MODELS:
    oof_base[k] = oof_base[k][:n_orig]

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
# Level 2: 3개 메타 모델 스태킹
# ========================================
print(f"\n{'=' * 60}")
print("Level 2: 메타 모델 비교")
print("=" * 60)

stack_train = np.column_stack([oof_base[k] for k in MODELS] + [oof_unit[k] for k in MODELS])
stack_test = np.column_stack([tpred_base[k] for k in MODELS] + [tpred_unit[k] for k in MODELS])

def meta_kfold(model_fn, stack_tr, stack_te, y, kf_meta):
    m_oof = np.zeros(len(y))
    m_test = np.zeros(len(stack_te))
    for fold, (tr, va) in enumerate(kf_meta.split(stack_tr)):
        m = model_fn()
        m.fit(stack_tr[tr], y[tr])
        m_oof[va] = m.predict(stack_tr[va])
        m_test += m.predict(stack_te) / kf_meta.n_splits
    rmse = np.sqrt(np.mean((m_oof - y) ** 2))
    return m_oof, m_test, rmse

# 기준: 전략 28 Ridge(α=10)
print("\n--- 전략 28 기준 (Ridge 단독) ---")
ridge_oof, ridge_test, ridge_rmse = meta_kfold(
    lambda: Ridge(alpha=10.0), stack_train, stack_test, y_true_orig, kf)
print(f"  Ridge(α=10): OOF {ridge_rmse:,.0f}")

# Meta 1: Ridge 여러 α
print("\n--- Meta 1: Ridge α sweep ---")
for alpha in [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]:
    _, _, r = meta_kfold(lambda a=alpha: Ridge(alpha=a), stack_train, stack_test, y_true_orig, kf)
    print(f"  Ridge(α={alpha:5.1f}): OOF {r:,.0f}")

# Meta 2: ElasticNet
print("\n--- Meta 2: ElasticNet ---")
en_results = {}
for alpha in [0.1, 1.0, 10.0]:
    for l1 in [0.1, 0.3, 0.5, 0.7, 0.9]:
        oof_en, test_en, r = meta_kfold(
            lambda a=alpha, l=l1: ElasticNet(alpha=a, l1_ratio=l, max_iter=5000),
            stack_train, stack_test, y_true_orig, kf)
        en_results[(alpha, l1)] = (oof_en, test_en, r)
        print(f"  EN(α={alpha:4.1f}, L1={l1:.1f}): OOF {r:,.0f}")

best_en_key = min(en_results, key=lambda k: en_results[k][2])
en_oof, en_test, en_rmse = en_results[best_en_key]
print(f"  ★ 최선 EN: α={best_en_key[0]}, L1={best_en_key[1]} → OOF {en_rmse:,.0f}")

# Meta 3: LightGBM (강한 정규화)
print("\n--- Meta 3: LightGBM 메타 ---")
lgb_meta_configs = [
    {'n_estimators': 100, 'num_leaves': 4, 'max_depth': 2, 'min_child_samples': 50,
     'reg_alpha': 1.0, 'reg_lambda': 1.0, 'learning_rate': 0.05},
    {'n_estimators': 200, 'num_leaves': 8, 'max_depth': 3, 'min_child_samples': 30,
     'reg_alpha': 0.5, 'reg_lambda': 0.5, 'learning_rate': 0.03},
    {'n_estimators': 50, 'num_leaves': 4, 'max_depth': 2, 'min_child_samples': 100,
     'reg_alpha': 5.0, 'reg_lambda': 5.0, 'learning_rate': 0.1},
]

best_lgb_rmse = float('inf')
best_lgb_oof = None
best_lgb_test = None
best_lgb_cfg = None

for i, cfg in enumerate(lgb_meta_configs):
    oof_lgb, test_lgb_m, r = meta_kfold(
        lambda c=cfg: lgb.LGBMRegressor(objective='regression', metric='rmse',
                                         verbose=-1, random_state=42, **c),
        stack_train, stack_test, y_true_orig, kf)
    print(f"  LGB config {i+1}: OOF {r:,.0f} (leaves={cfg['num_leaves']}, est={cfg['n_estimators']})")
    if r < best_lgb_rmse:
        best_lgb_rmse = r
        best_lgb_oof = oof_lgb
        best_lgb_test = test_lgb_m
        best_lgb_cfg = i+1

print(f"  ★ 최선 LGB config {best_lgb_cfg}: OOF {best_lgb_rmse:,.0f}")

# ========================================
# Level 3: 메타 모델 블렌딩
# ========================================
print(f"\n{'=' * 60}")
print("Level 3: 메타 모델 블렌딩")
print("=" * 60)

def calc_rmse(p, y):
    return np.sqrt(np.mean((p - y) ** 2))

# 2모델 조합
combos_2 = [
    ("Ridge+EN", ridge_oof, ridge_test, en_oof, en_test),
    ("Ridge+LGB", ridge_oof, ridge_test, best_lgb_oof, best_lgb_test),
    ("EN+LGB", en_oof, en_test, best_lgb_oof, best_lgb_test),
]

for name, o1, t1, o2, t2 in combos_2:
    for w in [0.3, 0.5, 0.7]:
        blend_oof = w * o1 + (1-w) * o2
        r = calc_rmse(blend_oof, y_true_orig)
        print(f"  {name} ({w:.0%}:{1-w:.0%}): OOF {r:,.0f}")

# 3모델 균등
blend3_oof = (ridge_oof + en_oof + best_lgb_oof) / 3
blend3_test = (ridge_test + en_test + best_lgb_test) / 3
r3 = calc_rmse(blend3_oof, y_true_orig)
print(f"  Ridge+EN+LGB (균등): OOF {r3:,.0f}")

# 3모델 가중 (Ridge 중심)
for rw in [0.5, 0.6, 0.7]:
    ew = (1-rw) / 2
    blend_oof = rw * ridge_oof + ew * en_oof + ew * best_lgb_oof
    blend_test = rw * ridge_test + ew * en_test + ew * best_lgb_test
    r = calc_rmse(blend_oof, y_true_orig)
    print(f"  Ridge+EN+LGB ({rw:.0%}:{ew:.0%}:{ew:.0%}): OOF {r:,.0f}")

# ========================================
# 최종 비교
# ========================================
print(f"\n{'=' * 60}")
print("최종 비교")
print("=" * 60)

all_options = {
    '전략28 Ridge(α=10)': (ridge_test, ridge_rmse),
    f'EN(α={best_en_key[0]},L1={best_en_key[1]})': (en_test, en_rmse),
    f'LGB config{best_lgb_cfg}': (best_lgb_test, best_lgb_rmse),
    'Ridge+EN+LGB 균등': (blend3_test, r3),
}

# 가중 블렌딩들 추가
for rw in [0.5, 0.6, 0.7]:
    ew = (1-rw)/2
    bt = rw*ridge_test + ew*en_test + ew*best_lgb_test
    bo = rw*ridge_oof + ew*en_oof + ew*best_lgb_oof
    br = calc_rmse(bo, y_true_orig)
    all_options[f'3모델 가중({rw:.0%})'] = (bt, br)

best_name = min(all_options, key=lambda k: all_options[k][1])
best_test_pred, best_rmse = all_options[best_name]

for name, (_, rmse) in sorted(all_options.items(), key=lambda x: x[1][1]):
    diff = rmse - ridge_rmse
    marker = " ★" if name == best_name else ""
    print(f"  {name:30s}: OOF {rmse:,.0f} ({diff:+,.0f} vs Ridge){marker}")

final_pred = best_test_pred * trend_correction

print(f"\n  ★ 최선: {best_name} → OOF {best_rmse:,.0f}")
print(f"  (전략 28 Ridge 단독: OOF {ridge_rmse:,.0f})")

# 제출 파일
sub = sample_sub.copy()
sub['Target'] = final_pred
sub.to_csv(os.path.join(os.path.dirname(__file__), '..', '..', 'submission_l4_40_2lv_stack.csv'), index=False)
print(f"\n제출 파일 생성 완료: submission_l4_40_2lv_stack.csv")

record_result('L4', 40, '2LV STACK',
              f'2단 스태킹 ({best_name}) + GTR', best_rmse, 'tested')
