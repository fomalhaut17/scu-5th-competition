"""
[L4-30] Iterative Pseudo Label (반복 PL)
──────────────────────────
축약명  : ITER PL
주요 전략: PL → 학습 → 더 좋은 예측 → PL 재생성 → 재학습 (2~3라운드)
차별점  : 전략 28의 1회 PL을 반복으로 확장, 매 라운드 8모델(기존+평당가)
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


def train_8models_with_pl(train_df, test_df, pseudo_labels, confidence, threshold_pct, round_label):
    """PL 추가 → 기존4 + 평당가4 = 8모델 학습, Ridge 스태킹"""
    threshold = np.percentile(confidence, 100 - threshold_pct)
    mask = confidence >= threshold
    n_pseudo = mask.sum()

    print(f"\n  PL 상위 {threshold_pct}% = {n_pseudo}건 추가")

    test_selected = test_df[mask].copy()
    test_selected['Target'] = pseudo_labels[mask]
    train_aug = pd.concat([train_df, test_selected], ignore_index=True)

    area_train_aug = train_aug['Exclusive_Area'].values

    kf2 = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    train_cb2, test_cb2, train_lgb2, test_lgb2 = prepare_data(train_aug, test_df)

    # 기존 4모델
    oof_base, tpred_base, _ = train_4models(
        train_cb2, test_cb2, train_lgb2, test_lgb2, kf2, label=f"{round_label}-기존")

    for k in MODELS:
        oof_base[k] = oof_base[k][:n_orig]

    # 평당가 4모델
    y_up_log = np.log1p(train_aug['Target'].values / area_train_aug)
    y_up_raw = (train_aug['Target'].values / area_train_aug).astype(float)

    oof_unit, tpred_unit, _ = train_4models(
        train_cb2, test_cb2, train_lgb2, test_lgb2, kf2,
        y_log_override=y_up_log, y_raw_override=y_up_raw, label=f"{round_label}-평당가")

    for k in MODELS:
        oof_unit[k] = oof_unit[k][:n_orig] * area_train_orig
        tpred_unit[k] = tpred_unit[k] * area_test

    # 8모델 Ridge 스태킹
    kf_meta = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    stack_train = np.column_stack([oof_base[k] for k in MODELS] + [oof_unit[k] for k in MODELS])
    stack_test_arr = np.column_stack([tpred_base[k] for k in MODELS] + [tpred_unit[k] for k in MODELS])

    best_rmse = float('inf')
    best_test = None
    best_alpha = None
    for alpha in [1.0, 10.0, 50.0]:
        s_oof = np.zeros(n_orig)
        s_test = np.zeros(len(test_df))
        for fold, (tr_idx, va_idx) in enumerate(kf_meta.split(stack_train)):
            meta = Ridge(alpha=alpha)
            meta.fit(stack_train[tr_idx], y_true_orig[tr_idx])
            s_oof[va_idx] = meta.predict(stack_train[va_idx])
            s_test += meta.predict(stack_test_arr) / N_SPLITS
        rmse = np.sqrt(np.mean((s_oof - y_true_orig) ** 2))
        if rmse < best_rmse:
            best_rmse = rmse
            best_test = s_test.copy()
            best_alpha = alpha

    # 8모델 단순평균도 계산
    avg_oof = np.mean([oof_base[k] for k in MODELS] + [oof_unit[k] for k in MODELS], axis=0)
    avg_test = np.mean([tpred_base[k] for k in MODELS] + [tpred_unit[k] for k in MODELS], axis=0)
    rmse_avg = np.sqrt(np.mean((avg_oof - y_true_orig) ** 2))

    print(f"  8모델 단순평균 OOF: {rmse_avg:,.0f}")
    print(f"  8모델 Ridge(α={best_alpha}) OOF: {best_rmse:,.0f}")

    # 최선의 test 예측을 다음 라운드 PL 소스로 반환
    if rmse_avg < best_rmse:
        return avg_test, rmse_avg, oof_base, tpred_base, oof_unit, tpred_unit
    return best_test, best_rmse, oof_base, tpred_base, oof_unit, tpred_unit


# ========================================
# Round 0: 원본 데이터 4모델 → 신뢰도 측정
# ========================================
print("=" * 60)
print("[Round 0] 원본 4모델 → 신뢰도 + 초기 PL 생성")
print("=" * 60)

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
train_cb, test_cb, train_lgb, test_lgb = prepare_data(train_orig, test_orig)

oof_s0, tpred_s0, fold_preds_s0 = train_4models(
    train_cb, test_cb, train_lgb, test_lgb, kf,
    return_fold_preds=True, label="R0")

pseudo_labels = np.mean([tpred_s0[k] for k in MODELS], axis=0)

model_means = np.array([tpred_s0[k] for k in MODELS])
model_disagreement = np.std(model_means, axis=0) / np.mean(model_means, axis=0)

fold_cvs = []
for k in MODELS:
    folds_arr = np.array(fold_preds_s0[k])
    fold_cvs.append(np.std(folds_arr, axis=0) / np.mean(folds_arr, axis=0))
fold_cv = np.mean(fold_cvs, axis=0)

confidence = 1 - (model_disagreement + fold_cv) / 2

avg_oof_s0 = np.mean([oof_s0[k] for k in MODELS], axis=0)
rmse_s0 = np.sqrt(np.mean((avg_oof_s0 - y_true_orig) ** 2))
print(f"\n  Round 0 OOF RMSE: {rmse_s0:,.0f}")
print(f"  신뢰도: mean={confidence.mean():.4f}")

# ========================================
# Round 1: PL(50%) → 8모델 (= 전략 28과 동일)
# ========================================
print(f"\n{'=' * 60}")
print("[Round 1] PL(50%) → 8모델 (전략 28 재현)")
print("=" * 60)

pl_preds_r1, rmse_r1, oof_base_r1, tpred_base_r1, oof_unit_r1, tpred_unit_r1 = \
    train_8models_with_pl(train_orig, test_orig, pseudo_labels, confidence, 50, "R1")

print(f"\n  ★ Round 1 최선 OOF: {rmse_r1:,.0f}")

# ========================================
# Round 2: Round 1의 더 좋은 예측으로 PL 재생성 → 8모델
# ========================================
print(f"\n{'=' * 60}")
print("[Round 2] Round 1 예측으로 PL 재생성 → 8모델")
print("=" * 60)

# Round 1의 8모델 예측 평균으로 새 PL 생성
new_pseudo_r2 = pl_preds_r1

# 신뢰도도 8모델 기반으로 재계산
all_preds_r1 = np.array(
    [tpred_base_r1[k] for k in MODELS] + [tpred_unit_r1[k] for k in MODELS])
new_disagreement = np.std(all_preds_r1, axis=0) / np.mean(all_preds_r1, axis=0)
new_confidence = 1 - new_disagreement

print(f"  새 신뢰도: mean={new_confidence.mean():.4f}")

pl_preds_r2, rmse_r2, oof_base_r2, tpred_base_r2, oof_unit_r2, tpred_unit_r2 = \
    train_8models_with_pl(train_orig, test_orig, new_pseudo_r2, new_confidence, 50, "R2")

print(f"\n  ★ Round 2 최선 OOF: {rmse_r2:,.0f}")

# ========================================
# Round 3: Round 2 예측으로 한 번 더
# ========================================
print(f"\n{'=' * 60}")
print("[Round 3] Round 2 예측으로 PL 재생성 → 8모델")
print("=" * 60)

new_pseudo_r3 = pl_preds_r2

all_preds_r2 = np.array(
    [tpred_base_r2[k] for k in MODELS] + [tpred_unit_r2[k] for k in MODELS])
new_disagreement_r3 = np.std(all_preds_r2, axis=0) / np.mean(all_preds_r2, axis=0)
new_confidence_r3 = 1 - new_disagreement_r3

print(f"  새 신뢰도: mean={new_confidence_r3.mean():.4f}")

pl_preds_r3, rmse_r3, _, _, _, _ = \
    train_8models_with_pl(train_orig, test_orig, new_pseudo_r3, new_confidence_r3, 50, "R3")

print(f"\n  ★ Round 3 최선 OOF: {rmse_r3:,.0f}")

# ========================================
# 최종 비교
# ========================================
print(f"\n{'=' * 60}")
print("반복 PL 라운드별 비교")
print("=" * 60)
print(f"  Round 0 (PL 없음)  : {rmse_s0:,.0f}")
print(f"  Round 1 (1회 PL)   : {rmse_r1:,.0f}  (전략 28)")
print(f"  Round 2 (2회 PL)   : {rmse_r2:,.0f}")
print(f"  Round 3 (3회 PL)   : {rmse_r3:,.0f}")
print(f"  ─────────────────────────────")
print(f"  전략 28 Public     : 2,096.8")

# 최선 라운드 선택
rounds = {'R1': (pl_preds_r1, rmse_r1), 'R2': (pl_preds_r2, rmse_r2), 'R3': (pl_preds_r3, rmse_r3)}
best_round = min(rounds, key=lambda k: rounds[k][1])
best_pred, best_rmse = rounds[best_round]

final_pred = best_pred * trend_correction

print(f"\n  ★ 최선: {best_round} → OOF {best_rmse:,.0f}")

# === 제출 파일 ===
sub = sample_sub.copy()
sub['Target'] = final_pred
sub.to_csv(os.path.join(os.path.dirname(__file__), '..', '..', 'submission_l4_30_iter_pl.csv'), index=False)
print(f"제출 파일 생성 완료: submission_l4_30_iter_pl.csv")

record_result('L4', 30, 'ITER PL',
              f'Iterative PL {best_round} + 8모델 Ridge + GTR', best_rmse, 'tested')
