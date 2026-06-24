"""
[L4-26] Pseudo Label v2 + Scale Blending 통합
──────────────────────────
축약명  : PL2+SCALE
주요 전략: 4모델(CB/LGB × log/raw) 신뢰도 필터 PL + 4모델 블렌딩 + GTR
차별점  : 전략24(PL2)의 데이터 증강 + 전략25(Scale)의 이종 스케일 앙상블 통합
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

MODELS = ['cb_log', 'cb_raw', 'lgb_log', 'lgb_raw']

def train_4models(train_cb, test_cb, train_lgb, test_lgb, kf, return_fold_preds=False):
    """4개 베이스 모델 학습, OOF + Test 예측 반환"""
    X_cb = train_cb.drop(columns=['Target'])
    X_test_cb = test_cb
    cat_idx = [X_cb.columns.get_loc(c) for c in CAT_FEATURES]

    X_lgb = train_lgb.drop(columns=['Target'])
    X_test_lgb = test_lgb

    y_log = np.log1p(train_cb['Target'])
    y_raw = train_cb['Target'].values

    oof = {k: np.zeros(len(X_cb)) for k in MODELS}
    tpred = {k: np.zeros(len(X_test_cb)) for k in MODELS}
    fold_test_preds = {k: [] for k in MODELS} if return_fold_preds else None

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_cb)):
        # M1: CB log
        m = CatBoostRegressor(loss_function='RMSE', random_seed=42, verbose=0,
                              iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
        m.fit(X_cb.iloc[tr_idx], y_log.iloc[tr_idx],
              eval_set=(X_cb.iloc[va_idx], y_log.iloc[va_idx]), cat_features=cat_idx)
        oof['cb_log'][va_idx] = np.expm1(m.predict(X_cb.iloc[va_idx]))
        fp = np.expm1(m.predict(X_test_cb))
        tpred['cb_log'] += fp / kf.n_splits
        if return_fold_preds:
            fold_test_preds['cb_log'].append(fp)

        # M2: CB raw
        m = CatBoostRegressor(loss_function='RMSE', random_seed=42, verbose=0,
                              iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
        m.fit(X_cb.iloc[tr_idx], y_raw[tr_idx],
              eval_set=(X_cb.iloc[va_idx], y_raw[va_idx]), cat_features=cat_idx)
        oof['cb_raw'][va_idx] = m.predict(X_cb.iloc[va_idx])
        fp = m.predict(X_test_cb)
        tpred['cb_raw'] += fp / kf.n_splits
        if return_fold_preds:
            fold_test_preds['cb_raw'].append(fp)

        # M3: LGB log
        m = lgb.LGBMRegressor(objective='regression', metric='rmse',
                              verbose=-1, random_state=42, n_estimators=3000, **LGB_PARAMS)
        m.fit(X_lgb.iloc[tr_idx], y_log.iloc[tr_idx],
              eval_set=[(X_lgb.iloc[va_idx], y_log.iloc[va_idx])],
              callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])
        oof['lgb_log'][va_idx] = np.expm1(m.predict(X_lgb.iloc[va_idx]))
        fp = np.expm1(m.predict(X_test_lgb))
        tpred['lgb_log'] += fp / kf.n_splits
        if return_fold_preds:
            fold_test_preds['lgb_log'].append(fp)

        # M4: LGB raw
        m = lgb.LGBMRegressor(objective='regression', metric='rmse',
                              verbose=-1, random_state=42, n_estimators=3000, **LGB_PARAMS)
        m.fit(X_lgb.iloc[tr_idx], y_raw[tr_idx],
              eval_set=[(X_lgb.iloc[va_idx], y_raw[va_idx])],
              callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])
        oof['lgb_raw'][va_idx] = m.predict(X_lgb.iloc[va_idx])
        fp = m.predict(X_test_lgb)
        tpred['lgb_raw'] += fp / kf.n_splits
        if return_fold_preds:
            fold_test_preds['lgb_raw'].append(fp)

    return oof, tpred, fold_test_preds


# ========================================
# Stage 1: 원본 데이터로 4모델 학습 → 신뢰도 측정
# ========================================
print("=" * 60)
print("[Stage 1] 원본 데이터 4모델 학습 + Test 신뢰도 측정")
print("=" * 60)

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
train_cb, test_cb, train_lgb, test_lgb = prepare_data(train_orig, test_orig)

oof_s1, tpred_s1, fold_preds_s1 = train_4models(
    train_cb, test_cb, train_lgb, test_lgb, kf, return_fold_preds=True)

# 4모델 평균으로 pseudo label 생성
pseudo_labels = np.mean([tpred_s1[k] for k in MODELS], axis=0)

# 신뢰도: 모델 간 불일치 + fold 간 변동
model_means = np.array([tpred_s1[k] for k in MODELS])
model_disagreement = np.std(model_means, axis=0) / np.mean(model_means, axis=0)

fold_cvs = []
for k in MODELS:
    folds_arr = np.array(fold_preds_s1[k])
    fold_cvs.append(np.std(folds_arr, axis=0) / np.mean(folds_arr, axis=0))
fold_cv = np.mean(fold_cvs, axis=0)

confidence = 1 - (model_disagreement + fold_cv) / 2

print(f"\n  모델 간 불일치 CV: mean={model_disagreement.mean():.4f}")
print(f"  Fold 변동계수:    mean={fold_cv.mean():.4f}")
print(f"  신뢰도:           mean={confidence.mean():.4f}, min={confidence.min():.4f}")

# Stage 1 블렌딩 OOF (참고용)
avg_oof_s1 = np.mean([oof_s1[k] for k in MODELS], axis=0)
rmse_s1 = np.sqrt(np.mean((avg_oof_s1 - y_true_orig) ** 2))
print(f"\n  Stage 1 (PL 없음) 4모델 평균 OOF RMSE: {rmse_s1:,.0f}")

# ========================================
# Stage 2: 신뢰도 필터링 → PL 추가 → 4모델 재학습
# ========================================
for threshold_pct in [50, 30]:
    threshold = np.percentile(confidence, 100 - threshold_pct)
    mask = confidence >= threshold
    n_pseudo = mask.sum()

    print(f"\n{'=' * 60}")
    print(f"[Stage 2] PL2 + Scale Blend (상위 {threshold_pct}% = {n_pseudo}건)")
    print(f"{'=' * 60}")

    test_selected = test_orig[mask].copy()
    test_selected['Target'] = pseudo_labels[mask]
    train_aug = pd.concat([train_orig, test_selected], ignore_index=True)

    kf2 = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    train_cb2, test_cb2, train_lgb2, test_lgb2 = prepare_data(train_aug, test_orig)

    oof_s2, tpred_s2, _ = train_4models(
        train_cb2, test_cb2, train_lgb2, test_lgb2, kf2, return_fold_preds=False)

    # OOF는 원본 Train 영역만 추출
    for k in MODELS:
        oof_s2[k] = oof_s2[k][:n_orig]

    # 방법 A: 단순 평균
    avg_oof = np.mean([oof_s2[k] for k in MODELS], axis=0)
    avg_test = np.mean([tpred_s2[k] for k in MODELS], axis=0)
    rmse_avg = np.sqrt(np.mean((avg_oof - y_true_orig) ** 2))

    # 방법 B: 가중 평균 (원본 60%)
    w_oof = 0.20*oof_s2['cb_log'] + 0.30*oof_s2['cb_raw'] + 0.20*oof_s2['lgb_log'] + 0.30*oof_s2['lgb_raw']
    w_test = 0.20*tpred_s2['cb_log'] + 0.30*tpred_s2['cb_raw'] + 0.20*tpred_s2['lgb_log'] + 0.30*tpred_s2['lgb_raw']
    rmse_w = np.sqrt(np.mean((w_oof - y_true_orig) ** 2))

    # 방법 C: Ridge 스태킹
    stack_train = np.column_stack([oof_s2[k] for k in MODELS])
    stack_test_arr = np.column_stack([tpred_s2[k] for k in MODELS])

    stack_oof = np.zeros(n_orig)
    stack_test_pred = np.zeros(len(test_orig))
    for fold, (tr_idx, va_idx) in enumerate(kf.split(stack_train)):
        meta = Ridge(alpha=1.0)
        meta.fit(stack_train[tr_idx], y_true_orig[tr_idx])
        stack_oof[va_idx] = meta.predict(stack_train[va_idx])
        stack_test_pred += meta.predict(stack_test_arr) / N_SPLITS
    rmse_ridge = np.sqrt(np.mean((stack_oof - y_true_orig) ** 2))

    print(f"  [A] 단순 평균   OOF RMSE: {rmse_avg:,.0f}")
    print(f"  [B] 가중(원본60%) OOF RMSE: {rmse_w:,.0f}")
    print(f"  [C] Ridge 스태킹  OOF RMSE: {rmse_ridge:,.0f}")

    # 최선 선택
    options = {'A': (avg_test, rmse_avg), 'B': (w_test, rmse_w), 'C': (stack_test_pred, rmse_ridge)}
    best_key = min(options, key=lambda k: options[k][1])
    best_test, best_rmse = options[best_key]

    if threshold_pct == 50:
        final_pred = best_test * trend_correction
        final_rmse = best_rmse
        final_method = best_key
        # 모든 방법의 제출 파일도 생성
        for key, (tpred_opt, rmse_opt) in options.items():
            sub = sample_sub.copy()
            sub['Target'] = tpred_opt * trend_correction
            fname = f'submission_l4_26_{key.lower()}.csv'
            sub.to_csv(os.path.join(os.path.dirname(__file__), '..', '..', fname), index=False)

# ========================================
# 최종 결과
# ========================================
print(f"\n{'=' * 60}")
print(f"최종 결과")
print(f"{'=' * 60}")
print(f"  전략 26 OOF RMSE: {final_rmse:,.0f} (방법 {final_method}, PL 50%)")
print(f"  전략 25 (Scale only): 2,226")
print(f"  전략 24 (PL2 only):   2,226")
print(f"  전략 08 (기존 최선):  2,234")
print(f"{'=' * 60}")

sub = sample_sub.copy()
sub['Target'] = final_pred
sub.to_csv(os.path.join(os.path.dirname(__file__), '..', '..', 'submission_l4_26_pl2_scale.csv'), index=False)
print("제출 파일 생성 완료: submission_l4_26_pl2_scale.csv")

record_result('L4', 26, 'PL2+SCALE',
              f'PL2 + Scale Blending (방법{final_method}) + 구별 트렌드', final_rmse, 'tested')
