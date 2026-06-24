"""
24 PL2 STACK GU TREND
파이프라인: FE → CB+LGB → Ridge 스태킹 → Pseudo Label (상위 50%) 재학습 → 구별 트렌드 보정
변경점: 모델 간 합의도가 높은 Test 샘플만 pseudo label로 추가 후 재학습
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from catboost import CatBoostRegressor
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

if os.path.exists('/kaggle/input'):
    INPUT_DIR = '/kaggle/input/competitions/scu-5th-ai-competition'
    OUTPUT_DIR = '/kaggle/working'
else:
    _DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    INPUT_DIR = _DIR
    OUTPUT_DIR = _DIR

N_SPLITS = 5
CAT_FEATURES = ['Gu', 'Dong']

# === 데이터 로드 ===
train_orig = pd.read_csv(f'{INPUT_DIR}/seoul_real_estate_train.csv')
test_orig = pd.read_csv(f'{INPUT_DIR}/seoul_real_estate_test.csv')
sample_sub = pd.read_csv(f'{INPUT_DIR}/sample_submission.csv')
y_true_orig = train_orig['Target'].values

# === 구별 트렌드 보정 계수 계산 ===
last_train_ym = train_orig['Transaction_YearMonth'].max()
last_train_seq = (last_train_ym // 100 - 2024) * 12 + last_train_ym % 100

gu_growth = {}
for gu in train_orig['Gu'].unique():
    monthly = train_orig[train_orig['Gu'] == gu].groupby('Transaction_YearMonth')['Target'].mean()
    gu_growth[gu] = monthly.pct_change().dropna().mean()

print("=== 구별 월성장률 ===")
for gu, g in sorted(gu_growth.items(), key=lambda x: -x[1]):
    print(f"  {gu:15s}: {g*100:+.2f}%")

test_seq = (test_orig['Transaction_YearMonth'] // 100 - 2024) * 12 + test_orig['Transaction_YearMonth'] % 100
months_ahead = test_seq.values - last_train_seq
trend_correction = np.array([(1 + gu_growth[gu]) ** m for gu, m in zip(test_orig['Gu'], months_ahead)])

print(f"\n보정 범위: +{(trend_correction.min()-1)*100:.1f}% ~ +{(trend_correction.max()-1)*100:.1f}%")

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

train_p = add_feature_engineering(base_preprocess(train_orig))
test_p = add_feature_engineering(base_preprocess(test_orig))

# ========================================
# Stage 1: 원본 모델로 Test 예측 + 신뢰도 측정
# ========================================
print("\n" + "=" * 50)
print("[Stage 1] 원본 모델로 Test 예측 + 신뢰도 측정")
print("=" * 50)

train_cb, test_cb = encode_categoricals(train_p, test_p, as_category=False)
X_cb = train_cb.drop(columns=['Target'])
y_cb = np.log1p(train_cb['Target'])
X_test_cb = test_cb
cat_indices = [X_cb.columns.get_loc(c) for c in CAT_FEATURES]

train_lgb, test_lgb = encode_categoricals(train_p, test_p, as_category=True)
X_lgb = train_lgb.drop(columns=['Target'])
y_lgb = np.log1p(train_lgb['Target'])
X_test_lgb = test_lgb

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

cb_test_folds = []
lgb_test_folds = []
cb_oof_orig = np.zeros(len(X_cb))
lgb_oof_orig = np.zeros(len(X_lgb))

print("\n  CatBoost + LightGBM 학습 중...")
for fold, (tr_idx, va_idx) in enumerate(kf.split(X_cb)):
    print(f"  Fold {fold+1}/{N_SPLITS}", end=" → ")
    m_cb = CatBoostRegressor(loss_function='RMSE', random_seed=42, verbose=0,
                              iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
    m_cb.fit(X_cb.iloc[tr_idx], y_cb.iloc[tr_idx],
             eval_set=(X_cb.iloc[va_idx], y_cb.iloc[va_idx]), cat_features=cat_indices)
    cb_oof_orig[va_idx] = np.expm1(m_cb.predict(X_cb.iloc[va_idx]))
    cb_test_folds.append(np.expm1(m_cb.predict(X_test_cb)))

    m_lgb = lgb.LGBMRegressor(objective='regression', metric='rmse',
                               verbose=-1, random_state=42, n_estimators=3000, **LGB_PARAMS)
    m_lgb.fit(X_lgb.iloc[tr_idx], y_lgb.iloc[tr_idx],
              eval_set=[(X_lgb.iloc[va_idx], y_lgb.iloc[va_idx])],
              callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])
    lgb_oof_orig[va_idx] = np.expm1(m_lgb.predict(X_lgb.iloc[va_idx]))
    lgb_test_folds.append(np.expm1(m_lgb.predict(X_test_lgb)))

    fold_rmse = np.sqrt(np.mean((cb_oof_orig[va_idx] - y_true_orig[va_idx]) ** 2))
    print(f"CB RMSE: {fold_rmse:,.0f}")

cb_test_mean = np.mean(cb_test_folds, axis=0)
lgb_test_mean = np.mean(lgb_test_folds, axis=0)

# 신뢰도: 모델 간 불일치 + fold 간 분산
model_disagreement = np.abs(cb_test_mean - lgb_test_mean) / ((cb_test_mean + lgb_test_mean) / 2)
fold_cv_cb = np.std(cb_test_folds, axis=0) / np.mean(cb_test_folds, axis=0)
fold_cv_lgb = np.std(lgb_test_folds, axis=0) / np.mean(lgb_test_folds, axis=0)
fold_cv = (fold_cv_cb + fold_cv_lgb) / 2
confidence = 1 - (model_disagreement + fold_cv) / 2

print(f"\n  모델 간 불일치: mean={model_disagreement.mean():.4f}")
print(f"  신뢰도: mean={confidence.mean():.4f}, min={confidence.min():.4f}")

# Pseudo label 생성 (Ridge 스태킹)
stack_train_orig = np.column_stack([cb_oof_orig, lgb_oof_orig])
stack_test_s1 = np.column_stack([cb_test_mean, lgb_test_mean])
meta_orig = Ridge(alpha=1.0)
meta_orig.fit(stack_train_orig, y_true_orig)
pseudo_labels = meta_orig.predict(stack_test_s1)

# 상위 50% 필터링
threshold = np.percentile(confidence, 50)
mask = confidence >= threshold
n_pseudo = mask.sum()
print(f"\n  Pseudo Label 선별: {n_pseudo}건 (신뢰도 상위 50%)")

# ========================================
# Stage 2: Pseudo Label 추가 후 재학습
# ========================================
print("\n" + "=" * 50)
print(f"[Stage 2] Pseudo Label {n_pseudo}건 추가 재학습")
print("=" * 50)

test_selected = test_orig[mask].copy()
test_selected['Target'] = pseudo_labels[mask]
train_aug = pd.concat([train_orig, test_selected], ignore_index=True)
n_orig = len(train_orig)

train_aug_p = add_feature_engineering(base_preprocess(train_aug))
test_p2 = add_feature_engineering(base_preprocess(test_orig))

train_cb2, test_cb2 = encode_categoricals(train_aug_p, test_p2, as_category=False)
X_cb2 = train_cb2.drop(columns=['Target'])
y_cb2 = np.log1p(train_cb2['Target'])
X_test_cb2 = test_cb2
cat_indices2 = [X_cb2.columns.get_loc(c) for c in CAT_FEATURES]

train_lgb2, test_lgb2 = encode_categoricals(train_aug_p, test_p2, as_category=True)
X_lgb2 = train_lgb2.drop(columns=['Target'])
y_lgb2 = np.log1p(train_lgb2['Target'])
X_test_lgb2 = test_lgb2

kf2 = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

cb_oof2 = np.zeros(len(X_cb2))
cb_test2 = np.zeros(len(X_test_cb2))
lgb_oof2 = np.zeros(len(X_lgb2))
lgb_test2 = np.zeros(len(X_test_lgb2))

print("\n[2-1] CatBoost 재학습")
for fold, (tr_idx, va_idx) in enumerate(kf2.split(X_cb2)):
    print(f"  Fold {fold+1}/{N_SPLITS}", end=" → ")
    m = CatBoostRegressor(loss_function='RMSE', random_seed=42, verbose=0,
                          iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
    m.fit(X_cb2.iloc[tr_idx], y_cb2.iloc[tr_idx],
          eval_set=(X_cb2.iloc[va_idx], y_cb2.iloc[va_idx]), cat_features=cat_indices2)
    cb_oof2[va_idx] = np.expm1(m.predict(X_cb2.iloc[va_idx]))
    cb_test2 += np.expm1(m.predict(X_test_cb2)) / N_SPLITS
    fold_rmse = np.sqrt(np.mean((cb_oof2[va_idx] - train_aug['Target'].values[va_idx]) ** 2))
    print(f"RMSE: {fold_rmse:,.0f}")

print("\n[2-2] LightGBM 재학습")
for fold, (tr_idx, va_idx) in enumerate(kf2.split(X_lgb2)):
    print(f"  Fold {fold+1}/{N_SPLITS}", end=" → ")
    m = lgb.LGBMRegressor(objective='regression', metric='rmse',
                          verbose=-1, random_state=42, n_estimators=3000, **LGB_PARAMS)
    m.fit(X_lgb2.iloc[tr_idx], y_lgb2.iloc[tr_idx],
          eval_set=[(X_lgb2.iloc[va_idx], y_lgb2.iloc[va_idx])],
          callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])
    lgb_oof2[va_idx] = np.expm1(m.predict(X_lgb2.iloc[va_idx]))
    lgb_test2 += np.expm1(m.predict(X_test_lgb2)) / N_SPLITS
    fold_rmse = np.sqrt(np.mean((lgb_oof2[va_idx] - train_aug['Target'].values[va_idx]) ** 2))
    print(f"RMSE: {fold_rmse:,.0f}")

# Ridge 스태킹 (원본 Train 부분만)
print("\n[2-3] Ridge 스태킹 + 구별 트렌드 보정")
stack_train2 = np.column_stack([cb_oof2[:n_orig], lgb_oof2[:n_orig]])
stack_test2 = np.column_stack([cb_test2, lgb_test2])

stack_oof2 = np.zeros(n_orig)
stack_test_pred2 = np.zeros(len(stack_test2))

for fold, (tr_idx, va_idx) in enumerate(kf.split(stack_train2)):
    meta = Ridge(alpha=1.0)
    meta.fit(stack_train2[tr_idx], y_true_orig[tr_idx])
    stack_oof2[va_idx] = meta.predict(stack_train2[va_idx])
    stack_test_pred2 += meta.predict(stack_test2) / N_SPLITS

stack_rmse = np.sqrt(np.mean((stack_oof2 - y_true_orig) ** 2))
print(f"\n스태킹 OOF RMSE: {stack_rmse:,.0f}")
print(f"Ridge 계수: CB={meta.coef_[0]:.4f}, LGB={meta.coef_[1]:.4f}, 절편={meta.intercept_:.1f}")

final_pred = stack_test_pred2 * trend_correction

print(f"\n트렌드 보정 전 평균: {stack_test_pred2.mean():,.0f}")
print(f"트렌드 보정 후 평균: {final_pred.mean():,.0f}")

# === 제출 파일 생성 ===
sub = sample_sub.copy()
sub['Target'] = final_pred
sub.to_csv(f'{OUTPUT_DIR}/submission.csv', index=False)
print(f"\n제출 파일 저장 완료: {OUTPUT_DIR}/submission.csv")
