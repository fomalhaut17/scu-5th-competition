"""
[L4-22] kNN Correction
──────────────────────────
축약명  : KNN+STK
주요 전략: 전략 08 파이프라인 + 예측 후 유사 아파트 잔차 보정
차별점  : 모델 예측 → kNN으로 유사 Train 검색 → 평균 잔차 더함
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from catboost import CatBoostRegressor
import lightgbm as lgb
from utils import load_data, record_result
import warnings
warnings.filterwarnings('ignore')

N_SPLITS = 5
CAT_FEATURES = ['Gu', 'Dong']

train, test, sample_sub = load_data()
y_true = train['Target'].values

# === 구별 트렌드 보정 ===
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

train_p = add_feature_engineering(base_preprocess(train))
test_p = add_feature_engineering(base_preprocess(test))

# ========================================
# 1단계: 베이스 모델 OOF (전략 08 동일)
# ========================================
train_cb, test_cb = encode_categoricals(train_p, test_p, as_category=False)
X_cb = train_cb.drop(columns=['Target'])
y_cb = np.log1p(train_cb['Target'])
X_test_cb = test_cb
cat_indices = [X_cb.columns.get_loc(c) for c in CAT_FEATURES]

train_lgb, test_lgb = encode_categoricals(train_p, test_p, as_category=True)
X_lgb = train_lgb.drop(columns=['Target'])
y_lgb = np.log1p(train_lgb['Target'])
X_test_lgb = test_lgb

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

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

cb_oof = np.zeros(len(X_cb))
cb_test_pred = np.zeros(len(X_test_cb))
lgb_oof = np.zeros(len(X_lgb))
lgb_test_pred = np.zeros(len(X_test_lgb))

print("=" * 50)
print("[1/4] CatBoost 학습")
print("=" * 50)
for fold, (tr_idx, va_idx) in enumerate(kf.split(X_cb)):
    print(f"  Fold {fold+1}/{N_SPLITS}", end=" → ")
    m = CatBoostRegressor(loss_function='RMSE', random_seed=42, verbose=0,
                          iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
    m.fit(X_cb.iloc[tr_idx], y_cb.iloc[tr_idx],
          eval_set=(X_cb.iloc[va_idx], y_cb.iloc[va_idx]), cat_features=cat_indices)
    cb_oof[va_idx] = np.expm1(m.predict(X_cb.iloc[va_idx]))
    cb_test_pred += np.expm1(m.predict(X_test_cb)) / N_SPLITS
    fold_rmse = np.sqrt(np.mean((cb_oof[va_idx] - y_true[va_idx]) ** 2))
    print(f"RMSE: {fold_rmse:,.0f}")
print(f"  OOF RMSE: {np.sqrt(np.mean((cb_oof - y_true) ** 2)):,.0f}")

print("\n" + "=" * 50)
print("[2/4] LightGBM 학습")
print("=" * 50)
for fold, (tr_idx, va_idx) in enumerate(kf.split(X_lgb)):
    print(f"  Fold {fold+1}/{N_SPLITS}", end=" → ")
    m = lgb.LGBMRegressor(objective='regression', metric='rmse',
                          verbose=-1, random_state=42, n_estimators=3000, **LGB_PARAMS)
    m.fit(X_lgb.iloc[tr_idx], y_lgb.iloc[tr_idx],
          eval_set=[(X_lgb.iloc[va_idx], y_lgb.iloc[va_idx])],
          callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])
    lgb_oof[va_idx] = np.expm1(m.predict(X_lgb.iloc[va_idx]))
    lgb_test_pred += np.expm1(m.predict(X_test_lgb)) / N_SPLITS
    fold_rmse = np.sqrt(np.mean((lgb_oof[va_idx] - y_true[va_idx]) ** 2))
    print(f"RMSE: {fold_rmse:,.0f}")
print(f"  OOF RMSE: {np.sqrt(np.mean((lgb_oof - y_true) ** 2)):,.0f}")

# ========================================
# 2단계: Ridge 스태킹 (전략 08 동일)
# ========================================
print("\n" + "=" * 50)
print("[3/4] Ridge 스태킹")
print("=" * 50)

stack_train = np.column_stack([cb_oof, lgb_oof])
stack_test = np.column_stack([cb_test_pred, lgb_test_pred])

stack_oof = np.zeros(len(y_true))
stack_test_pred = np.zeros(len(stack_test))

for fold, (tr_idx, va_idx) in enumerate(kf.split(stack_train)):
    meta = Ridge(alpha=1.0)
    meta.fit(stack_train[tr_idx], y_true[tr_idx])
    stack_oof[va_idx] = meta.predict(stack_train[va_idx])
    stack_test_pred += meta.predict(stack_test) / N_SPLITS

stack_rmse = np.sqrt(np.mean((stack_oof - y_true) ** 2))
print(f"스태킹 OOF RMSE (보정 전): {stack_rmse:,.0f}")

# ========================================
# 3단계: kNN Residual Correction
# ========================================
print("\n" + "=" * 50)
print("[4/4] kNN 잔차 보정")
print("=" * 50)

# kNN에 사용할 피처: 핵심 부동산 속성
KNN_FEATURES = ['Exclusive_Area', 'Floor', 'Age', 'Distance_to_Subway',
                'Brand_Apartment', 'Nearby_Parks']

# Gu를 LabelEncoding해서 kNN 피처에 포함
le_gu = LabelEncoder()
train_gu_encoded = le_gu.fit_transform(train_p['Gu'])
test_gu_encoded = le_gu.transform(test_p['Gu'])

knn_train = train_p[KNN_FEATURES].copy()
knn_train['Gu_enc'] = train_gu_encoded
knn_test = test_p[KNN_FEATURES].copy()
knn_test['Gu_enc'] = test_gu_encoded

scaler = StandardScaler()
knn_train_scaled = scaler.fit_transform(knn_train)
knn_test_scaled = scaler.transform(knn_test)

# OOF 잔차 계산
oof_residuals = y_true - stack_oof

# k값 탐색
print("\n  k값별 OOF kNN 보정 결과:")
best_k, best_rmse = 0, stack_rmse
results = []

for k in [5, 10, 15, 20, 30, 50]:
    corrected_oof = stack_oof.copy()

    for fold, (tr_idx, va_idx) in enumerate(kf.split(knn_train_scaled)):
        nn = NearestNeighbors(n_neighbors=k, metric='euclidean')
        nn.fit(knn_train_scaled[tr_idx])

        distances, indices = nn.kneighbors(knn_train_scaled[va_idx])
        for i, va_i in enumerate(va_idx):
            neighbor_orig_idx = tr_idx[indices[i]]
            neighbor_residuals = oof_residuals[neighbor_orig_idx]
            dist_weights = 1 / (distances[i] + 1e-6)
            correction = np.average(neighbor_residuals, weights=dist_weights)
            corrected_oof[va_i] += correction

    corrected_rmse = np.sqrt(np.mean((corrected_oof - y_true) ** 2))
    results.append((k, corrected_rmse))
    marker = " ★" if corrected_rmse < best_rmse else ""
    print(f"    k={k:3d}: OOF RMSE {corrected_rmse:,.0f}{marker}")
    if corrected_rmse < best_rmse:
        best_k, best_rmse = k, corrected_rmse

print(f"\n  최적 k={best_k}, OOF RMSE: {best_rmse:,.0f}")

if best_k == 0:
    print("  kNN 보정 효과 없음 → 보정 없이 진행")
    final_pred = stack_test_pred * trend_correction
    final_rmse = stack_rmse
else:
    # 최적 k로 Test 보정
    nn_final = NearestNeighbors(n_neighbors=best_k, metric='euclidean')
    nn_final.fit(knn_train_scaled)

    distances, indices = nn_final.kneighbors(knn_test_scaled)
    test_corrections = np.zeros(len(test))
    for i in range(len(test)):
        neighbor_residuals = oof_residuals[indices[i]]
        dist_weights = 1 / (distances[i] + 1e-6)
        test_corrections[i] = np.average(neighbor_residuals, weights=dist_weights)

    corrected_test = stack_test_pred + test_corrections
    final_pred = corrected_test * trend_correction
    final_rmse = best_rmse

    print(f"\n  Test 보정량: mean={test_corrections.mean():+,.0f}, "
          f"min={test_corrections.min():+,.0f}, max={test_corrections.max():+,.0f}")

# === 비교 ===
print("\n" + "=" * 50)
print(f"전략 22 OOF RMSE: {final_rmse:,.0f}")
print(f"전략 08 OOF RMSE: 2,234 / Public: 2,155")
print("=" * 50)

sub = sample_sub.copy()
sub['Target'] = final_pred
sub.to_csv(os.path.join(os.path.dirname(__file__), '..', '..', 'submission_l4_22_knn.csv'), index=False)
print("제출 파일 생성 완료")

record_result('L4', 22, 'KNN+STK', 'kNN 잔차 보정 + CB+LGB Ridge 스태킹 + 구별 트렌드', final_rmse, 'tested')
