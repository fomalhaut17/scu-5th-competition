"""
14 STACK RESIDUAL
파이프라인: FE → CatBoost 1차 예측 → 잔차 LightGBM 학습 → 합산 → 구별 트렌드 보정
변경점: 스태킹 대신 Residual Modeling (1차 예측 오차를 2차 모델이 보정)
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
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
train = pd.read_csv(f'{INPUT_DIR}/seoul_real_estate_train.csv')
test = pd.read_csv(f'{INPUT_DIR}/seoul_real_estate_test.csv')
sample_sub = pd.read_csv(f'{INPUT_DIR}/sample_submission.csv')
y_true = train['Target'].values

# === 구별 트렌드 보정 계수 계산 ===
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
# 1단계: CatBoost 1차 예측 (OOF)
# ========================================
train_cb, test_cb = encode_categoricals(train_p, test_p, as_category=False)
X_cb = train_cb.drop(columns=['Target'])
y_cb = np.log1p(train_cb['Target'])
X_test_cb = test_cb
cat_indices = [X_cb.columns.get_loc(c) for c in CAT_FEATURES]

CB_PARAMS = {
    'learning_rate': 0.010118898857677389,
    'depth': 3,
    'l2_leaf_reg': 4.944272225334265,
    'bagging_temperature': 1.4823308606638113,
    'random_strength': 0.4685604025205004,
    'min_data_in_leaf': 46,
}

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

cb_oof = np.zeros(len(X_cb))
cb_test_pred = np.zeros(len(X_test_cb))

print("=" * 50)
print("[1/3] CatBoost 1차 예측")
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
cb_rmse = np.sqrt(np.mean((cb_oof - y_true) ** 2))
print(f"  1차 OOF RMSE: {cb_rmse:,.0f}")

# ========================================
# 2단계: LightGBM 잔차 모델
# ========================================
residual = y_true - cb_oof

print(f"\n  잔차 통계: mean={residual.mean():,.0f}, std={residual.std():,.0f}")
print(f"  잔차 범위: {residual.min():,.0f} ~ {residual.max():,.0f}")

train_lgb, test_lgb = encode_categoricals(train_p, test_p, as_category=True)
X_lgb = train_lgb.drop(columns=['Target'])
X_test_lgb = test_lgb

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

res_oof = np.zeros(len(X_lgb))
res_test_pred = np.zeros(len(X_test_lgb))

print("\n" + "=" * 50)
print("[2/3] LightGBM 잔차 학습")
print("=" * 50)
for fold, (tr_idx, va_idx) in enumerate(kf.split(X_lgb)):
    print(f"  Fold {fold+1}/{N_SPLITS}", end=" → ")
    m_lgb = lgb.LGBMRegressor(objective='regression', metric='rmse',
                          verbose=-1, random_state=42, n_estimators=3000, **LGB_PARAMS)
    m_lgb.fit(X_lgb.iloc[tr_idx], residual[tr_idx],
          eval_set=[(X_lgb.iloc[va_idx], residual[va_idx])],
          callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])
    res_oof[va_idx] = m_lgb.predict(X_lgb.iloc[va_idx])
    res_test_pred += m_lgb.predict(X_test_lgb) / N_SPLITS
    fold_rmse = np.sqrt(np.mean((res_oof[va_idx] - residual[va_idx]) ** 2))
    print(f"잔차 RMSE: {fold_rmse:,.0f}")
res_rmse = np.sqrt(np.mean((res_oof - residual) ** 2))
print(f"  잔차 OOF RMSE: {res_rmse:,.0f}")

# ========================================
# 3단계: 합산 + 트렌드 보정
# ========================================
print("\n" + "=" * 50)
print("[3/3] 합산 + 구별 트렌드 보정")
print("=" * 50)

final_oof = cb_oof + res_oof
final_test = cb_test_pred + res_test_pred

combined_rmse = np.sqrt(np.mean((final_oof - y_true) ** 2))
print(f"합산 OOF RMSE: {combined_rmse:,.0f}")

print(f"\n=== 전략 08 대비 ===")
print(f"  전략 08 OOF: 2,234")
print(f"  전략 14 OOF: {combined_rmse:,.0f} ({combined_rmse - 2234:+,.0f})")

# 구별 트렌드 보정 적용
final_pred = final_test * trend_correction

print(f"\n트렌드 보정 전 평균: {final_test.mean():,.0f}")
print(f"트렌드 보정 후 평균: {final_pred.mean():,.0f}")

# === 제출 파일 생성 ===
sub = sample_sub.copy()
sub['Target'] = final_pred
sub.to_csv(f'{OUTPUT_DIR}/submission.csv', index=False)
print(f"\n제출 파일 저장 완료: {OUTPUT_DIR}/submission.csv")
