"""
05 BL REAL TREND
파이프라인: FE → CatBoost + LightGBM → 블렌딩 → 실제 2026 상승률 기반 트렌드 보정
변경점: 학습 데이터 추정(0.41%/월) 대신 실제 2026년 서울 아파트 상승률(1.9%/월) 사용
구별 비율은 학습 데이터 기준 유지, 월 5% 상한 적용
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

# === 실제 데이터 기반 구별 트렌드 보정 ===
REAL_MONTHLY_GROWTH = 0.019  # 2026년 2월 서울 아파트 실거래 전월대비 +1.9%

last_train_ym = train['Transaction_YearMonth'].max()
last_train_seq = (last_train_ym // 100 - 2024) * 12 + last_train_ym % 100

# 학습 데이터에서 구별 상대 비율 계산
gu_growth_raw = {}
for gu in train['Gu'].unique():
    monthly = train[train['Gu'] == gu].groupby('Transaction_YearMonth')['Target'].mean()
    gu_growth_raw[gu] = monthly.pct_change().dropna().mean()

avg_growth = np.mean(list(gu_growth_raw.values()))
scale = REAL_MONTHLY_GROWTH / avg_growth

gu_growth = {}
for gu, g in gu_growth_raw.items():
    adjusted = g * scale
    gu_growth[gu] = min(adjusted, 0.05)  # 월 5% 상한

print("=== 구별 보정 월성장률 (실제 데이터 반영) ===")
for gu, g in sorted(gu_growth.items(), key=lambda x: -x[1]):
    print(f"  {gu:15s}: {g*100:+.2f}%")

test_seq = (test['Transaction_YearMonth'] // 100 - 2024) * 12 + test['Transaction_YearMonth'] % 100
months_ahead = test_seq.values - last_train_seq
trend_correction = np.array([(1 + gu_growth[gu]) ** m for gu, m in zip(test['Gu'], months_ahead)])

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

train_p = add_feature_engineering(base_preprocess(train))
test_p = add_feature_engineering(base_preprocess(test))

def kfold_train_predict(X, y, X_test, model_fn, fit_fn):
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    oof_pred = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
        print(f"  Fold {fold+1}/{N_SPLITS}", end=" → ")
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        model = model_fn()
        model = fit_fn(model, X_tr, y_tr, X_va, y_va)
        val_pred = np.expm1(model.predict(X_va))
        oof_pred[va_idx] = val_pred
        test_preds += np.expm1(model.predict(X_test)) / N_SPLITS
        fold_rmse = np.sqrt(np.mean((val_pred - np.expm1(y_va.values)) ** 2))
        print(f"RMSE: {fold_rmse:,.0f}")
    overall_rmse = np.sqrt(np.mean((oof_pred - np.expm1(y.values)) ** 2))
    print(f"  OOF RMSE: {overall_rmse:,.0f}")
    return oof_pred, test_preds

# ========================================
# 1. CatBoost
# ========================================
print("\n" + "=" * 50)
print("[1/3] CatBoost 학습")
print("=" * 50)

train_cb, test_cb = encode_categoricals(train_p, test_p, as_category=False)
X_cb = train_cb.drop(columns=['Target'])
y_cb = np.log1p(train_cb['Target'])
X_test_cb = test_cb
cat_indices = [X_cb.columns.get_loc(c) for c in CAT_FEATURES]

CB_PARAMS = {
    'learning_rate': 0.04186725763971498,
    'depth': 4,
    'l2_leaf_reg': 1.460650267194142,
    'bagging_temperature': 0.1569993366109908,
    'random_strength': 0.0017567158792249488,
    'min_data_in_leaf': 14,
}

def cb_model_fn():
    return CatBoostRegressor(
        loss_function='RMSE', random_seed=42, verbose=0,
        iterations=2000, early_stopping_rounds=50, **CB_PARAMS)

def cb_fit_fn(model, X_tr, y_tr, X_va, y_va):
    model.fit(X_tr, y_tr, eval_set=(X_va, y_va), cat_features=cat_indices)
    return model

cb_oof, cb_test = kfold_train_predict(X_cb, y_cb, X_test_cb, cb_model_fn, cb_fit_fn)

# ========================================
# 2. LightGBM
# ========================================
print("\n" + "=" * 50)
print("[2/3] LightGBM 학습")
print("=" * 50)

train_lgb, test_lgb = encode_categoricals(train_p, test_p, as_category=True)
X_lgb = train_lgb.drop(columns=['Target'])
y_lgb = np.log1p(train_lgb['Target'])
X_test_lgb = test_lgb

LGB_PARAMS = {
    'learning_rate': 0.021710596202878886,
    'num_leaves': 53,
    'max_depth': 4,
    'min_child_samples': 26,
    'subsample': 0.610712316805277,
    'colsample_bytree': 0.9096767746644444,
    'reg_alpha': 0.0028576231547823992,
    'reg_lambda': 0.2296812973846381,
}

def lgb_model_fn():
    return lgb.LGBMRegressor(
        objective='regression', metric='rmse',
        verbose=-1, random_state=42, n_estimators=2000, **LGB_PARAMS)

def lgb_fit_fn(model, X_tr, y_tr, X_va, y_va):
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
    return model

lgb_oof, lgb_test = kfold_train_predict(X_lgb, y_lgb, X_test_lgb, lgb_model_fn, lgb_fit_fn)

# ========================================
# 3. 블렌딩 + 실제 데이터 기반 트렌드 보정
# ========================================
print("\n" + "=" * 50)
print("[3/3] 블렌딩 + 실제 데이터 기반 트렌드 보정")
print("=" * 50)

best_rmse = float('inf')
best_w = 0.5
for w in np.arange(0, 1.05, 0.05):
    pred = w * cb_oof + (1 - w) * lgb_oof
    rmse = np.sqrt(np.mean((pred - y_true) ** 2))
    if rmse < best_rmse:
        best_rmse = rmse
        best_w = w

print(f"CatBoost  가중치: {best_w:.0%}")
print(f"LightGBM 가중치: {1 - best_w:.0%}")
print(f"블렌딩 OOF RMSE: {best_rmse:,.0f}")

# 트렌드 보정 적용
final_pred_raw = best_w * cb_test + (1 - best_w) * lgb_test
final_pred = final_pred_raw * trend_correction

print(f"\n트렌드 보정 전 평균: {final_pred_raw.mean():,.0f}")
print(f"트렌드 보정 후 평균: {final_pred.mean():,.0f}")

# === 제출 파일 생성 ===
sub = sample_sub.copy()
sub['Target'] = final_pred
sub.to_csv(f'{OUTPUT_DIR}/submission.csv', index=False)
print(f"\n제출 파일 저장 완료: {OUTPUT_DIR}/submission.csv")
