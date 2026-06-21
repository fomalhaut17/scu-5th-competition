import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
import os
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. 데이터 로드
# ---------------------------------------------------------
if not os.path.exists('seoul_real_estate_train.csv'):
    import gdown
    gdown.download('https://drive.google.com/uc?id=1Jf2eaIaEA-yfRyYWl_Wk7SfozaaRfaft', 'seoul_real_estate_train.csv', quiet=False)
    gdown.download('https://drive.google.com/uc?id=1WUnwAnuXTSBGu3DpRda-5NgAHys154EK', 'seoul_real_estate_test.csv', quiet=False)
    gdown.download('https://drive.google.com/uc?id=1v3CsMpnKci14OcqYEADPzAJyUcT3U-RA', 'sample_submission.csv', quiet=False)

train = pd.read_csv('seoul_real_estate_train.csv')
test = pd.read_csv('seoul_real_estate_test.csv')
sample_sub = pd.read_csv('sample_submission.csv')

# ---------------------------------------------------------
# 2. 전처리 및 피처 엔지니어링
# ---------------------------------------------------------
def preprocess_data(df):
    df = df.copy()

    df['Distance_to_Subway'] = df['Distance_to_Subway'].fillna(df['Distance_to_Subway'].median())

    df['Age'] = 2026 - df['Year_Built']
    df['Year'] = df['Transaction_YearMonth'] // 100
    df['Month'] = df['Transaction_YearMonth'] % 100

    df['YearMonth_Seq'] = (df['Year'] - 2024) * 12 + df['Month']
    df['Area_x_Floor'] = df['Exclusive_Area'] * df['Floor']
    df['Floor_per_Area'] = df['Floor'] / df['Exclusive_Area']
    df['Brand_x_Area'] = df['Brand_Apartment'] * df['Exclusive_Area']

    df = df.drop(columns=['ID', 'Transaction_YearMonth', 'Year_Built'])

    return df

train_processed = preprocess_data(train)
test_processed = preprocess_data(test)

# ---------------------------------------------------------
# 3. 범주형 변수 인코딩
# ---------------------------------------------------------
cat_features = ['Gu', 'Dong']
for col in cat_features:
    le = LabelEncoder()
    combined_classes = list(train_processed[col].astype(str)) + list(test_processed[col].astype(str))
    le.fit(combined_classes)

    train_processed[col] = le.transform(train_processed[col].astype(str))
    test_processed[col] = le.transform(test_processed[col].astype(str))

# ---------------------------------------------------------
# 4. 데이터 분할 및 타겟 변환
# ---------------------------------------------------------
X = train_processed.drop(columns=['Target'])
y = np.log1p(train_processed['Target'])

X_test = test_processed

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------------------------------------
# 5. 모델 1: LightGBM
# ---------------------------------------------------------
print("=" * 50)
print("[1/3] LightGBM 학습 중...")
print("=" * 50)

lgb_model = lgb.LGBMRegressor(
    objective='regression',
    metric='rmse',
    learning_rate=0.05,
    n_estimators=1000,
    random_state=42,
    verbose=-1,
)

X_train_lgb = X_train.copy()
X_val_lgb = X_val.copy()
X_test_lgb = X_test.copy()
for col in cat_features:
    X_train_lgb[col] = X_train_lgb[col].astype('category')
    X_val_lgb[col] = X_val_lgb[col].astype('category')
    X_test_lgb[col] = X_test_lgb[col].astype('category')

lgb_model.fit(
    X_train_lgb, y_train,
    eval_set=[(X_val_lgb, y_val)],
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=100)],
)

lgb_val_pred = np.expm1(lgb_model.predict(X_val_lgb))
lgb_test_pred = np.expm1(lgb_model.predict(X_test_lgb))

lgb_rmse = np.sqrt(np.mean((lgb_val_pred - np.expm1(y_val))**2))
print(f"  → LightGBM Validation RMSE: {lgb_rmse:,.0f} 만원\n")

# ---------------------------------------------------------
# 6. 모델 2: XGBoost
# ---------------------------------------------------------
print("=" * 50)
print("[2/3] XGBoost 학습 중...")
print("=" * 50)

xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    eval_metric='rmse',
    learning_rate=0.05,
    n_estimators=1000,
    random_state=42,
    verbosity=0,
    enable_categorical=True,
)

X_train_xgb = X_train.copy()
X_val_xgb = X_val.copy()
X_test_xgb = X_test.copy()
for col in cat_features:
    X_train_xgb[col] = X_train_xgb[col].astype('category')
    X_val_xgb[col] = X_val_xgb[col].astype('category')
    X_test_xgb[col] = X_test_xgb[col].astype('category')

xgb_model.fit(
    X_train_xgb, y_train,
    eval_set=[(X_val_xgb, y_val)],
    verbose=100,
)

xgb_val_pred = np.expm1(xgb_model.predict(X_val_xgb))
xgb_test_pred = np.expm1(xgb_model.predict(X_test_xgb))

xgb_rmse = np.sqrt(np.mean((xgb_val_pred - np.expm1(y_val))**2))
print(f"  → XGBoost Validation RMSE: {xgb_rmse:,.0f} 만원\n")

# ---------------------------------------------------------
# 7. 모델 3: CatBoost
# ---------------------------------------------------------
print("=" * 50)
print("[3/3] CatBoost 학습 중...")
print("=" * 50)

cat_model = CatBoostRegressor(
    loss_function='RMSE',
    learning_rate=0.05,
    iterations=1000,
    random_seed=42,
    verbose=100,
    early_stopping_rounds=50,
)

cat_model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    cat_features=[X_train.columns.get_loc(c) for c in cat_features],
)

cat_val_pred = np.expm1(cat_model.predict(X_val))
cat_test_pred = np.expm1(cat_model.predict(X_test))

cat_rmse = np.sqrt(np.mean((cat_val_pred - np.expm1(y_val))**2))
print(f"  → CatBoost Validation RMSE: {cat_rmse:,.0f} 만원\n")

# ---------------------------------------------------------
# 8. 최적 가중치 탐색
# ---------------------------------------------------------
print("=" * 50)
print("최적 가중치 탐색 중...")
print("=" * 50)

val_true = np.expm1(y_val)
best_rmse = float('inf')
best_weights = (1/3, 1/3, 1/3)

for w_lgb in np.arange(0, 1.05, 0.05):
    for w_xgb in np.arange(0, 1.05 - w_lgb, 0.05):
        w_cat = 1.0 - w_lgb - w_xgb
        if w_cat < 0:
            continue
        pred = w_lgb * lgb_val_pred + w_xgb * xgb_val_pred + w_cat * cat_val_pred
        rmse = np.sqrt(np.mean((pred - val_true)**2))
        if rmse < best_rmse:
            best_rmse = rmse
            best_weights = (w_lgb, w_xgb, w_cat)

w_lgb, w_xgb, w_cat = best_weights

# ---------------------------------------------------------
# 9. 결과 비교
# ---------------------------------------------------------
print("=" * 50)
print("최종 결과 비교")
print("=" * 50)

simple_val_pred = (lgb_val_pred + xgb_val_pred + cat_val_pred) / 3
simple_rmse = np.sqrt(np.mean((simple_val_pred - val_true)**2))

weighted_val_pred = w_lgb * lgb_val_pred + w_xgb * xgb_val_pred + w_cat * cat_val_pred
weighted_rmse = np.sqrt(np.mean((weighted_val_pred - val_true)**2))

print(f"  LightGBM 단독      RMSE: {lgb_rmse:,.0f} 만원")
print(f"  XGBoost  단독      RMSE: {xgb_rmse:,.0f} 만원")
print(f"  CatBoost 단독      RMSE: {cat_rmse:,.0f} 만원")
print(f"  ──────────────────────────")
print(f"  단순 평균 앙상블   RMSE: {simple_rmse:,.0f} 만원")
print(f"  가중 평균 앙상블   RMSE: {weighted_rmse:,.0f} 만원")
print(f"    → 최적 가중치: LGB {w_lgb:.2f} / XGB {w_xgb:.2f} / CAT {w_cat:.2f}")

# ---------------------------------------------------------
# 10. 제출 파일 생성 (CatBoost 단독 + 가중 앙상블)
# ---------------------------------------------------------
sample_sub['Target'] = cat_test_pred
sample_sub.to_csv('submission_catboost.csv', index=False)
print(f"\n제출 파일 'submission_catboost.csv' (CatBoost 단독) 생성 완료")

weighted_test_pred = w_lgb * lgb_test_pred + w_xgb * xgb_test_pred + w_cat * cat_test_pred
sample_sub['Target'] = weighted_test_pred
sample_sub.to_csv('submission_ensemble.csv', index=False)
print(f"제출 파일 'submission_ensemble.csv' (가중 앙상블) 생성 완료")
