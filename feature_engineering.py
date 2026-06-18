import pandas as pd
import numpy as np
import lightgbm as lgb
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
# 2. 전처리 및 피처 엔지니어링 (baseline 대비 추가 피처 5개)
# ---------------------------------------------------------
def preprocess_data(df):
    df = df.copy()

    df['Distance_to_Subway'] = df['Distance_to_Subway'].fillna(df['Distance_to_Subway'].median())

    # baseline 피처
    df['Age'] = 2026 - df['Year_Built']
    df['Year'] = df['Transaction_YearMonth'] // 100
    df['Month'] = df['Transaction_YearMonth'] % 100

    # 추가 피처 1: 거래 시점을 연속 숫자로 (시세 트렌드 반영)
    df['YearMonth_Seq'] = (df['Year'] - 2024) * 12 + df['Month']

    # 추가 피처 2: 면적 × 층수 교차항
    df['Area_x_Floor'] = df['Exclusive_Area'] * df['Floor']

    # 추가 피처 3: 층수 / 면적 비율
    df['Floor_per_Area'] = df['Floor'] / df['Exclusive_Area']

    # 추가 피처 4: 브랜드 아파트 × 면적 교차항
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

    train_processed[col] = train_processed[col].astype('category')
    test_processed[col] = test_processed[col].astype('category')

# ---------------------------------------------------------
# 5. 데이터 분할 및 타겟 변환
# ---------------------------------------------------------
X = train_processed.drop(columns=['Target'])
y = np.log1p(train_processed['Target'])

X_test = test_processed

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------------------------------------
# 6. 모델 학습
# ---------------------------------------------------------
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.05,
    'n_estimators': 1000,
    'random_state': 42,
    'verbose': -1
}

model = lgb.LGBMRegressor(**params)

callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=100)]

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=callbacks
)

# ---------------------------------------------------------
# 7. 예측 및 제출 파일 생성
# ---------------------------------------------------------
val_preds = np.expm1(model.predict(X_val))
val_true = np.expm1(y_val)
val_rmse = np.sqrt(np.mean((val_preds - val_true)**2))
print(f"\n[Validation RMSE] {val_rmse:,.0f} 만원")

test_preds = np.expm1(model.predict(X_test))

sample_sub['Target'] = test_preds
sample_sub.to_csv('submission_fe.csv', index=False)
print(model.feature_importances_)
print("제출 파일 'submission_fe.csv'이 성공적으로 생성되었습니다.")
