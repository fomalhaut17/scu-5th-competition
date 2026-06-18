# [1]
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
import gdown
warnings.filterwarnings('ignore')

# [2]
train_filed_id = '1Jf2eaIaEA-yfRyYWl_Wk7SfozaaRfaft'
gdown.download(f'https://drive.google.com/uc?id={train_filed_id}', 'seoul_real_estate_train.csv', quiet=False)

test_filed_id = '1WUnwAnuXTSBGu3DpRda-5NgAHys154EK'
gdown.download(f'https://drive.google.com/uc?id={test_filed_id}', 'seoul_real_estate_test.csv', quiet=False)

submission_filed_id = '1v3CsMpnKci14OcqYEADPzAJyUcT3U-RA'
gdown.download(f'https://drive.google.com/uc?id={submission_filed_id}', 'sample_submission.csv', quiet=False)

# [3]
# ---------------------------------------------------------
# 1. 데이터 로드 (Data Loading)
# ---------------------------------------------------------
# 이전 단계에서 생성한 파일들을 불러옵니다.
train = pd.read_csv('seoul_real_estate_train.csv')
test = pd.read_csv('seoul_real_estate_test.csv')
sample_sub = pd.read_csv('sample_submission.csv')

# [4]
# ---------------------------------------------------------
# 2. 전처리 및 피처 엔지니어링 (Preprocessing & Feature Engineering)
# ---------------------------------------------------------
def preprocess_data(df):
    df = df.copy()

    # [Point 1] 결측치 처리: 'Distance_to_Subway'의 빈 값을 중앙값으로 대체
    df['Distance_to_Subway'] = df['Distance_to_Subway'].fillna(df['Distance_to_Subway'].median())

    # [Point 2] 파생 변수 생성: 건축 연도를 활용하여 '노후도(Age)' 변수 생성 (2026년 기준)
    df['Age'] = 2026 - df['Year_Built']

    # [Point 3] 시계열 데이터 분해: 'Transaction_YearMonth'를 연/월로 분리
    df['Year'] = df['Transaction_YearMonth'] // 100
    df['Month'] = df['Transaction_YearMonth'] % 100

    # 모델 학습에 불필요한 식별자 및 원본 컬럼 제거
    drop_cols = ['ID', 'Transaction_YearMonth', 'Year_Built']
    df = df.drop(columns=drop_cols)

    return df

# [5]
# Train과 Test 데이터에 동일한 전처리 함수 적용
train_processed = preprocess_data(train)
test_processed = preprocess_data(test)

# [6]
# ---------------------------------------------------------
# 3. 범주형 변수 인코딩 (Categorical Encoding)
# ---------------------------------------------------------
# 문자열 형태인 'Gu(구)'와 'Dong(동)'을 LightGBM이 이해할 수 있도록 변환
cat_features = ['Gu', 'Dong']
for col in cat_features:
    le = LabelEncoder()
    # Unseen label 에러를 방지하기 위해 Train/Test 범주를 합쳐서 학습
    combined_classes = list(train_processed[col].astype(str)) + list(test_processed[col].astype(str))
    le.fit(combined_classes)

    train_processed[col] = le.transform(train_processed[col].astype(str))
    test_processed[col] = le.transform(test_processed[col].astype(str))

    # LightGBM의 범주형 변수 인식 기능 활성화
    train_processed[col] = train_processed[col].astype('category')
    test_processed[col] = test_processed[col].astype('category')
    
# [7]
# ---------------------------------------------------------
# 4. 데이터 분할 및 타겟 변환 (Data Splitting & Transformation)
# ---------------------------------------------------------
X = train_processed.drop(columns=['Target'])
# 부동산 데이터는 우측 꼬리가 긴(Right-skewed) 형태이므로 로그 변환 적용
y = np.log1p(train_processed['Target'])

X_test = test_processed

# 모델 검증을 위한 Train/Validation 분할 (8:2 비율)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# [8]
# ---------------------------------------------------------
# 5. 모델 학습 (Model Training)
# ---------------------------------------------------------
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.05,
    'n_estimators': 1000,
    'random_state': 42,
    'verbose': -1
}

# [9]
model = lgb.LGBMRegressor(**params)

# [10]
# 과적합 방지를 위한 Early Stopping 적용
callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=100)]

# [11]
# 학습시작
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=callbacks
)

# [12]
# ---------------------------------------------------------
# 6. 예측 및 제출 파일 생성 (Prediction & Submission)
# ---------------------------------------------------------
# 검증 데이터 셋 성능 확인 (원래 스케일로 역변환)
val_preds = np.expm1(model.predict(X_val))
val_true = np.expm1(y_val)
val_rmse = np.sqrt(np.mean((val_preds - val_true)**2))
print(f"\n[Validation RMSE] {val_rmse:,.0f} 만원")

# Test 데이터 예측 및 지수 역변환(expm1) 적용
test_preds = np.expm1(model.predict(X_test))

# 제출 양식에 예측값 채우기
sample_sub['Target'] = test_preds
sample_sub.to_csv('submission.csv', index=False)
print("\n")
print("제출 파일 'submission.csv'이 성공적으로 생성되었습니다.")
print("Kaggle에 제출하세요.")