"""
63 Hybrid Skeleton
파이프라인: Linear Skeleton (Ridge) -> GBDT Residuals (CB/LGB) -> Weighted Blend
가설: 합성 데이터는 곱셈 구조(Price = Base * Factors)를 가지므로, Log-Linear 모델이 '골격'을 잡고 
       GBDT가 '잔차'를 보정하는 구조가 더 정확하다.
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
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

def encode_categoricals(train_df, test_df):
    train_df, test_df = train_df.copy(), test_df.copy()
    for col in CAT_FEATURES:
        le = LabelEncoder()
        combined = list(train_df[col].astype(str)) + list(test_df[col].astype(str))
        le.fit(combined)
        train_df[col] = le.transform(train_df[col].astype(str))
        test_df[col] = le.transform(test_df[col].astype(str))
    return train_df, test_df

# === Linear Skeleton (Log-Ridge) ===
def train_linear_skeleton(train_df, test_df, kf):
    X = train_df.drop(columns=['Target'])
    X_test = test_df
    
    # 수치형 변수 로그 변환 (곱셈 구조 포착)
    num_cols = X.select_dtypes(include=[np.number]).columns
    X_log = X.copy()
    X_log[num_cols] = np.log1p(X[num_cols])
    X_test_log = X_test.copy()
    X_test_log[num_cols] = np.log1p(X_test[num_cols])
    
    # 표준화
    scaler = StandardScaler()
    X_log_sc = scaler.fit_transform(X_log)
    X_test_log_sc = scaler.transform(X_test_log)
    
    y_log = np.log1p(train_df['Target'])
    
    oof_log = np.zeros(len(X))
    tpred_log = np.zeros(len(X_test))
    
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_log)):
        model = Ridge(alpha=1.0)
        model.fit(X_log_sc[tr_idx], y_log[tr_idx])
        oof_log[va_idx] = model.predict(X_log_sc[va_idx])
        tpred_log += model.predict(X_test_log_sc) / kf.n_splits
        
    return np.expm1(oof_log), np.expm1(tpred_log)

# === GBDT Residual Learner ===
def train_residual_learner(train_df, test_df, skeleton_oof, skeleton_tpred, kf):
    X = train_df.drop(columns=['Target'])
    X_test = test_df
    
    # 타겟을 잔차(Residual)로 설정
    y_res = train_df['Target'].values - skeleton_oof
    
    oof_res = np.zeros(len(X))
    tpred_res = np.zeros(len(X_test))
    
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
        # CatBoost 사용
        m = CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=4, 
                              random_seed=42, verbose=0, loss_function='RMSE')
        m.fit(X.iloc[tr_idx], y_res[tr_idx], 
              eval_set=(X.iloc[va_idx], y_res[va_idx]), 
              early_stopping_rounds=50, cat_features=[X.columns.get_loc(c) for c in CAT_FEATURES])
        
        oof_res[va_idx] = m.predict(X.iloc[va_idx])
        tpred_res += m.predict(X_test) / kf.n_splits
        
    return oof_res, tpred_res

# === 메인 실행 ===
print("Starting Hybrid Skeleton Pipeline...")
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
train_p = add_feature_engineering(base_preprocess(train_orig))
test_p = add_feature_engineering(base_preprocess(test_orig))
train_enc, test_enc = encode_categoricals(train_p, test_p)

# 1. Linear Skeleton
print("\n[1] Training Linear Skeleton (Log-Ridge)...")
skeleton_oof, skeleton_tpred = train_linear_skeleton(train_enc, test_enc, kf)
rmse_skel = np.sqrt(np.mean((skeleton_oof - y_true_orig)**2))
print(f"  Skeleton OOF RMSE: {rmse_skel:,.0f}")

# 2. GBDT Residual Learner
print("\n[2] Training GBDT Residual Learner...")
res_oof, res_tpred = train_residual_learner(train_enc, test_enc, skeleton_oof, skeleton_tpred, kf)
rmse_res = np.sqrt(np.mean((res_oof - y_true_orig - skeleton_oof)**2)) # This is just for the residual itself
print(f"  Residual Learner OOF RMSE: {rmse_res:,.0f}")

# 3. Final Combination
final_oof = skeleton_oof + res_oof
final_tpred = skeleton_tpred + res_tpred

final_rmse = np.sqrt(np.mean((final_oof - y_true_orig)**2))
print(f"\n  ★ Final Hybrid OOF RMSE: {final_rmse:,.0f}")

# 저장
sub = sample_sub.copy()
sub['Target'] = final_tpred
sub.to_csv(f'{OUTPUT_DIR}/submission.csv', index=False)
print("\nSubmission file saved.")
