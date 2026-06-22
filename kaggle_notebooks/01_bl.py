"""
SCU 5th AI Competition - Kaggle 제출용 통합 스크립트
파이프라인: FE → CatBoost+Optuna + LightGBM+Optuna → 블렌딩 (50:50)
"""
import subprocess, os
subprocess.run(['pip', 'install', 'optuna', '-q'])

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
import optuna
from catboost import CatBoostRegressor
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

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
# 1. CatBoost + Optuna
# ========================================
print("=" * 50)
print("[1/3] CatBoost Optuna 튜닝 (100 trials)")
print("=" * 50)

train_cb, test_cb = encode_categoricals(train_p, test_p, as_category=False)
X_cb = train_cb.drop(columns=['Target'])
y_cb = np.log1p(train_cb['Target'])
X_test_cb = test_cb
cat_indices = [X_cb.columns.get_loc(c) for c in CAT_FEATURES]

def cb_objective(trial):
    params = {
        'loss_function': 'RMSE', 'random_seed': 42, 'verbose': 0,
        'iterations': 2000, 'early_stopping_rounds': 50,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'random_strength': trial.suggest_float('random_strength', 1e-3, 10.0, log=True),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 30),
    }
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    oof = np.zeros(len(X_cb))
    for tr_idx, va_idx in kf.split(X_cb):
        m = CatBoostRegressor(**params)
        m.fit(X_cb.iloc[tr_idx], y_cb.iloc[tr_idx],
              eval_set=(X_cb.iloc[va_idx], y_cb.iloc[va_idx]), cat_features=cat_indices)
        oof[va_idx] = np.expm1(m.predict(X_cb.iloc[va_idx]))
    return np.sqrt(np.mean((oof - np.expm1(y_cb.values)) ** 2))

study_cb = optuna.create_study(direction='minimize')
study_cb.optimize(cb_objective, n_trials=100, show_progress_bar=True)
print(f"\nCatBoost 최적 RMSE: {study_cb.best_value:,.0f}")
best_cb = study_cb.best_params

def cb_model_fn():
    return CatBoostRegressor(
        loss_function='RMSE', random_seed=42, verbose=0,
        iterations=2000, early_stopping_rounds=50, **best_cb)

def cb_fit_fn(model, X_tr, y_tr, X_va, y_va):
    model.fit(X_tr, y_tr, eval_set=(X_va, y_va), cat_features=cat_indices)
    return model

print("\nCatBoost 최종 5-Fold CV:")
cb_oof, cb_test = kfold_train_predict(X_cb, y_cb, X_test_cb, cb_model_fn, cb_fit_fn)

# ========================================
# 2. LightGBM + Optuna
# ========================================
print("\n" + "=" * 50)
print("[2/3] LightGBM Optuna 튜닝 (100 trials)")
print("=" * 50)

train_lgb, test_lgb = encode_categoricals(train_p, test_p, as_category=True)
X_lgb = train_lgb.drop(columns=['Target'])
y_lgb = np.log1p(train_lgb['Target'])
X_test_lgb = test_lgb

def lgb_objective(trial):
    params = {
        'objective': 'regression', 'metric': 'rmse',
        'verbose': -1, 'random_state': 42, 'n_estimators': 2000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15),
        'num_leaves': trial.suggest_int('num_leaves', 16, 128),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
    }
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    oof = np.zeros(len(X_lgb))
    for tr_idx, va_idx in kf.split(X_lgb):
        m = lgb.LGBMRegressor(**params)
        m.fit(X_lgb.iloc[tr_idx], y_lgb.iloc[tr_idx],
              eval_set=[(X_lgb.iloc[va_idx], y_lgb.iloc[va_idx])],
              callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
        oof[va_idx] = np.expm1(m.predict(X_lgb.iloc[va_idx]))
    return np.sqrt(np.mean((oof - np.expm1(y_lgb.values)) ** 2))

study_lgb = optuna.create_study(direction='minimize')
study_lgb.optimize(lgb_objective, n_trials=100, show_progress_bar=True)
print(f"\nLightGBM 최적 RMSE: {study_lgb.best_value:,.0f}")
best_lgb = study_lgb.best_params

def lgb_model_fn():
    return lgb.LGBMRegressor(
        objective='regression', metric='rmse',
        verbose=-1, random_state=42, n_estimators=2000, **best_lgb)

def lgb_fit_fn(model, X_tr, y_tr, X_va, y_va):
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
    return model

print("\nLightGBM 최종 5-Fold CV:")
lgb_oof, lgb_test = kfold_train_predict(X_lgb, y_lgb, X_test_lgb, lgb_model_fn, lgb_fit_fn)

# ========================================
# 3. 블렌딩
# ========================================
print("\n" + "=" * 50)
print("[3/3] 최적 블렌딩")
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

# === 제출 파일 생성 ===
final_pred = best_w * cb_test + (1 - best_w) * lgb_test
sub = sample_sub.copy()
sub['Target'] = final_pred
sub.to_csv(f'{OUTPUT_DIR}/submission.csv', index=False)
print(f"\n제출 파일 저장 완료: {OUTPUT_DIR}/submission.csv")
