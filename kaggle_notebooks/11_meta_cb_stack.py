"""
11 META CB STACK
파이프라인: FE → CatBoost + LightGBM → CatBoost 메타 스태킹 (OOF + 원본 피처) → 구별 트렌드 보정
변경점: Ridge 메타 → CatBoost 메타, OOF 예측값 + 원본 피처를 메타 입력으로 사용
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

print("=== 구별 월성장률 ===")
for gu, g in sorted(gu_growth.items(), key=lambda x: -x[1]):
    print(f"  {gu:15s}: {g*100:+.2f}%")

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
# 1단계: 베이스 모델 OOF + 테스트 예측
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

print("\n" + "=" * 50)
print("[1/3] CatBoost 학습")
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
print("[2/3] LightGBM 학습")
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
# 2단계: CatBoost 메타 스태킹 (OOF + 원본 피처)
# ========================================
print("\n" + "=" * 50)
print("[3/3] CatBoost 메타 스태킹 + 구별 트렌드 보정")
print("=" * 50)

# 메타 피처: OOF 예측값 + 원본 피처
META_FEATURES = ['Gu', 'Dong', 'Exclusive_Area', 'Floor', 'Brand_Apartment', 'Age', 'Distance_to_Subway']
META_CAT_FEATURES = ['Gu', 'Dong']

train_meta_base = train.copy()
train_meta_base['Age'] = 2026 - train_meta_base['Year_Built']
train_meta_base['Distance_to_Subway'] = train_meta_base['Distance_to_Subway'].fillna(train_meta_base['Distance_to_Subway'].median())
test_meta_base = test.copy()
test_meta_base['Age'] = 2026 - test_meta_base['Year_Built']
test_meta_base['Distance_to_Subway'] = test_meta_base['Distance_to_Subway'].fillna(train_meta_base['Distance_to_Subway'].median())

# LabelEncode for meta
le_meta = {}
for col in META_CAT_FEATURES:
    le = LabelEncoder()
    combined = list(train_meta_base[col].astype(str)) + list(test_meta_base[col].astype(str))
    le.fit(combined)
    train_meta_base[col] = le.transform(train_meta_base[col].astype(str))
    test_meta_base[col] = le.transform(test_meta_base[col].astype(str))
    le_meta[col] = le

train_meta_df = train_meta_base[META_FEATURES].copy()
train_meta_df['cb_pred'] = cb_oof
train_meta_df['lgb_pred'] = lgb_oof

test_meta_df = test_meta_base[META_FEATURES].copy()
test_meta_df['cb_pred'] = cb_test_pred
test_meta_df['lgb_pred'] = lgb_test_pred

meta_cat_indices = [train_meta_df.columns.get_loc(c) for c in META_CAT_FEATURES]

META_CB_PARAMS = {
    'learning_rate': 0.03,
    'depth': 4,
    'l2_leaf_reg': 10,
    'min_data_in_leaf': 20,
    'random_strength': 0.5,
    'bagging_temperature': 1.0,
}

stack_oof = np.zeros(len(y_true))
stack_test_pred = np.zeros(len(test_meta_df))

for fold, (tr_idx, va_idx) in enumerate(kf.split(train_meta_df)):
    print(f"  Meta Fold {fold+1}/{N_SPLITS}", end=" → ")
    meta = CatBoostRegressor(loss_function='RMSE', random_seed=42, verbose=0,
                             iterations=2000, early_stopping_rounds=100, **META_CB_PARAMS)
    meta.fit(train_meta_df.iloc[tr_idx], y_true[tr_idx],
             eval_set=(train_meta_df.iloc[va_idx], y_true[va_idx]),
             cat_features=meta_cat_indices)
    stack_oof[va_idx] = meta.predict(train_meta_df.iloc[va_idx])
    stack_test_pred += meta.predict(test_meta_df) / N_SPLITS
    fold_rmse = np.sqrt(np.mean((stack_oof[va_idx] - y_true[va_idx]) ** 2))
    print(f"RMSE: {fold_rmse:,.0f}")

stack_rmse = np.sqrt(np.mean((stack_oof - y_true) ** 2))
print(f"  메타 스태킹 OOF RMSE: {stack_rmse:,.0f}")

# 피처 중요도
importance = meta.get_feature_importance()
feat_names = train_meta_df.columns.tolist()
print(f"\n  메타 모델 피처 중요도:")
for name, imp in sorted(zip(feat_names, importance), key=lambda x: -x[1]):
    print(f"    {name:20s}: {imp:.1f}")

# 구별 트렌드 보정 적용
final_pred = stack_test_pred * trend_correction

print(f"\n트렌드 보정 전 평균: {stack_test_pred.mean():,.0f}")
print(f"트렌드 보정 후 평균: {final_pred.mean():,.0f}")

# === 제출 파일 생성 ===
sub = sample_sub.copy()
sub['Target'] = final_pred
sub.to_csv(f'{OUTPUT_DIR}/submission.csv', index=False)
print(f"\n제출 파일 저장 완료: {OUTPUT_DIR}/submission.csv")
