"""
12 STACK RELATIVE FE
파이프라인: FE + 상대피처 → CatBoost + LightGBM → Ridge 스태킹 → 구별 트렌드 보정
변경점: Error Analysis 기반 상대 피처 추가 (구 내 상대면적, 구×면적 인터랙션 등)
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

def add_relative_features(train_df, test_df):
    """구 내 상대적 위치를 나타내는 피처 추가"""
    target_backup = train_df['Target'].copy()
    train_df = train_df.drop(columns=['Target'])
    combined = pd.concat([train_df, test_df], keys=['train', 'test'])

    # 구별 평균 대비 면적
    gu_area_mean = combined.groupby('Gu')['Exclusive_Area'].transform('mean')
    combined['Area_vs_Gu'] = combined['Exclusive_Area'] / gu_area_mean

    # 구별 평균 대비 층수
    gu_floor_mean = combined.groupby('Gu')['Floor'].transform('mean')
    combined['Floor_vs_Gu'] = combined['Floor'] / gu_floor_mean

    # 구별 평균 대비 건축연도
    gu_age_mean = combined.groupby('Gu')['Age'].transform('mean')
    combined['Age_vs_Gu'] = combined['Age'] / gu_age_mean

    # 동별 평균 대비 면적
    dong_area_mean = combined.groupby('Dong')['Exclusive_Area'].transform('mean')
    combined['Area_vs_Dong'] = combined['Exclusive_Area'] / dong_area_mean

    # 대형 면적 플래그 (Error Analysis: 120㎡+ 과대표)
    combined['Is_Large'] = (combined['Exclusive_Area'] >= 120).astype(int)

    # 면적 × 브랜드 × 구 인터랙션 (수치로)
    combined['Age_x_Area'] = combined['Age'] * combined['Exclusive_Area']

    train_out = combined.loc['train'].reset_index(drop=True)
    test_out = combined.loc['test'].reset_index(drop=True)
    train_out['Target'] = target_backup.values
    return train_out, test_out

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
train_p, test_p = add_relative_features(train_p, test_p)

new_features = ['Area_vs_Gu', 'Floor_vs_Gu', 'Age_vs_Gu', 'Area_vs_Dong', 'Is_Large', 'Age_x_Area']
print("=== 추가된 상대 피처 ===")
for f in new_features:
    print(f"  {f}: mean={train_p[f].mean():.3f}, std={train_p[f].std():.3f}")

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
cb_rmse = np.sqrt(np.mean((cb_oof - y_true) ** 2))
print(f"  OOF RMSE: {cb_rmse:,.0f}")

# CatBoost 피처 중요도
cb_importance = m.get_feature_importance()
cb_feat_names = X_cb.columns.tolist()
print(f"\n  CatBoost 피처 중요도 (상대피처):")
for name, imp in sorted(zip(cb_feat_names, cb_importance), key=lambda x: -x[1]):
    if name in new_features:
        print(f"    {name:20s}: {imp:.1f}")

print("\n" + "=" * 50)
print("[2/3] LightGBM 학습")
print("=" * 50)
for fold, (tr_idx, va_idx) in enumerate(kf.split(X_lgb)):
    print(f"  Fold {fold+1}/{N_SPLITS}", end=" → ")
    m_lgb = lgb.LGBMRegressor(objective='regression', metric='rmse',
                          verbose=-1, random_state=42, n_estimators=3000, **LGB_PARAMS)
    m_lgb.fit(X_lgb.iloc[tr_idx], y_lgb.iloc[tr_idx],
          eval_set=[(X_lgb.iloc[va_idx], y_lgb.iloc[va_idx])],
          callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])
    lgb_oof[va_idx] = np.expm1(m_lgb.predict(X_lgb.iloc[va_idx]))
    lgb_test_pred += np.expm1(m_lgb.predict(X_test_lgb)) / N_SPLITS
    fold_rmse = np.sqrt(np.mean((lgb_oof[va_idx] - y_true[va_idx]) ** 2))
    print(f"RMSE: {fold_rmse:,.0f}")
lgb_rmse = np.sqrt(np.mean((lgb_oof - y_true) ** 2))
print(f"  OOF RMSE: {lgb_rmse:,.0f}")

# LightGBM 피처 중요도
lgb_importance = m_lgb.feature_importances_
lgb_feat_names = X_lgb.columns.tolist()
print(f"\n  LightGBM 피처 중요도 (상대피처):")
for name, imp in sorted(zip(lgb_feat_names, lgb_importance), key=lambda x: -x[1]):
    if name in new_features:
        print(f"    {name:20s}: {imp}")

# ========================================
# 2단계: Ridge 스태킹
# ========================================
print("\n" + "=" * 50)
print("[3/3] Ridge 스태킹 + 구별 트렌드 보정")
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
print(f"스태킹 OOF RMSE: {stack_rmse:,.0f}")
print(f"Ridge 계수: CB={meta.coef_[0]:.4f}, LGB={meta.coef_[1]:.4f}, 절편={meta.intercept_:.1f}")

print(f"\n=== 전략 08 대비 ===")
print(f"  전략 08 OOF: 2,234")
print(f"  전략 12 OOF: {stack_rmse:,.0f} ({stack_rmse - 2234:+,.0f})")

# 구별 트렌드 보정 적용
final_pred = stack_test_pred * trend_correction

print(f"\n트렌드 보정 전 평균: {stack_test_pred.mean():,.0f}")
print(f"트렌드 보정 후 평균: {final_pred.mean():,.0f}")

# === 제출 파일 생성 ===
sub = sample_sub.copy()
sub['Target'] = final_pred
sub.to_csv(f'{OUTPUT_DIR}/submission.csv', index=False)
print(f"\n제출 파일 저장 완료: {OUTPUT_DIR}/submission.csv")
