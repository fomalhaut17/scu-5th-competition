"""
[L4-23] Domain Adaptation (분포 적응)
──────────────────────────
축약명  : DA+STK
주요 전략: Train vs Test 분류기로 Test와 유사한 Train에 가중치
차별점  : 도메인 지식 아닌 데이터 분포 유사도로 가중치 결정
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
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
# Domain Adaptation: Train vs Test 분류기
# ========================================
print("=" * 50)
print("[0/3] Domain Adaptation 가중치 계산")
print("=" * 50)

# Train/Test 합쳐서 분류기 학습 (Target 제외)
train_feat = train_p.drop(columns=['Target']).copy()
test_feat = test_p.copy()

# LabelEncode for classifier
for col in CAT_FEATURES:
    le = LabelEncoder()
    combined = list(train_feat[col].astype(str)) + list(test_feat[col].astype(str))
    le.fit(combined)
    train_feat[col] = le.transform(train_feat[col].astype(str))
    test_feat[col] = le.transform(test_feat[col].astype(str))

X_domain = pd.concat([train_feat, test_feat], ignore_index=True)
y_domain = np.array([0] * len(train_feat) + [1] * len(test_feat))

# 5-Fold로 Train 샘플의 "Test 유사도" 추정
domain_proba = np.zeros(len(X_domain))
kf_domain = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (tr_idx, va_idx) in enumerate(kf_domain.split(X_domain)):
    clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05,
                              num_leaves=31, verbose=-1, random_state=42)
    clf.fit(X_domain.iloc[tr_idx], y_domain[tr_idx])
    domain_proba[va_idx] = clf.predict_proba(X_domain.iloc[va_idx])[:, 1]

# Train 샘플의 Test 유사도 → 가중치
train_proba = domain_proba[:len(train)]
# p/(1-p) 변환 (importance weight) + clipping
train_proba = np.clip(train_proba, 0.01, 0.99)
importance_weights = train_proba / (1 - train_proba)
# 극단값 제한
importance_weights = np.clip(importance_weights, 0.1, 10.0)
# 정규화: 평균 1
importance_weights = importance_weights / importance_weights.mean()

print(f"  가중치 분포: min={importance_weights.min():.2f}, "
      f"mean={importance_weights.mean():.2f}, max={importance_weights.max():.2f}")
print(f"  가중치 > 1.5: {(importance_weights > 1.5).sum()}건 "
      f"({(importance_weights > 1.5).sum()/len(train)*100:.1f}%)")
print(f"  가중치 < 0.5: {(importance_weights < 0.5).sum()}건 "
      f"({(importance_weights < 0.5).sum()/len(train)*100:.1f}%)")

# ========================================
# 1단계: 베이스 모델 OOF (가중치 적용)
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
print("[1/3] CatBoost 학습 (Domain Adaptation Weight)")
print("=" * 50)
for fold, (tr_idx, va_idx) in enumerate(kf.split(X_cb)):
    print(f"  Fold {fold+1}/{N_SPLITS}", end=" → ")
    m = CatBoostRegressor(loss_function='RMSE', random_seed=42, verbose=0,
                          iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
    m.fit(X_cb.iloc[tr_idx], y_cb.iloc[tr_idx],
          eval_set=(X_cb.iloc[va_idx], y_cb.iloc[va_idx]),
          cat_features=cat_indices, sample_weight=importance_weights[tr_idx])
    cb_oof[va_idx] = np.expm1(m.predict(X_cb.iloc[va_idx]))
    cb_test_pred += np.expm1(m.predict(X_test_cb)) / N_SPLITS
    fold_rmse = np.sqrt(np.mean((cb_oof[va_idx] - y_true[va_idx]) ** 2))
    print(f"RMSE: {fold_rmse:,.0f}")
print(f"  OOF RMSE: {np.sqrt(np.mean((cb_oof - y_true) ** 2)):,.0f}")

print("\n" + "=" * 50)
print("[2/3] LightGBM 학습 (Domain Adaptation Weight)")
print("=" * 50)
for fold, (tr_idx, va_idx) in enumerate(kf.split(X_lgb)):
    print(f"  Fold {fold+1}/{N_SPLITS}", end=" → ")
    m = lgb.LGBMRegressor(objective='regression', metric='rmse',
                          verbose=-1, random_state=42, n_estimators=3000, **LGB_PARAMS)
    m.fit(X_lgb.iloc[tr_idx], y_lgb.iloc[tr_idx],
          eval_set=[(X_lgb.iloc[va_idx], y_lgb.iloc[va_idx])],
          callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
          sample_weight=importance_weights[tr_idx])
    lgb_oof[va_idx] = np.expm1(m.predict(X_lgb.iloc[va_idx]))
    lgb_test_pred += np.expm1(m.predict(X_test_lgb)) / N_SPLITS
    fold_rmse = np.sqrt(np.mean((lgb_oof[va_idx] - y_true[va_idx]) ** 2))
    print(f"RMSE: {fold_rmse:,.0f}")
print(f"  OOF RMSE: {np.sqrt(np.mean((lgb_oof - y_true) ** 2)):,.0f}")

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

final_pred = stack_test_pred * trend_correction

# === 비교 ===
print("\n" + "=" * 50)
print(f"전략 23 OOF RMSE: {stack_rmse:,.0f}")
print(f"전략 08 OOF RMSE: 2,234 / Public: 2,155")
print("=" * 50)

sub = sample_sub.copy()
sub['Target'] = final_pred
sub.to_csv(os.path.join(os.path.dirname(__file__), '..', '..', 'submission_l4_23_da.csv'), index=False)
print("제출 파일 생성 완료")

record_result('L4', 23, 'DA+STK', 'Domain Adaptation Weight + CB+LGB Ridge 스태킹 + 구별 트렌드', stack_rmse, 'tested')
