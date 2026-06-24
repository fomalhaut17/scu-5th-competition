"""
19 PSEUDO LABEL
파이프라인: 전략 08로 Test 예측 → 안정적 구의 예측을 Train에 편입 → 재학습 → 트렌드 보정
변경점: Pseudo Labeling으로 2026년 가격 패턴을 간접 학습
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

# Pseudo Label 대상: OOF RMSE가 낮았던 안정적인 구
STABLE_GUS = ['Eunpyeong-gu', 'Songpa-gu', 'Mapo-gu']

# === 데이터 로드 ===
train = pd.read_csv(f'{INPUT_DIR}/seoul_real_estate_train.csv')
test = pd.read_csv(f'{INPUT_DIR}/seoul_real_estate_test.csv')
sample_sub = pd.read_csv(f'{INPUT_DIR}/sample_submission.csv')
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

CB_PARAMS = {
    'learning_rate': 0.010118898857677389, 'depth': 3, 'l2_leaf_reg': 4.944272225334265,
    'bagging_temperature': 1.4823308606638113, 'random_strength': 0.4685604025205004, 'min_data_in_leaf': 46,
}
LGB_PARAMS = {
    'learning_rate': 0.022992006545037823, 'num_leaves': 110, 'max_depth': 3, 'min_child_samples': 27,
    'subsample': 0.9312452053625488, 'colsample_bytree': 0.8234901310320267,
    'reg_alpha': 0.012423757285817386, 'reg_lambda': 0.04673443002441543,
}

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# ========================================
# 1단계: 전략 08 파이프라인으로 Test 예측 (Pseudo Label 생성용)
# ========================================
print("=" * 60)
print("[1/2] 전략 08 파이프라인으로 Pseudo Label 생성")
print("=" * 60)

train_p = add_feature_engineering(base_preprocess(train))
test_p = add_feature_engineering(base_preprocess(test))

train_cb, test_cb = encode_categoricals(train_p, test_p, as_category=False)
X_cb = train_cb.drop(columns=['Target'])
y_cb = np.log1p(train_cb['Target'])
X_test_cb = test_cb
cat_indices = [X_cb.columns.get_loc(c) for c in CAT_FEATURES]

train_lgb, test_lgb = encode_categoricals(train_p, test_p, as_category=True)
X_lgb = train_lgb.drop(columns=['Target'])
y_lgb = np.log1p(train_lgb['Target'])
X_test_lgb = test_lgb

cb_oof = np.zeros(len(X_cb)); cb_test = np.zeros(len(X_test_cb))
lgb_oof = np.zeros(len(X_lgb)); lgb_test = np.zeros(len(X_test_lgb))

print("CatBoost...", end=" ")
for fold, (tr, va) in enumerate(kf.split(X_cb)):
    m = CatBoostRegressor(loss_function='RMSE', random_seed=42, verbose=0,
                          iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
    m.fit(X_cb.iloc[tr], y_cb.iloc[tr], eval_set=(X_cb.iloc[va], y_cb.iloc[va]), cat_features=cat_indices)
    cb_oof[va] = np.expm1(m.predict(X_cb.iloc[va]))
    cb_test += np.expm1(m.predict(X_test_cb)) / N_SPLITS
print(f"OOF {np.sqrt(np.mean((cb_oof - y_true)**2)):,.0f}")

print("LightGBM...", end=" ")
for fold, (tr, va) in enumerate(kf.split(X_lgb)):
    m2 = lgb.LGBMRegressor(objective='regression', metric='rmse', verbose=-1,
                            random_state=42, n_estimators=3000, **LGB_PARAMS)
    m2.fit(X_lgb.iloc[tr], y_lgb.iloc[tr], eval_set=[(X_lgb.iloc[va], y_lgb.iloc[va])],
           callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])
    lgb_oof[va] = np.expm1(m2.predict(X_lgb.iloc[va]))
    lgb_test += np.expm1(m2.predict(X_test_lgb)) / N_SPLITS
print(f"OOF {np.sqrt(np.mean((lgb_oof - y_true)**2)):,.0f}")

# Ridge 스태킹으로 Test 예측
stack_tr = np.column_stack([cb_oof, lgb_oof])
stack_te = np.column_stack([cb_test, lgb_test])
test_pred_raw = np.zeros(len(stack_te))
for fold, (tr, va) in enumerate(kf.split(stack_tr)):
    meta = Ridge(alpha=1.0)
    meta.fit(stack_tr[tr], y_true[tr])
    test_pred_raw += meta.predict(stack_te) / N_SPLITS

print(f"\nTest 예측 완료: 평균 {test_pred_raw.mean():,.0f}")

# ========================================
# Pseudo Label 생성: 안정적 구만 필터
# ========================================
pseudo_mask = test['Gu'].isin(STABLE_GUS)
pseudo_count = pseudo_mask.sum()

pseudo_df = test[pseudo_mask].copy()
pseudo_df['Target'] = test_pred_raw[pseudo_mask]

print(f"\nPseudo Label: {pseudo_count}건 ({', '.join(STABLE_GUS)})")
print(f"  Pseudo 평균 가격: {pseudo_df['Target'].mean():,.0f}")
print(f"  Train 평균 가격: {train['Target'].mean():,.0f}")

# Train + Pseudo 합치기
train_aug = pd.concat([train, pseudo_df], ignore_index=True)
y_aug = train_aug['Target'].values
print(f"\n합산 데이터: {len(train_aug)}건 (Train {len(train)} + Pseudo {pseudo_count})")

# ========================================
# 2단계: 합산 데이터로 재학습 → 전체 Test 예측
# ========================================
print("\n" + "=" * 60)
print("[2/2] 합산 데이터로 재학습")
print("=" * 60)

train_aug_p = add_feature_engineering(base_preprocess(train_aug))
test_p2 = add_feature_engineering(base_preprocess(test))

train_cb2, test_cb2 = encode_categoricals(train_aug_p, test_p2, as_category=False)
X_cb2 = train_cb2.drop(columns=['Target'])
y_cb2 = np.log1p(train_cb2['Target'])
X_test_cb2 = test_cb2
cat_indices2 = [X_cb2.columns.get_loc(c) for c in CAT_FEATURES]

train_lgb2, test_lgb2 = encode_categoricals(train_aug_p, test_p2, as_category=True)
X_lgb2 = train_lgb2.drop(columns=['Target'])
y_lgb2 = np.log1p(train_lgb2['Target'])
X_test_lgb2 = test_lgb2

cb_oof2 = np.zeros(len(X_cb2)); cb_test2 = np.zeros(len(X_test_cb2))
lgb_oof2 = np.zeros(len(X_lgb2)); lgb_test2 = np.zeros(len(X_test_lgb2))

kf2 = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

print("CatBoost 재학습...")
for fold, (tr, va) in enumerate(kf2.split(X_cb2)):
    print(f"  Fold {fold+1}/{N_SPLITS}", end=" → ")
    m = CatBoostRegressor(loss_function='RMSE', random_seed=42, verbose=0,
                          iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
    m.fit(X_cb2.iloc[tr], y_cb2.iloc[tr], eval_set=(X_cb2.iloc[va], y_cb2.iloc[va]), cat_features=cat_indices2)
    cb_oof2[va] = np.expm1(m.predict(X_cb2.iloc[va]))
    cb_test2 += np.expm1(m.predict(X_test_cb2)) / N_SPLITS
    fold_rmse = np.sqrt(np.mean((cb_oof2[va] - y_aug[va]) ** 2))
    print(f"RMSE: {fold_rmse:,.0f}")
print(f"  OOF RMSE: {np.sqrt(np.mean((cb_oof2 - y_aug)**2)):,.0f}")

print("\nLightGBM 재학습...")
for fold, (tr, va) in enumerate(kf2.split(X_lgb2)):
    print(f"  Fold {fold+1}/{N_SPLITS}", end=" → ")
    m2 = lgb.LGBMRegressor(objective='regression', metric='rmse', verbose=-1,
                            random_state=42, n_estimators=3000, **LGB_PARAMS)
    m2.fit(X_lgb2.iloc[tr], y_lgb2.iloc[tr], eval_set=[(X_lgb2.iloc[va], y_lgb2.iloc[va])],
           callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])
    lgb_oof2[va] = np.expm1(m2.predict(X_lgb2.iloc[va]))
    lgb_test2 += np.expm1(m2.predict(X_test_lgb2)) / N_SPLITS
    fold_rmse = np.sqrt(np.mean((lgb_oof2[va] - y_aug[va]) ** 2))
    print(f"RMSE: {fold_rmse:,.0f}")
print(f"  OOF RMSE: {np.sqrt(np.mean((lgb_oof2 - y_aug)**2)):,.0f}")

# Ridge 스태킹
print("\nRidge 스태킹...")
stack_tr2 = np.column_stack([cb_oof2, lgb_oof2])
stack_te2 = np.column_stack([cb_test2, lgb_test2])
s_oof2 = np.zeros(len(y_aug)); s_test2 = np.zeros(len(stack_te2))
for fold, (tr, va) in enumerate(kf2.split(stack_tr2)):
    meta = Ridge(alpha=1.0)
    meta.fit(stack_tr2[tr], y_aug[tr])
    s_oof2[va] = meta.predict(stack_tr2[va])
    s_test2 += meta.predict(stack_te2) / N_SPLITS

rmse_aug = np.sqrt(np.mean((s_oof2 - y_aug)**2))
rmse_orig = np.sqrt(np.mean((s_oof2[:len(y_true)] - y_true)**2))
print(f"  합산 OOF RMSE: {rmse_aug:,.0f}")
print(f"  원본 Train만 OOF RMSE: {rmse_orig:,.0f}")

# 구별 트렌드 보정
final_pred = s_test2 * trend_correction

print(f"\n트렌드 보정 전 평균: {s_test2.mean():,.0f}")
print(f"트렌드 보정 후 평균: {final_pred.mean():,.0f}")

# === 제출 파일 생성 ===
sub = sample_sub.copy()
sub['Target'] = final_pred
sub.to_csv(f'{OUTPUT_DIR}/submission.csv', index=False)
print(f"\n제출 파일 저장 완료: {OUTPUT_DIR}/submission.csv")
