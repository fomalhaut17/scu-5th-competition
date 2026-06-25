"""
27 UNIT PRICE CROSS BLEND GTR
파이프라인: FE → 기존 4모델 + 평당가 4모델 = 8모델 단순평균 → 구별 트렌드 보정
변경점: Target/Area 예측 후 Area 환산, 기존 모델과 크로스 블렌딩, OOF 2,217
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
MODELS = ['cb_log', 'cb_raw', 'lgb_log', 'lgb_raw']

# === 데이터 로드 ===
train = pd.read_csv(f'{INPUT_DIR}/seoul_real_estate_train.csv')
test = pd.read_csv(f'{INPUT_DIR}/seoul_real_estate_test.csv')
sample_sub = pd.read_csv(f'{INPUT_DIR}/sample_submission.csv')
y_true = train['Target'].values
area_train = train['Exclusive_Area'].values
area_test = test['Exclusive_Area'].values

# === 구별 트렌드 보정 ===
last_train_ym = train['Transaction_YearMonth'].max()
last_train_seq = (last_train_ym // 100 - 2024) * 12 + last_train_ym % 100

gu_growth = {}
for gu in train['Gu'].unique():
    monthly = train[train['Gu'] == gu].groupby('Transaction_YearMonth')['Target'].mean()
    gu_growth[gu] = monthly.pct_change().dropna().mean()

print("=== 구별 월성장률 ===")
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

# === 전처리 실행 ===
train_p = add_feature_engineering(base_preprocess(train))
test_p = add_feature_engineering(base_preprocess(test))

train_cb, test_cb = encode_categoricals(train_p, test_p, as_category=False)
X_cb = train_cb.drop(columns=['Target'])
X_test_cb = test_cb
cat_indices = [X_cb.columns.get_loc(c) for c in CAT_FEATURES]

train_lgb, test_lgb = encode_categoricals(train_p, test_p, as_category=True)
X_lgb = train_lgb.drop(columns=['Target'])
X_test_lgb = test_lgb

# === 타겟 정의 ===
y_log = np.log1p(y_true)
y_raw = y_true.astype(float)

unit_price = y_true / area_train
y_up_log = np.log1p(unit_price)
y_up_raw = unit_price.astype(float)

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

def train_4models(y_log_t, y_raw_t, label):
    """4모델 학습, OOF + Test 예측 반환 (원본 스케일)"""
    oof = {k: np.zeros(len(X_cb)) for k in MODELS}
    tpred = {k: np.zeros(len(X_test_cb)) for k in MODELS}

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_cb)):
        print(f"  [{label}] Fold {fold+1}/{N_SPLITS}")

        m = CatBoostRegressor(loss_function='RMSE', random_seed=42, verbose=0,
                              iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
        m.fit(X_cb.iloc[tr_idx], y_log_t[tr_idx],
              eval_set=(X_cb.iloc[va_idx], y_log_t[va_idx]), cat_features=cat_indices)
        oof['cb_log'][va_idx] = np.expm1(m.predict(X_cb.iloc[va_idx]))
        tpred['cb_log'] += np.expm1(m.predict(X_test_cb)) / N_SPLITS

        m = CatBoostRegressor(loss_function='RMSE', random_seed=42, verbose=0,
                              iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
        m.fit(X_cb.iloc[tr_idx], y_raw_t[tr_idx],
              eval_set=(X_cb.iloc[va_idx], y_raw_t[va_idx]), cat_features=cat_indices)
        oof['cb_raw'][va_idx] = m.predict(X_cb.iloc[va_idx])
        tpred['cb_raw'] += m.predict(X_test_cb) / N_SPLITS

        m = lgb.LGBMRegressor(objective='regression', metric='rmse',
                              verbose=-1, random_state=42, n_estimators=3000, **LGB_PARAMS)
        m.fit(X_lgb.iloc[tr_idx], y_log_t[tr_idx],
              eval_set=[(X_lgb.iloc[va_idx], y_log_t[va_idx])],
              callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])
        oof['lgb_log'][va_idx] = np.expm1(m.predict(X_lgb.iloc[va_idx]))
        tpred['lgb_log'] += np.expm1(m.predict(X_test_lgb)) / N_SPLITS

        m = lgb.LGBMRegressor(objective='regression', metric='rmse',
                              verbose=-1, random_state=42, n_estimators=3000, **LGB_PARAMS)
        m.fit(X_lgb.iloc[tr_idx], y_raw_t[tr_idx],
              eval_set=[(X_lgb.iloc[va_idx], y_raw_t[va_idx])],
              callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])
        oof['lgb_raw'][va_idx] = m.predict(X_lgb.iloc[va_idx])
        tpred['lgb_raw'] += m.predict(X_test_lgb) / N_SPLITS

    return oof, tpred

# ========================================
# 기존 4모델 (Target 직접 예측)
# ========================================
print("\n" + "=" * 60)
print("[Part 1] 기존 4모델: Target 직접 예측")
print("=" * 60)

oof_base, tpred_base = train_4models(y_log, y_raw, "기존")

print("\n  개별 OOF RMSE:")
for k in MODELS:
    rmse = np.sqrt(np.mean((oof_base[k] - y_true) ** 2))
    print(f"    {k:10s}: {rmse:,.0f}")

# ========================================
# 평당가 4모델 (Target/Area 예측 → ×Area 환산)
# ========================================
print("\n" + "=" * 60)
print("[Part 2] 평당가 4모델: Target/Area → ×Area")
print("=" * 60)

oof_unit, tpred_unit = train_4models(y_up_log, y_up_raw, "평당가")

for k in MODELS:
    oof_unit[k] = oof_unit[k] * area_train
    tpred_unit[k] = tpred_unit[k] * area_test

print("\n  개별 OOF RMSE (총가격 환산 후):")
for k in MODELS:
    rmse = np.sqrt(np.mean((oof_unit[k] - y_true) ** 2))
    print(f"    {k:10s}: {rmse:,.0f}")

# ========================================
# 8모델 크로스 블렌딩
# ========================================
print("\n" + "=" * 60)
print("[Part 3] 8모델 크로스 블렌딩")
print("=" * 60)

all_oof = np.mean([oof_base[k] for k in MODELS] + [oof_unit[k] for k in MODELS], axis=0)
all_test = np.mean([tpred_base[k] for k in MODELS] + [tpred_unit[k] for k in MODELS], axis=0)
rmse_all = np.sqrt(np.mean((all_oof - y_true) ** 2))

base_avg_oof = np.mean([oof_base[k] for k in MODELS], axis=0)
rmse_base = np.sqrt(np.mean((base_avg_oof - y_true) ** 2))

print(f"  기존 4모델 단순평균  OOF RMSE: {rmse_base:,.0f}")
print(f"  8모델 크로스 단순평균 OOF RMSE: {rmse_all:,.0f}")
print(f"  개선: {rmse_base - rmse_all:+,.0f}")

# === 트렌드 보정 + 제출 ===
final_pred = all_test * trend_correction

print(f"\n{'=' * 60}")
print(f"최종: 8모델 단순평균 OOF RMSE {rmse_all:,.0f}")
print(f"전략 26: OOF 2,215 / Public 2,149.6")
print(f"전략 25: OOF 2,226")
print(f"{'=' * 60}")

print(f"\n트렌드 보정 전 평균: {all_test.mean():,.0f}")
print(f"트렌드 보정 후 평균: {final_pred.mean():,.0f}")

sub = sample_sub.copy()
sub['Target'] = final_pred
sub.to_csv(f'{OUTPUT_DIR}/submission.csv', index=False)
print(f"\n제출 파일 저장 완료: {OUTPUT_DIR}/submission.csv")
