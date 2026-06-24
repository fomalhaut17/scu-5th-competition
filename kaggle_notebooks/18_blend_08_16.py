"""
18 BLEND 08+16
파이프라인: 전략 08(기본) + 전략 16(지리피처) 예측을 70:30 블렌딩
변경점: 다른 피처 세트를 쓴 두 파이프라인의 예측을 조합하여 다양성 확보
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
BLEND_W08 = 0.95

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

# === 공통 전처리 ===
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

# === 지리 피처 (전략 16용) ===
DONG_COORDS = {
    'Apgujeong': (37.527, 127.028), 'Daechi': (37.494, 127.063), 'Samsung': (37.511, 127.059),
    'Bangbae': (37.483, 126.981), 'Banpo': (37.505, 127.008), 'Seocho': (37.492, 127.007),
    'Garak': (37.497, 127.118), 'Jamsil': (37.513, 127.100), 'Munjjeong': (37.485, 127.123),
    'Geumho': (37.556, 127.018), 'Oksu': (37.543, 127.017), 'Seongsu': (37.544, 127.056),
    'Hannam': (37.535, 127.003), 'Ichon': (37.522, 126.970), 'Itaewon': (37.534, 126.994),
    'Ahyeon': (37.558, 126.957), 'Hapjeong': (37.550, 126.914), 'Sangam': (37.577, 126.890),
    'Junggye': (37.643, 127.072), 'Sanggye': (37.653, 127.072), 'Wolgye': (37.632, 127.068),
    'Bulgwang': (37.615, 126.928), 'Susaek': (37.584, 126.905), 'Yeonsinnae': (37.619, 126.922),
}
GANGNAM_STN = (37.4979, 127.0276)
SEOUL_STN = (37.5547, 126.9707)
HAN_RIVER_LAT = 37.528

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

def add_geo_features(df):
    df = df.copy()
    lats = df['Dong'].map(lambda d: DONG_COORDS[d][0])
    lons = df['Dong'].map(lambda d: DONG_COORDS[d][1])
    df['Dist_Gangnam'] = [haversine_km(la, lo, *GANGNAM_STN) for la, lo in zip(lats, lons)]
    df['Dist_Seoul_Stn'] = [haversine_km(la, lo, *SEOUL_STN) for la, lo in zip(lats, lons)]
    df['Dist_Han_River'] = np.abs(lats - HAN_RIVER_LAT) * 111
    df['Latitude'] = lats
    df['Longitude'] = lons
    return df

# === 공통 모델 파라미터 ===
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

def run_pipeline(train_p, test_p, label):
    """CB + LGB → Ridge 스태킹 파이프라인 실행, 보정 전 예측값 반환"""
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

    print(f"\n{'='*50}")
    print(f"[{label}] CatBoost 학습")
    print(f"{'='*50}")
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_cb)):
        print(f"  Fold {fold+1}/{N_SPLITS}", end=" → ")
        m = CatBoostRegressor(loss_function='RMSE', random_seed=42, verbose=0,
                              iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
        m.fit(X_cb.iloc[tr_idx], y_cb.iloc[tr_idx],
              eval_set=(X_cb.iloc[va_idx], y_cb.iloc[va_idx]), cat_features=cat_indices)
        cb_oof[va_idx] = np.expm1(m.predict(X_cb.iloc[va_idx]))
        cb_test += np.expm1(m.predict(X_test_cb)) / N_SPLITS
        fold_rmse = np.sqrt(np.mean((cb_oof[va_idx] - y_true[va_idx]) ** 2))
        print(f"RMSE: {fold_rmse:,.0f}")
    print(f"  OOF RMSE: {np.sqrt(np.mean((cb_oof - y_true) ** 2)):,.0f}")

    print(f"\n{'='*50}")
    print(f"[{label}] LightGBM 학습")
    print(f"{'='*50}")
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_lgb)):
        print(f"  Fold {fold+1}/{N_SPLITS}", end=" → ")
        m2 = lgb.LGBMRegressor(objective='regression', metric='rmse',
                                verbose=-1, random_state=42, n_estimators=3000, **LGB_PARAMS)
        m2.fit(X_lgb.iloc[tr_idx], y_lgb.iloc[tr_idx],
               eval_set=[(X_lgb.iloc[va_idx], y_lgb.iloc[va_idx])],
               callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])
        lgb_oof[va_idx] = np.expm1(m2.predict(X_lgb.iloc[va_idx]))
        lgb_test += np.expm1(m2.predict(X_test_lgb)) / N_SPLITS
        fold_rmse = np.sqrt(np.mean((lgb_oof[va_idx] - y_true[va_idx]) ** 2))
        print(f"RMSE: {fold_rmse:,.0f}")
    print(f"  OOF RMSE: {np.sqrt(np.mean((lgb_oof - y_true) ** 2)):,.0f}")

    stack_tr = np.column_stack([cb_oof, lgb_oof])
    stack_te = np.column_stack([cb_test, lgb_test])
    s_oof = np.zeros(len(y_true)); s_test = np.zeros(len(stack_te))
    for fold, (tr_idx, va_idx) in enumerate(kf.split(stack_tr)):
        meta = Ridge(alpha=1.0)
        meta.fit(stack_tr[tr_idx], y_true[tr_idx])
        s_oof[va_idx] = meta.predict(stack_tr[va_idx])
        s_test += meta.predict(stack_te) / N_SPLITS

    rmse = np.sqrt(np.mean((s_oof - y_true) ** 2))
    print(f"\n  [{label}] 스태킹 OOF RMSE: {rmse:,.0f}")
    return s_test

# ========================================
# 파이프라인 A: 전략 08 (기본)
# ========================================
print("=" * 60)
print("파이프라인 A: 전략 08 (기본)")
print("=" * 60)
train_08 = add_feature_engineering(base_preprocess(train))
test_08 = add_feature_engineering(base_preprocess(test))
pred_08 = run_pipeline(train_08, test_08, "08")

# ========================================
# 파이프라인 B: 전략 16 (지리 피처)
# ========================================
print("\n\n" + "=" * 60)
print("파이프라인 B: 전략 16 (지리 피처)")
print("=" * 60)
train_16 = add_geo_features(add_feature_engineering(base_preprocess(train)))
test_16 = add_geo_features(add_feature_engineering(base_preprocess(test)))
pred_16 = run_pipeline(train_16, test_16, "16")

# ========================================
# 블렌딩 + 트렌드 보정
# ========================================
print("\n\n" + "=" * 60)
print(f"블렌딩: 08 × {BLEND_W08:.0%} + 16 × {1-BLEND_W08:.0%}")
print("=" * 60)

blended = BLEND_W08 * pred_08 + (1 - BLEND_W08) * pred_16
final_pred = blended * trend_correction

print(f"  트렌드 보정 전 평균: {blended.mean():,.0f}")
print(f"  트렌드 보정 후 평균: {final_pred.mean():,.0f}")

# === 제출 파일 생성 ===
sub = sample_sub.copy()
sub['Target'] = final_pred
sub.to_csv(f'{OUTPUT_DIR}/submission.csv', index=False)
print(f"\n제출 파일 저장 완료: {OUTPUT_DIR}/submission.csv")
