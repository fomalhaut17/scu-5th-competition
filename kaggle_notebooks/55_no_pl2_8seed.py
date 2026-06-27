"""
55 NO-PL2 8SEED
파이프라인: PL2 제거 + 12모델 × 8시드 평균 → Ridge → GTR
변경점: 53(4시드 Public 2,088)에서 시드 8개로 확장, 분산 추가 감소
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
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
SEEDS = [42, 123, 456, 789, 7, 2024, 314, 999]

# === 데이터 로드 ===
train_orig = pd.read_csv(f'{INPUT_DIR}/seoul_real_estate_train.csv')
test_orig = pd.read_csv(f'{INPUT_DIR}/seoul_real_estate_test.csv')
sample_sub = pd.read_csv(f'{INPUT_DIR}/sample_submission.csv')
y_true_orig = train_orig['Target'].values
n_orig = len(train_orig)
area_train = train_orig['Exclusive_Area'].values
area_test = test_orig['Exclusive_Area'].values

# === GTR ===
last_train_ym = train_orig['Transaction_YearMonth'].max()
last_train_seq = (last_train_ym // 100 - 2024) * 12 + last_train_ym % 100
gu_growth = {}
for gu in train_orig['Gu'].unique():
    monthly = train_orig[train_orig['Gu'] == gu].groupby('Transaction_YearMonth')['Target'].mean()
    gu_growth[gu] = monthly.pct_change().dropna().mean()
test_seq = (test_orig['Transaction_YearMonth'] // 100 - 2024) * 12 + test_orig['Transaction_YearMonth'] % 100
months_ahead = test_seq.values - last_train_seq
trend_correction = np.array([(1 + gu_growth[gu]) ** m for gu, m in zip(test_orig['Gu'], months_ahead)])

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

def prepare_data(train_df, test_df):
    train_p = add_feature_engineering(base_preprocess(train_df))
    test_p = add_feature_engineering(base_preprocess(test_df))
    train_cb, test_cb = encode_categoricals(train_p, test_p, as_category=False)
    train_lgb, test_lgb = encode_categoricals(train_p, test_p, as_category=True)
    return train_cb, test_cb, train_lgb, test_lgb

CB_PARAMS = {
    'learning_rate': 0.010118898857677389, 'depth': 3,
    'l2_leaf_reg': 4.944272225334265, 'bagging_temperature': 1.4823308606638113,
    'random_strength': 0.4685604025205004, 'min_data_in_leaf': 46,
}
LGB_PARAMS = {
    'learning_rate': 0.022992006545037823, 'num_leaves': 110, 'max_depth': 3,
    'min_child_samples': 27, 'subsample': 0.9312452053625488,
    'colsample_bytree': 0.8234901310320267, 'reg_alpha': 0.012423757285817386,
    'reg_lambda': 0.04673443002441543,
}
LGB_ET_PARAMS = {
    'extra_trees': True, 'max_depth': 3, 'num_leaves': 31,
    'feature_fraction': 0.6, 'bagging_fraction': 0.7, 'bagging_freq': 1,
    'min_child_samples': 30, 'learning_rate': 0.02,
}

train_cb, test_cb, train_lgb, test_lgb = prepare_data(train_orig, test_orig)
X_cb = train_cb.drop(columns=['Target'])
X_test_cb = test_cb
cat_idx = [X_cb.columns.get_loc(c) for c in CAT_FEATURES]
X_lgb = train_lgb.drop(columns=['Target'])
X_test_lgb = test_lgb

y_log = np.log1p(y_true_orig)
y_raw = y_true_orig.copy()
y_up_log = np.log1p(y_true_orig / area_train)
y_up_raw = (y_true_orig / area_train).astype(float)

# ========================================
# 4시드 × 12모델 학습 (PL2 없이)
# ========================================
all_seed_tests = []

for seed in SEEDS:
    print(f"\n{'=' * 60}")
    print(f"[Seed {seed}] 12모델 학습 (PL2 없음)")
    print("=" * 60)

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)

    all_oof = {}
    all_tpred = {}

    for fold, (tr, va) in enumerate(kf.split(X_cb)):
        print(f"  Fold {fold+1}/5")

        m = CatBoostRegressor(loss_function='RMSE', random_seed=seed, verbose=0, iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
        m.fit(X_cb.iloc[tr], y_log[tr], eval_set=(X_cb.iloc[va], y_log[va]), cat_features=cat_idx)
        all_oof.setdefault('cb_log', np.zeros(n_orig))[va] = np.expm1(m.predict(X_cb.iloc[va]))
        all_tpred['cb_log'] = all_tpred.get('cb_log', np.zeros(len(test_orig))) + np.expm1(m.predict(X_test_cb)) / 5

        m = CatBoostRegressor(loss_function='RMSE', random_seed=seed, verbose=0, iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
        m.fit(X_cb.iloc[tr], y_raw[tr], eval_set=(X_cb.iloc[va], y_raw[va]), cat_features=cat_idx)
        all_oof.setdefault('cb_raw', np.zeros(n_orig))[va] = m.predict(X_cb.iloc[va])
        all_tpred['cb_raw'] = all_tpred.get('cb_raw', np.zeros(len(test_orig))) + m.predict(X_test_cb) / 5

        m = lgb.LGBMRegressor(objective='regression', metric='rmse', verbose=-1, random_state=seed, n_estimators=3000, **LGB_PARAMS)
        m.fit(X_lgb.iloc[tr], y_log[tr], eval_set=[(X_lgb.iloc[va], y_log[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
        all_oof.setdefault('lgb_log', np.zeros(n_orig))[va] = np.expm1(m.predict(X_lgb.iloc[va]))
        all_tpred['lgb_log'] = all_tpred.get('lgb_log', np.zeros(len(test_orig))) + np.expm1(m.predict(X_test_lgb)) / 5

        m = lgb.LGBMRegressor(objective='regression', metric='rmse', verbose=-1, random_state=seed, n_estimators=3000, **LGB_PARAMS)
        m.fit(X_lgb.iloc[tr], y_raw[tr], eval_set=[(X_lgb.iloc[va], y_raw[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
        all_oof.setdefault('lgb_raw', np.zeros(n_orig))[va] = m.predict(X_lgb.iloc[va])
        all_tpred['lgb_raw'] = all_tpred.get('lgb_raw', np.zeros(len(test_orig))) + m.predict(X_test_lgb) / 5

        m = CatBoostRegressor(loss_function='RMSE', random_seed=seed, verbose=0, iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
        m.fit(X_cb.iloc[tr], y_up_log[tr], eval_set=(X_cb.iloc[va], y_up_log[va]), cat_features=cat_idx)
        all_oof.setdefault('cb_up_log', np.zeros(n_orig))[va] = np.expm1(m.predict(X_cb.iloc[va]))
        all_tpred['cb_up_log'] = all_tpred.get('cb_up_log', np.zeros(len(test_orig))) + np.expm1(m.predict(X_test_cb)) / 5

        m = CatBoostRegressor(loss_function='RMSE', random_seed=seed, verbose=0, iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
        m.fit(X_cb.iloc[tr], y_up_raw[tr], eval_set=(X_cb.iloc[va], y_up_raw[va]), cat_features=cat_idx)
        all_oof.setdefault('cb_up_raw', np.zeros(n_orig))[va] = m.predict(X_cb.iloc[va])
        all_tpred['cb_up_raw'] = all_tpred.get('cb_up_raw', np.zeros(len(test_orig))) + m.predict(X_test_cb) / 5

        m = lgb.LGBMRegressor(objective='regression', metric='rmse', verbose=-1, random_state=seed, n_estimators=3000, **LGB_PARAMS)
        m.fit(X_lgb.iloc[tr], y_up_log[tr], eval_set=[(X_lgb.iloc[va], y_up_log[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
        all_oof.setdefault('lgb_up_log', np.zeros(n_orig))[va] = np.expm1(m.predict(X_lgb.iloc[va]))
        all_tpred['lgb_up_log'] = all_tpred.get('lgb_up_log', np.zeros(len(test_orig))) + np.expm1(m.predict(X_test_lgb)) / 5

        m = lgb.LGBMRegressor(objective='regression', metric='rmse', verbose=-1, random_state=seed, n_estimators=3000, **LGB_PARAMS)
        m.fit(X_lgb.iloc[tr], y_up_raw[tr], eval_set=[(X_lgb.iloc[va], y_up_raw[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
        all_oof.setdefault('lgb_up_raw', np.zeros(n_orig))[va] = m.predict(X_lgb.iloc[va])
        all_tpred['lgb_up_raw'] = all_tpred.get('lgb_up_raw', np.zeros(len(test_orig))) + m.predict(X_test_lgb) / 5

        scaler = StandardScaler()
        X_sc_tr = scaler.fit_transform(X_cb.iloc[tr])
        X_sc_va = scaler.transform(X_cb.iloc[va])
        X_sc_te = scaler.transform(X_test_cb)

        m = ExtraTreesRegressor(n_estimators=500, max_depth=12, min_samples_leaf=10, random_state=seed, n_jobs=-1)
        m.fit(X_sc_tr, y_log[tr])
        all_oof.setdefault('et_log', np.zeros(n_orig))[va] = np.expm1(m.predict(X_sc_va))
        all_tpred['et_log'] = all_tpred.get('et_log', np.zeros(len(test_orig))) + np.expm1(m.predict(X_sc_te)) / 5

        m = ExtraTreesRegressor(n_estimators=500, max_depth=12, min_samples_leaf=10, random_state=seed, n_jobs=-1)
        m.fit(X_sc_tr, y_raw[tr])
        all_oof.setdefault('et_raw', np.zeros(n_orig))[va] = m.predict(X_sc_va)
        all_tpred['et_raw'] = all_tpred.get('et_raw', np.zeros(len(test_orig))) + m.predict(X_sc_te) / 5

        m = lgb.LGBMRegressor(objective='regression', metric='rmse', verbose=-1, random_state=seed, n_estimators=3000, **LGB_ET_PARAMS)
        m.fit(X_lgb.iloc[tr], y_log[tr], eval_set=[(X_lgb.iloc[va], y_log[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
        all_oof.setdefault('lgbet_log', np.zeros(n_orig))[va] = np.expm1(m.predict(X_lgb.iloc[va]))
        all_tpred['lgbet_log'] = all_tpred.get('lgbet_log', np.zeros(len(test_orig))) + np.expm1(m.predict(X_test_lgb)) / 5

        m = lgb.LGBMRegressor(objective='regression', metric='rmse', verbose=-1, random_state=seed, n_estimators=3000, **LGB_ET_PARAMS)
        m.fit(X_lgb.iloc[tr], y_raw[tr], eval_set=[(X_lgb.iloc[va], y_raw[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
        all_oof.setdefault('lgbet_raw', np.zeros(n_orig))[va] = m.predict(X_lgb.iloc[va])
        all_tpred['lgbet_raw'] = all_tpred.get('lgbet_raw', np.zeros(len(test_orig))) + m.predict(X_test_lgb) / 5

    # 평당가 환산
    for k in ['cb_up_log', 'cb_up_raw', 'lgb_up_log', 'lgb_up_raw']:
        all_oof[k] = all_oof[k] * area_train
        all_tpred[k] = all_tpred[k] * area_test

    # Ridge 스태킹
    base_12 = ['cb_log', 'cb_raw', 'lgb_log', 'lgb_raw',
               'cb_up_log', 'cb_up_raw', 'lgb_up_log', 'lgb_up_raw',
               'et_log', 'et_raw', 'lgbet_log', 'lgbet_raw']

    st_tr = np.column_stack([all_oof[k] for k in base_12])
    st_te = np.column_stack([all_tpred[k] for k in base_12])

    kf_meta = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    best_rmse = float('inf')
    best_test = None

    for alpha in [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]:
        s_oof = np.zeros(n_orig)
        s_test = np.zeros(len(test_orig))
        for fold, (tr, va) in enumerate(kf_meta.split(st_tr)):
            meta = Ridge(alpha=alpha)
            meta.fit(st_tr[tr], y_true_orig[tr])
            s_oof[va] = meta.predict(st_tr[va])
            s_test += meta.predict(st_te) / N_SPLITS
        rmse = np.sqrt(np.mean((s_oof - y_true_orig) ** 2))
        if rmse < best_rmse:
            best_rmse = rmse
            best_test = s_test.copy()

    print(f"  Seed {seed}: OOF {best_rmse:,.0f}")
    all_seed_tests.append(best_test)

# ========================================
# 4시드 평균 + GTR
# ========================================
print(f"\n{'=' * 60}")
print("4시드 평균 + GTR")
print("=" * 60)

avg_test = np.mean(all_seed_tests, axis=0)
final_pred = avg_test * trend_correction

sub = sample_sub.copy()
sub['Target'] = final_pred
sub.to_csv(f'{OUTPUT_DIR}/submission.csv', index=False)
print(f"\n제출 파일 저장 완료: {OUTPUT_DIR}/submission.csv")
