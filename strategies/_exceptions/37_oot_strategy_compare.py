"""
[L4-37] OOT 전략 비교 검증
──────────────────────────
목적: 전략 08/25/26/28을 OOT(마지막 3개월 holdout)로 비교하여
      각 구성요소(Scale, PL2, 평당가)의 미래 데이터 일반화 효과 측정
배울 점:
  - OOF vs OOT vs Public 세 축 비교 → 어떤 검증이 실제 순위와 가장 가까운가
  - 각 구성요소의 OOT 기여도 → 진짜 효과 vs OOF 낙관
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
from utils import load_data, oot_split
import warnings
warnings.filterwarnings('ignore')

N_SPLITS = 5
CAT_FEATURES = ['Gu', 'Dong']
MODELS = ['cb_log', 'cb_raw', 'lgb_log', 'lgb_raw']

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

# === 공통 함수 ===
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

def calc_rmse(pred, true):
    return np.sqrt(np.mean((pred - true) ** 2))

def compute_gu_trend(train_raw, val_raw):
    last_ym = train_raw['Transaction_YearMonth'].max()
    last_seq = (last_ym // 100 - 2024) * 12 + last_ym % 100
    gu_growth = {}
    for gu in train_raw['Gu'].unique():
        monthly = train_raw[train_raw['Gu'] == gu].groupby('Transaction_YearMonth')['Target'].mean()
        gu_growth[gu] = monthly.pct_change().dropna().mean()
    val_seq = (val_raw['Transaction_YearMonth'] // 100 - 2024) * 12 + val_raw['Transaction_YearMonth'] % 100
    months_ahead = val_seq.values - last_seq
    correction = np.array([(1 + gu_growth.get(gu, 0)) ** m
                           for gu, m in zip(val_raw['Gu'], months_ahead)])
    return correction

def train_models(train_cb, test_cb, train_lgb, test_lgb, kf,
                 y_log_override=None, y_raw_override=None,
                 return_fold_preds=False, label="", models=None):
    if models is None:
        models = MODELS

    X_cb = train_cb.drop(columns=['Target'])
    X_test_cb = test_cb.drop(columns=['Target']) if 'Target' in test_cb.columns else test_cb
    cat_idx = [X_cb.columns.get_loc(c) for c in CAT_FEATURES]

    X_lgb = train_lgb.drop(columns=['Target'])
    X_test_lgb = test_lgb.drop(columns=['Target']) if 'Target' in test_lgb.columns else test_lgb

    if y_log_override is not None:
        y_log = y_log_override
        y_raw = y_raw_override
    else:
        y_log = np.log1p(train_cb['Target'])
        y_raw = train_cb['Target'].values

    oof = {k: np.zeros(len(X_cb)) for k in models}
    tpred = {k: np.zeros(len(X_test_cb)) for k in models}
    fold_test_preds = {k: [] for k in models} if return_fold_preds else None

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_cb)):
        if label:
            print(f"    [{label}] Fold {fold+1}/{kf.n_splits}")

        if 'cb_log' in models:
            m = CatBoostRegressor(loss_function='RMSE', random_seed=42, verbose=0,
                                  iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
            m.fit(X_cb.iloc[tr_idx], y_log.iloc[tr_idx] if hasattr(y_log, 'iloc') else y_log[tr_idx],
                  eval_set=(X_cb.iloc[va_idx], y_log.iloc[va_idx] if hasattr(y_log, 'iloc') else y_log[va_idx]),
                  cat_features=cat_idx)
            oof['cb_log'][va_idx] = np.expm1(m.predict(X_cb.iloc[va_idx]))
            fp = np.expm1(m.predict(X_test_cb))
            tpred['cb_log'] += fp / kf.n_splits
            if return_fold_preds: fold_test_preds['cb_log'].append(fp)

        if 'cb_raw' in models:
            m = CatBoostRegressor(loss_function='RMSE', random_seed=42, verbose=0,
                                  iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
            m.fit(X_cb.iloc[tr_idx], y_raw[tr_idx],
                  eval_set=(X_cb.iloc[va_idx], y_raw[va_idx]), cat_features=cat_idx)
            oof['cb_raw'][va_idx] = m.predict(X_cb.iloc[va_idx])
            fp = m.predict(X_test_cb)
            tpred['cb_raw'] += fp / kf.n_splits
            if return_fold_preds: fold_test_preds['cb_raw'].append(fp)

        if 'lgb_log' in models:
            m = lgb.LGBMRegressor(objective='regression', metric='rmse',
                                  verbose=-1, random_state=42, n_estimators=3000, **LGB_PARAMS)
            m.fit(X_lgb.iloc[tr_idx], y_log.iloc[tr_idx] if hasattr(y_log, 'iloc') else y_log[tr_idx],
                  eval_set=[(X_lgb.iloc[va_idx], y_log.iloc[va_idx] if hasattr(y_log, 'iloc') else y_log[va_idx])],
                  callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])
            oof['lgb_log'][va_idx] = np.expm1(m.predict(X_lgb.iloc[va_idx]))
            fp = np.expm1(m.predict(X_test_lgb))
            tpred['lgb_log'] += fp / kf.n_splits
            if return_fold_preds: fold_test_preds['lgb_log'].append(fp)

        if 'lgb_raw' in models:
            m = lgb.LGBMRegressor(objective='regression', metric='rmse',
                                  verbose=-1, random_state=42, n_estimators=3000, **LGB_PARAMS)
            m.fit(X_lgb.iloc[tr_idx], y_raw[tr_idx],
                  eval_set=[(X_lgb.iloc[va_idx], y_raw[va_idx])],
                  callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])
            oof['lgb_raw'][va_idx] = m.predict(X_lgb.iloc[va_idx])
            fp = m.predict(X_test_lgb)
            tpred['lgb_raw'] += fp / kf.n_splits
            if return_fold_preds: fold_test_preds['lgb_raw'].append(fp)

    return oof, tpred, fold_test_preds


# ========================================
# 데이터 로드 + OOT 분할
# ========================================
train_raw, test_raw, sample_sub = load_data()

for holdout in [3]:
    train_idx, val_idx, cutoff_ym = oot_split(train_raw, holdout_months=holdout)
    oot_train = train_raw.iloc[train_idx].reset_index(drop=True)
    oot_val = train_raw.iloc[val_idx].reset_index(drop=True)
    y_val_true = oot_val['Target'].values
    n_oot_train = len(oot_train)

    print(f"\n{'=' * 70}")
    print(f"OOT 분할: Train {n_oot_train}건 (< {cutoff_ym}), Val {len(oot_val)}건 (>= {cutoff_ym})")
    print(f"Val YearMonth: {sorted(oot_val['Transaction_YearMonth'].unique())}")
    print(f"{'=' * 70}")

    trend_correction = compute_gu_trend(oot_train, oot_val)

    # ========================================
    # 전략 08: CB+LGB (log) → Ridge 스태킹
    # ========================================
    print(f"\n{'─' * 50}")
    print(f"[전략 08] CB+LGB log → Ridge 스태킹")
    print(f"{'─' * 50}")

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    train_cb, val_cb, train_lgb, val_lgb = prepare_data(oot_train, oot_val)

    oof_08, vpred_08, _ = train_models(
        train_cb, val_cb, train_lgb, val_lgb, kf,
        models=['cb_log', 'lgb_log'], label="08")

    y_oot_train = oot_train['Target'].values
    stack_train_08 = np.column_stack([oof_08['cb_log'], oof_08['lgb_log']])
    stack_val_08 = np.column_stack([vpred_08['cb_log'], vpred_08['lgb_log']])

    s08_oof = np.zeros(n_oot_train)
    s08_val = np.zeros(len(oot_val))
    for fold, (tr, va) in enumerate(kf.split(stack_train_08)):
        meta = Ridge(alpha=1.0)
        meta.fit(stack_train_08[tr], y_oot_train[tr])
        s08_oof[va] = meta.predict(stack_train_08[va])
        s08_val += meta.predict(stack_val_08) / N_SPLITS

    s08_oof_rmse = calc_rmse(s08_oof, y_oot_train)
    s08_oot_raw = calc_rmse(s08_val, y_val_true)
    s08_oot_gtr = calc_rmse(s08_val * trend_correction, y_val_true)
    print(f"  OOF: {s08_oof_rmse:,.0f} | OOT(raw): {s08_oot_raw:,.0f} | OOT(+GTR): {s08_oot_gtr:,.0f}")

    # ========================================
    # 전략 25: CB+LGB × log/raw → 4모델 블렌딩 (Scale)
    # ========================================
    print(f"\n{'─' * 50}")
    print(f"[전략 25] 4모델 Scale Blending (PL 없음)")
    print(f"{'─' * 50}")

    oof_25, vpred_25, _ = train_models(
        train_cb, val_cb, train_lgb, val_lgb, kf, label="25")

    # 가중 평균 (raw 60%)
    s25_oof_arr = (0.20*oof_25['cb_log'] + 0.30*oof_25['cb_raw'] +
                   0.20*oof_25['lgb_log'] + 0.30*oof_25['lgb_raw'])
    s25_val_arr = (0.20*vpred_25['cb_log'] + 0.30*vpred_25['cb_raw'] +
                   0.20*vpred_25['lgb_log'] + 0.30*vpred_25['lgb_raw'])

    s25_oof_rmse = calc_rmse(s25_oof_arr, y_oot_train)
    s25_oot_raw = calc_rmse(s25_val_arr, y_val_true)
    s25_oot_gtr = calc_rmse(s25_val_arr * trend_correction, y_val_true)
    print(f"  OOF: {s25_oof_rmse:,.0f} | OOT(raw): {s25_oot_raw:,.0f} | OOT(+GTR): {s25_oot_gtr:,.0f}")

    # ========================================
    # 전략 26: PL2 + 4모델 Scale Blending
    # ========================================
    print(f"\n{'─' * 50}")
    print(f"[전략 26] PL2 + 4모델 Scale Blending")
    print(f"{'─' * 50}")

    # Stage 1: OOT-val에 대한 pseudo label 생성
    print("  Stage 1: pseudo label 생성...")
    oof_s1, vpred_s1, fold_preds_s1 = train_models(
        train_cb, val_cb, train_lgb, val_lgb, kf,
        return_fold_preds=True, label="S1")

    pseudo_labels = np.mean([vpred_s1[k] for k in MODELS], axis=0)
    model_means = np.array([vpred_s1[k] for k in MODELS])
    model_disagreement = np.std(model_means, axis=0) / np.mean(model_means, axis=0)
    fold_cvs = []
    for k in MODELS:
        folds_arr = np.array(fold_preds_s1[k])
        fold_cvs.append(np.std(folds_arr, axis=0) / np.mean(folds_arr, axis=0))
    fold_cv = np.mean(fold_cvs, axis=0)
    confidence = 1 - (model_disagreement + fold_cv) / 2

    threshold = np.percentile(confidence, 50)
    mask = confidence >= threshold
    n_pseudo = mask.sum()
    print(f"  PL 상위 50%: {n_pseudo}건 추가")

    # OOT-val의 고신뢰 샘플을 OOT-train에 추가
    val_selected = oot_val[mask].copy()
    val_selected['Target'] = pseudo_labels[mask]
    train_aug = pd.concat([oot_train, val_selected], ignore_index=True)

    # Stage 2: 증강 데이터로 재학습
    print("  Stage 2: PL2 증강 데이터 학습...")
    kf2 = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    train_cb2, val_cb2, train_lgb2, val_lgb2 = prepare_data(train_aug, oot_val)

    oof_26, vpred_26, fold_preds_26 = train_models(
        train_cb2, val_cb2, train_lgb2, val_lgb2, kf2,
        return_fold_preds=True, label="26")

    for k in MODELS:
        oof_26[k] = oof_26[k][:n_oot_train]

    s26_oof_arr = (0.20*oof_26['cb_log'] + 0.30*oof_26['cb_raw'] +
                   0.20*oof_26['lgb_log'] + 0.30*oof_26['lgb_raw'])
    s26_val_arr = (0.20*vpred_26['cb_log'] + 0.30*vpred_26['cb_raw'] +
                   0.20*vpred_26['lgb_log'] + 0.30*vpred_26['lgb_raw'])

    s26_oof_rmse = calc_rmse(s26_oof_arr, y_oot_train)
    s26_oot_raw = calc_rmse(s26_val_arr, y_val_true)
    s26_oot_gtr = calc_rmse(s26_val_arr * trend_correction, y_val_true)
    print(f"  OOF: {s26_oof_rmse:,.0f} | OOT(raw): {s26_oot_raw:,.0f} | OOT(+GTR): {s26_oot_gtr:,.0f}")

    # ========================================
    # 전략 28: PL2 + 기존4 + 평당가4 → 8모델 Ridge
    # ========================================
    print(f"\n{'─' * 50}")
    print(f"[전략 28] PL2 + 8모델 (기존+평당가) Ridge")
    print(f"{'─' * 50}")

    area_train = train_aug['Exclusive_Area'].values
    area_val = oot_val['Exclusive_Area'].values
    area_oot_train = oot_train['Exclusive_Area'].values

    # 평당가 4모델
    print("  평당가 4모델 학습...")
    y_up_log = np.log1p(train_aug['Target'].values / area_train)
    y_up_raw = (train_aug['Target'].values / area_train).astype(float)

    oof_unit, vpred_unit, _ = train_models(
        train_cb2, val_cb2, train_lgb2, val_lgb2, kf2,
        y_log_override=y_up_log, y_raw_override=y_up_raw, label="평당가")

    for k in MODELS:
        oof_unit[k] = oof_unit[k][:n_oot_train] * area_oot_train
        vpred_unit[k] = vpred_unit[k] * area_val

    # 8모델 Ridge 스태킹
    stack_train_28 = np.column_stack(
        [oof_26[k] for k in MODELS] + [oof_unit[k] for k in MODELS])
    stack_val_28 = np.column_stack(
        [vpred_26[k] for k in MODELS] + [vpred_unit[k] for k in MODELS])

    best_rmse_28 = float('inf')
    best_alpha_28 = None
    best_val_28 = None
    best_oof_28 = None

    for alpha in [0.1, 0.5, 1.0, 5.0, 10.0]:
        s_oof = np.zeros(n_oot_train)
        s_val = np.zeros(len(oot_val))
        for fold, (tr, va) in enumerate(kf.split(stack_train_28)):
            meta = Ridge(alpha=alpha)
            meta.fit(stack_train_28[tr], y_oot_train[tr])
            s_oof[va] = meta.predict(stack_train_28[va])
            s_val += meta.predict(stack_val_28) / N_SPLITS
        r = calc_rmse(s_oof, y_oot_train)
        if r < best_rmse_28:
            best_rmse_28 = r
            best_alpha_28 = alpha
            best_val_28 = s_val.copy()
            best_oof_28 = s_oof.copy()

    s28_oof_rmse = best_rmse_28
    s28_oot_raw = calc_rmse(best_val_28, y_val_true)
    s28_oot_gtr = calc_rmse(best_val_28 * trend_correction, y_val_true)
    print(f"  Ridge α={best_alpha_28}")
    print(f"  OOF: {s28_oof_rmse:,.0f} | OOT(raw): {s28_oot_raw:,.0f} | OOT(+GTR): {s28_oot_gtr:,.0f}")

    # ========================================
    # 전체 비교 테이블
    # ========================================
    print(f"\n{'=' * 70}")
    print(f"전략 비교 (OOT holdout={holdout}개월, cutoff={cutoff_ym})")
    print(f"{'=' * 70}")
    print(f"  {'전략':20s} {'OOF':>8s} {'OOT(raw)':>10s} {'OOT(+GTR)':>10s} │ {'실제OOF':>8s} {'Public':>8s}")
    print(f"  {'─'*20} {'─'*8} {'─'*10} {'─'*10} │ {'─'*8} {'─'*8}")

    strategies = [
        ("08 CB+LGB Ridge",     s08_oof_rmse, s08_oot_raw, s08_oot_gtr, 2234, 2155),
        ("25 Scale Blend",      s25_oof_rmse, s25_oot_raw, s25_oot_gtr, 2226, None),
        ("26 PL2+Scale",        s26_oof_rmse, s26_oot_raw, s26_oot_gtr, 2215, 2150),
        ("28 PL2+Unit Ridge",   s28_oof_rmse, s28_oot_raw, s28_oot_gtr, 2196, 2097),
    ]

    for name, oof, oot_r, oot_g, real_oof, public in strategies:
        pub_str = f"{public:,}" if public else "—"
        print(f"  {name:20s} {oof:>8,.0f} {oot_r:>10,.0f} {oot_g:>10,.0f} │ {real_oof:>8,} {pub_str:>8s}")

    # 구성요소별 기여도
    print(f"\n{'─' * 70}")
    print(f"구성요소별 OOT 기여도 (GTR 적용 기준)")
    print(f"{'─' * 70}")
    print(f"  Scale (08→25):   {s08_oot_gtr - s25_oot_gtr:+,.0f}  (raw 모델 추가)")
    print(f"  PL2   (25→26):   {s25_oot_gtr - s26_oot_gtr:+,.0f}  (Pseudo Label)")
    print(f"  평당가 (26→28):  {s26_oot_gtr - s28_oot_gtr:+,.0f}  (Unit Price 4모델)")
    print(f"  GTR   (28 raw→gtr): {s28_oot_raw - s28_oot_gtr:+,.0f}  (구별 트렌드)")
    print(f"  전체  (08→28):   {s08_oot_gtr - s28_oot_gtr:+,.0f}  (누적)")

    # OOF vs OOT 상관 분석
    print(f"\n{'─' * 70}")
    print(f"검증 방식별 순위 비교")
    print(f"{'─' * 70}")

    oof_rank = sorted(range(4), key=lambda i: strategies[i][1])
    oot_rank = sorted(range(4), key=lambda i: strategies[i][3])
    public_vals = [s[5] for s in strategies]

    names = [s[0] for s in strategies]
    print(f"  OOF 순위:    {' < '.join(names[i] for i in oof_rank)}")
    print(f"  OOT 순위:    {' < '.join(names[i] for i in oot_rank)}")

    has_public = [i for i in range(4) if strategies[i][5] is not None]
    if has_public:
        public_rank = sorted(has_public, key=lambda i: strategies[i][5])
        print(f"  Public 순위: {' < '.join(names[i] for i in public_rank)}")

    # 구별 OOT 분석
    print(f"\n{'─' * 70}")
    print(f"구별 OOT RMSE (전략 28, GTR 적용)")
    print(f"{'─' * 70}")
    pred_28_gtr = best_val_28 * trend_correction
    for gu in sorted(oot_val['Gu'].unique()):
        gu_mask = oot_val['Gu'] == gu
        n_gu = gu_mask.sum()
        gu_rmse = calc_rmse(pred_28_gtr[gu_mask], y_val_true[gu_mask])
        print(f"  {gu:15s} ({n_gu:3d}건): {gu_rmse:,.0f}")
