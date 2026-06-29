"""
59 NEURAL NETWORK DIVERSITY
TabNet + MLP를 기존 12모델 스택에 추가하여 다양성 기여 측정.
목표: 단독 성능이 아니라, 기존 GBDT와 낮은 상관 → 스태킹 개선.
ET(상관 0.96)가 성공한 것처럼, NN도 구조적으로 다른 예측을 만들면 된다.
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
import torch
import torch.nn as nn
from pytorch_tabnet.tab_model import TabNetRegressor
import sys
import warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONUNBUFFERED'] = '1'

_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
INPUT_DIR = _DIR
train_orig = pd.read_csv(f'{INPUT_DIR}/seoul_real_estate_train.csv')
test_orig = pd.read_csv(f'{INPUT_DIR}/seoul_real_estate_test.csv')
y_true = train_orig['Target'].values
n_orig = len(train_orig)
area_train = train_orig['Exclusive_Area'].values
area_test = test_orig['Exclusive_Area'].values

N_SPLITS = 5
CAT_FEATURES = ['Gu', 'Dong']
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

def base_preprocess(df):
    df = df.copy()
    df['Distance_to_Subway'] = df['Distance_to_Subway'].fillna(df['Distance_to_Subway'].median())
    df['Age'] = 2026 - df['Year_Built']
    df['Year'] = df['Transaction_YearMonth'] // 100
    df['Month'] = df['Transaction_YearMonth'] % 100
    df = df.drop(columns=['ID', 'Transaction_YearMonth', 'Year_Built'])
    return df

def add_fe(df):
    df = df.copy()
    df['YearMonth_Seq'] = (df['Year'] - 2024) * 12 + df['Month']
    df['Area_x_Floor'] = df['Exclusive_Area'] * df['Floor']
    df['Floor_per_Area'] = df['Floor'] / df['Exclusive_Area']
    df['Brand_x_Area'] = df['Brand_Apartment'] * df['Exclusive_Area']
    return df

def encode_cat(train_df, test_df, as_category=False):
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
    train_p = add_fe(base_preprocess(train_df))
    test_p = add_fe(base_preprocess(test_df))
    tr_cb, te_cb = encode_cat(train_p, test_p, as_category=False)
    tr_lgb, te_lgb = encode_cat(train_p, test_p, as_category=True)
    return tr_cb, te_cb, tr_lgb, te_lgb

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

tr_cb, te_cb, tr_lgb, te_lgb = prepare_data(train_orig, test_orig)
X_cb = tr_cb.drop(columns=['Target'])
X_te_cb = te_cb
cat_idx = [X_cb.columns.get_loc(c) for c in CAT_FEATURES]
X_lgb = tr_lgb.drop(columns=['Target'])
X_te_lgb = te_lgb

y_log = np.log1p(y_true)
y_raw = y_true.copy().astype(float)
y_up_log = np.log1p(y_true / area_train)
y_up_raw = (y_true / area_train).astype(float)


# === MLP 모델 정의 ===
class TabularMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(X_tr, y_tr, X_va, y_va, X_te, seed=42,
              hidden_dims=[128, 64, 32], dropout=0.3, lr=0.001,
              weight_decay=0.01, epochs=500, patience=50):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = TabularMLP(X_tr.shape[1], hidden_dims, dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    X_tr_t = torch.FloatTensor(X_tr)
    y_tr_t = torch.FloatTensor(y_tr)
    X_va_t = torch.FloatTensor(X_va)
    y_va_t = torch.FloatTensor(y_va)
    X_te_t = torch.FloatTensor(X_te)

    best_loss = float('inf')
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_tr_t)
        loss = nn.MSELoss()(pred, y_tr_t)
        loss.backward()
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            va_pred = model(X_va_t)
            va_loss = nn.MSELoss()(va_pred, y_va_t).item()

        if va_loss < best_loss:
            best_loss = va_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        p_va = model(X_va_t).numpy()
        p_te = model(X_te_t).numpy()
    return p_va, p_te


# =============================================
# PHASE 1: 기존 12모델 학습 (baseline)
# =============================================
print("=" * 60)
print("PHASE 1: 기존 12모델 + NN 모델 학습")
print("=" * 60)

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
oof = {}
tpred = {}
n_test = len(X_te_cb)

for fold, (tr, va) in enumerate(kf.split(X_cb)):
    print(f"\n  Fold {fold+1}/{N_SPLITS}", flush=True)

    # --- 기존 GBDT 12모델 ---
    for name, y, xform, mtype in [
        ('cb_log', y_log, 'log', 'cb'), ('cb_raw', y_raw, 'raw', 'cb'),
        ('cb_up_log', y_up_log, 'log', 'cb'), ('cb_up_raw', y_up_raw, 'raw', 'cb'),
    ]:
        m = CatBoostRegressor(loss_function='RMSE', random_seed=SEED, verbose=0,
                              iterations=3000, early_stopping_rounds=100, **CB_PARAMS)
        m.fit(X_cb.iloc[tr], y[tr], eval_set=(X_cb.iloc[va], y[va]), cat_features=cat_idx)
        p_va = np.expm1(m.predict(X_cb.iloc[va])) if xform == 'log' else m.predict(X_cb.iloc[va])
        p_te = np.expm1(m.predict(X_te_cb)) if xform == 'log' else m.predict(X_te_cb)
        oof.setdefault(name, np.zeros(n_orig))[va] = p_va
        tpred[name] = tpred.get(name, np.zeros(n_test)) + p_te / N_SPLITS

    for name, y, xform in [
        ('lgb_log', y_log, 'log'), ('lgb_raw', y_raw, 'raw'),
        ('lgb_up_log', y_up_log, 'log'), ('lgb_up_raw', y_up_raw, 'raw'),
    ]:
        m = lgb.LGBMRegressor(objective='regression', metric='rmse', verbose=-1,
                               random_state=SEED, n_estimators=3000, **LGB_PARAMS)
        m.fit(X_lgb.iloc[tr], y[tr], eval_set=[(X_lgb.iloc[va], y[va])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        p_va = np.expm1(m.predict(X_lgb.iloc[va])) if xform == 'log' else m.predict(X_lgb.iloc[va])
        p_te = np.expm1(m.predict(X_te_lgb)) if xform == 'log' else m.predict(X_te_lgb)
        oof.setdefault(name, np.zeros(n_orig))[va] = p_va
        tpred[name] = tpred.get(name, np.zeros(n_test)) + p_te / N_SPLITS

    scaler = StandardScaler()
    X_sc_tr = scaler.fit_transform(X_cb.iloc[tr])
    X_sc_va = scaler.transform(X_cb.iloc[va])
    X_sc_te = scaler.transform(X_te_cb)

    for name, y, xform in [('et_log', y_log, 'log'), ('et_raw', y_raw, 'raw')]:
        m = ExtraTreesRegressor(n_estimators=500, max_depth=12, min_samples_leaf=10,
                                 random_state=SEED, n_jobs=1)
        m.fit(X_sc_tr, y[tr])
        p_va = np.expm1(m.predict(X_sc_va)) if xform == 'log' else m.predict(X_sc_va)
        p_te = np.expm1(m.predict(X_sc_te)) if xform == 'log' else m.predict(X_sc_te)
        oof.setdefault(name, np.zeros(n_orig))[va] = p_va
        tpred[name] = tpred.get(name, np.zeros(n_test)) + p_te / N_SPLITS

    for name, y, xform in [('lgbet_log', y_log, 'log'), ('lgbet_raw', y_raw, 'raw')]:
        m = lgb.LGBMRegressor(objective='regression', metric='rmse', verbose=-1,
                               random_state=SEED, n_estimators=3000, **LGB_ET_PARAMS)
        m.fit(X_lgb.iloc[tr], y[tr], eval_set=[(X_lgb.iloc[va], y[va])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        p_va = np.expm1(m.predict(X_lgb.iloc[va])) if xform == 'log' else m.predict(X_lgb.iloc[va])
        p_te = np.expm1(m.predict(X_te_lgb)) if xform == 'log' else m.predict(X_te_lgb)
        oof.setdefault(name, np.zeros(n_orig))[va] = p_va
        tpred[name] = tpred.get(name, np.zeros(n_test)) + p_te / N_SPLITS

    # --- NN 모델 (scaled features 사용) ---
    print(f"    → TabNet 학습 중...", flush=True)

    # TabNet log
    tabnet = TabNetRegressor(
        n_d=8, n_a=8, n_steps=3, gamma=1.5,
        lambda_sparse=0.01, optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=0.02, weight_decay=0.01),
        scheduler_fn=torch.optim.lr_scheduler.CosineAnnealingLR,
        scheduler_params=dict(T_max=200),
        mask_type='entmax', seed=SEED, verbose=0,
    )
    tabnet.fit(
        X_sc_tr, y_log[tr].reshape(-1, 1),
        eval_set=[(X_sc_va, y_log[va].reshape(-1, 1))],
        eval_metric=['rmse'], max_epochs=500, patience=50,
        batch_size=256, virtual_batch_size=128, num_workers=0, drop_last=False,
    )
    p_va = np.expm1(tabnet.predict(X_sc_va).flatten())
    p_te = np.expm1(tabnet.predict(X_sc_te).flatten())
    oof.setdefault('tabnet_log', np.zeros(n_orig))[va] = p_va
    tpred['tabnet_log'] = tpred.get('tabnet_log', np.zeros(n_test)) + p_te / N_SPLITS

    # TabNet raw
    tabnet2 = TabNetRegressor(
        n_d=8, n_a=8, n_steps=3, gamma=1.5,
        lambda_sparse=0.01, optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=0.02, weight_decay=0.01),
        scheduler_fn=torch.optim.lr_scheduler.CosineAnnealingLR,
        scheduler_params=dict(T_max=200),
        mask_type='entmax', seed=SEED, verbose=0,
    )
    tabnet2.fit(
        X_sc_tr, y_raw[tr].reshape(-1, 1),
        eval_set=[(X_sc_va, y_raw[va].reshape(-1, 1))],
        eval_metric=['rmse'], max_epochs=500, patience=50,
        batch_size=256, virtual_batch_size=128, num_workers=0, drop_last=False,
    )
    p_va = tabnet2.predict(X_sc_va).flatten()
    p_te = tabnet2.predict(X_sc_te).flatten()
    oof.setdefault('tabnet_raw', np.zeros(n_orig))[va] = p_va
    tpred['tabnet_raw'] = tpred.get('tabnet_raw', np.zeros(n_test)) + p_te / N_SPLITS

    # MLP log (3-layer, heavy regularization)
    print(f"    → MLP 학습 중...", flush=True)
    p_va, p_te = train_mlp(X_sc_tr, y_log[tr], X_sc_va, y_log[va], X_sc_te,
                            seed=SEED, hidden_dims=[128, 64, 32], dropout=0.4,
                            lr=0.001, weight_decay=0.01, epochs=500, patience=50)
    oof.setdefault('mlp_log', np.zeros(n_orig))[va] = np.expm1(p_va)
    tpred['mlp_log'] = tpred.get('mlp_log', np.zeros(n_test)) + np.expm1(p_te) / N_SPLITS

    # MLP raw
    p_va, p_te = train_mlp(X_sc_tr, y_raw[tr], X_sc_va, y_raw[va], X_sc_te,
                            seed=SEED, hidden_dims=[128, 64, 32], dropout=0.4,
                            lr=0.001, weight_decay=0.01, epochs=500, patience=50)
    oof.setdefault('mlp_raw', np.zeros(n_orig))[va] = p_va
    tpred['mlp_raw'] = tpred.get('mlp_raw', np.zeros(n_test)) + p_te / N_SPLITS

# unit_price 환산
UP_MODELS = ['cb_up_log', 'cb_up_raw', 'lgb_up_log', 'lgb_up_raw']
for k in UP_MODELS:
    oof[k] = oof[k] * area_train
    tpred[k] = tpred[k] * area_test


# =============================================
# PHASE 2: 분석
# =============================================
print(f"\n{'=' * 60}")
print("PHASE 2: 개별 모델 분석")
print("=" * 60)

print("\n=== 개별 모델 OOF RMSE ===")
for name in sorted(oof.keys()):
    rmse = np.sqrt(np.mean((oof[name] - y_true) ** 2))
    tag = " ★NN" if any(t in name for t in ['tabnet', 'mlp']) else ""
    print(f"  {name:15s}: {rmse:,.0f}{tag}")

print("\n=== NN 모델 상관관계 (기존 모델 대비) ===")
nn_models = ['tabnet_log', 'tabnet_raw', 'mlp_log', 'mlp_raw']
ref_models = ['cb_log', 'cb_raw', 'lgb_raw', 'et_log']
for nn_name in nn_models:
    corrs = []
    for ref_name in ref_models:
        corr = np.corrcoef(oof[ref_name], oof[nn_name])[0, 1]
        corrs.append(f"{ref_name}:{corr:.4f}")
    print(f"  {nn_name:15s} → {', '.join(corrs)}")


# =============================================
# PHASE 3: Ridge 스태킹 비교
# =============================================
print(f"\n{'=' * 60}")
print("PHASE 3: Ridge 스태킹 비교")
print("=" * 60)

def ridge_stack(oof_dict, tpred_dict, y_true, model_names, label=""):
    n = len(y_true)
    n_test = len(list(tpred_dict.values())[0])
    st_tr = np.column_stack([oof_dict[k] for k in model_names])
    st_te = np.column_stack([tpred_dict[k] for k in model_names])
    best_rmse = float('inf')
    best_alpha = None
    best_test = None
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    for alpha in [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 200.0, 500.0]:
        s_oof = np.zeros(n)
        s_test = np.zeros(n_test)
        for tr, va in kf.split(st_tr):
            meta = Ridge(alpha=alpha)
            meta.fit(st_tr[tr], y_true[tr])
            s_oof[va] = meta.predict(st_tr[va])
            s_test += meta.predict(st_te) / N_SPLITS
        rmse = np.sqrt(np.mean((s_oof - y_true) ** 2))
        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = alpha
            best_test = s_test.copy()
    print(f"  [{label}] {len(model_names)}모델 Ridge(α={best_alpha}) → OOF RMSE: {best_rmse:,.1f}")
    return best_test, best_rmse

BASE_12 = ['cb_log', 'cb_raw', 'lgb_log', 'lgb_raw',
           'cb_up_log', 'cb_up_raw', 'lgb_up_log', 'lgb_up_raw',
           'et_log', 'et_raw', 'lgbet_log', 'lgbet_raw']
TABNET_2 = ['tabnet_log', 'tabnet_raw']
MLP_2 = ['mlp_log', 'mlp_raw']

_, r12 = ridge_stack(oof, tpred, y_true, BASE_12, "기존 12모델")
_, r12t = ridge_stack(oof, tpred, y_true, BASE_12 + TABNET_2, "+TabNet = 14모델")
_, r12m = ridge_stack(oof, tpred, y_true, BASE_12 + MLP_2, "+MLP = 14모델")
_, r16 = ridge_stack(oof, tpred, y_true, BASE_12 + TABNET_2 + MLP_2, "+TabNet+MLP = 16모델")

# TabNet만 / MLP만도 시도
_, rt_only = ridge_stack(oof, tpred, y_true, TABNET_2, "TabNet only")
_, rm_only = ridge_stack(oof, tpred, y_true, MLP_2, "MLP only")
_, rnn_only = ridge_stack(oof, tpred, y_true, TABNET_2 + MLP_2, "NN only (4모델)")

print(f"\n  12 → +TabNet: {r12 - r12t:+.1f}")
print(f"  12 → +MLP: {r12 - r12m:+.1f}")
print(f"  12 → +All NN: {r12 - r16:+.1f}")

# =============================================
# 요약
# =============================================
print(f"\n{'=' * 60}")
print("=== 최종 요약 ===")
print("=" * 60)
print(f"  기존 12모델:     OOF {r12:,.1f}")
print(f"  +TabNet(2):      OOF {r12t:,.1f} ({r12 - r12t:+.1f})")
print(f"  +MLP(2):         OOF {r12m:,.1f} ({r12 - r12m:+.1f})")
print(f"  +TabNet+MLP(4):  OOF {r16:,.1f} ({r12 - r16:+.1f})")
print(f"\n  참고: 전략 47(12모델+PL2) = 2,191")
print(f"  참고: 전략 53(12모델 no-PL2) = 2,229")
print(f"  참고: ET 추가 시 -4점 개선 (전략 44→45)")
