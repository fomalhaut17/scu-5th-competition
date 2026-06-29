"""
59 MLP DIVERSITY
순수 PyTorch MLP를 기존 12모델 스택에 추가하여 다양성 기여 측정.
여러 MLP 변형: 다른 아키텍처/dropout/타겟으로 다양성 극대화.
"""
import os, sys, numpy as np, pandas as pd
os.environ['PYTHONUNBUFFERED'] = '1'
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb
import torch, torch.nn as nn
import warnings; warnings.filterwarnings('ignore')

_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
train_orig = pd.read_csv(f'{_DIR}/seoul_real_estate_train.csv')
test_orig = pd.read_csv(f'{_DIR}/seoul_real_estate_test.csv')
y_true = train_orig['Target'].values
n_orig = len(train_orig)
area_train = train_orig['Exclusive_Area'].values
area_test = test_orig['Exclusive_Area'].values
N_SPLITS = 5; CAT_FEATURES = ['Gu', 'Dong']; SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)

def preprocess(df):
    df = df.copy()
    df['Distance_to_Subway'] = df['Distance_to_Subway'].fillna(df['Distance_to_Subway'].median())
    df['Age'] = 2026 - df['Year_Built']
    df['Year'] = df['Transaction_YearMonth'] // 100
    df['Month'] = df['Transaction_YearMonth'] % 100
    df = df.drop(columns=['ID', 'Transaction_YearMonth', 'Year_Built'])
    df['YearMonth_Seq'] = (df['Year'] - 2024) * 12 + df['Month']
    df['Area_x_Floor'] = df['Exclusive_Area'] * df['Floor']
    df['Floor_per_Area'] = df['Floor'] / df['Exclusive_Area']
    df['Brand_x_Area'] = df['Brand_Apartment'] * df['Exclusive_Area']
    return df

def encode(tr, te, as_cat=False):
    tr, te = tr.copy(), te.copy()
    for c in CAT_FEATURES:
        le = LabelEncoder()
        le.fit(list(tr[c].astype(str)) + list(te[c].astype(str)))
        tr[c] = le.transform(tr[c].astype(str))
        te[c] = le.transform(te[c].astype(str))
        if as_cat: tr[c] = tr[c].astype('category'); te[c] = te[c].astype('category')
    return tr, te

tr_p, te_p = preprocess(train_orig), preprocess(test_orig)
tr_cb, te_cb = encode(tr_p, te_p)
tr_lgb, te_lgb = encode(tr_p, te_p, as_cat=True)
X_cb = tr_cb.drop(columns=['Target']); X_te_cb = te_cb
cat_idx = [X_cb.columns.get_loc(c) for c in CAT_FEATURES]
X_lgb = tr_lgb.drop(columns=['Target']); X_te_lgb = te_lgb
y_log = np.log1p(y_true); y_raw = y_true.astype(float)
y_up_log = np.log1p(y_true / area_train); y_up_raw = (y_true / area_train).astype(float)

CB_P = {'learning_rate':0.0101,'depth':3,'l2_leaf_reg':4.944,'bagging_temperature':1.482,'random_strength':0.469,'min_data_in_leaf':46}
LGB_P = {'learning_rate':0.0230,'num_leaves':110,'max_depth':3,'min_child_samples':27,'subsample':0.931,'colsample_bytree':0.823,'reg_alpha':0.012,'reg_lambda':0.047}
LGB_ET_P = {'extra_trees':True,'max_depth':3,'num_leaves':31,'feature_fraction':0.6,'bagging_fraction':0.7,'bagging_freq':1,'min_child_samples':30,'learning_rate':0.02}

# === MLP ===
class MLP(nn.Module):
    def __init__(self, d_in, layers, drop=0.3):
        super().__init__()
        parts = []
        for h in layers:
            parts += [nn.Linear(d_in, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(drop)]
            d_in = h
        parts.append(nn.Linear(d_in, 1))
        self.net = nn.Sequential(*parts)
    def forward(self, x): return self.net(x).squeeze(-1)

def train_mlp(X_tr, y_tr, X_va, y_va, X_te, layers=[128,64,32], drop=0.3, lr=1e-3, wd=0.01, epochs=500, pat=50, seed=42):
    torch.manual_seed(seed)
    y_mean, y_std = y_tr.mean(), y_tr.std() + 1e-8
    y_tr_n = (y_tr - y_mean) / y_std
    y_va_n = (y_va - y_mean) / y_std
    m = MLP(X_tr.shape[1], layers, drop)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    Xt, yt = torch.FloatTensor(X_tr), torch.FloatTensor(y_tr_n)
    Xv, yv = torch.FloatTensor(X_va), torch.FloatTensor(y_va_n)
    Xte = torch.FloatTensor(X_te)
    best_loss, best_st, wait = 1e9, None, 0
    for ep in range(epochs):
        m.train(); opt.zero_grad()
        loss = nn.MSELoss()(m(Xt), yt); loss.backward(); opt.step(); sch.step()
        m.eval()
        with torch.no_grad(): vl = nn.MSELoss()(m(Xv), yv).item()
        if vl < best_loss: best_loss = vl; best_st = {k:v.clone() for k,v in m.state_dict().items()}; wait = 0
        else:
            wait += 1
            if wait >= pat: break
    m.load_state_dict(best_st); m.eval()
    with torch.no_grad():
        pv = m(Xv).numpy() * y_std + y_mean
        pte = m(Xte).numpy() * y_std + y_mean
    return pv, pte

# =============================================
# 학습
# =============================================
print("=" * 60)
print("12 GBDT + 6 MLP 학습")
print("=" * 60, flush=True)

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
oof, tpred = {}, {}
n_te = len(X_te_cb)

for fold, (tr, va) in enumerate(kf.split(X_cb)):
    print(f"\n  Fold {fold+1}/{N_SPLITS}", flush=True)

    # GBDT 12모델
    for nm, y, xf in [('cb_log',y_log,'log'),('cb_raw',y_raw,'raw'),('cb_up_log',y_up_log,'log'),('cb_up_raw',y_up_raw,'raw')]:
        m = CatBoostRegressor(loss_function='RMSE',random_seed=SEED,verbose=0,iterations=3000,early_stopping_rounds=100,**CB_P)
        m.fit(X_cb.iloc[tr],y[tr],eval_set=(X_cb.iloc[va],y[va]),cat_features=cat_idx)
        pv = np.expm1(m.predict(X_cb.iloc[va])) if xf=='log' else m.predict(X_cb.iloc[va])
        pt = np.expm1(m.predict(X_te_cb)) if xf=='log' else m.predict(X_te_cb)
        oof.setdefault(nm, np.zeros(n_orig))[va] = pv
        tpred[nm] = tpred.get(nm, np.zeros(n_te)) + pt/N_SPLITS

    for nm, y, xf in [('lgb_log',y_log,'log'),('lgb_raw',y_raw,'raw'),('lgb_up_log',y_up_log,'log'),('lgb_up_raw',y_up_raw,'raw')]:
        m = lgb.LGBMRegressor(objective='regression',metric='rmse',verbose=-1,random_state=SEED,n_estimators=3000,**LGB_P)
        m.fit(X_lgb.iloc[tr],y[tr],eval_set=[(X_lgb.iloc[va],y[va])],callbacks=[lgb.early_stopping(100,verbose=False)])
        pv = np.expm1(m.predict(X_lgb.iloc[va])) if xf=='log' else m.predict(X_lgb.iloc[va])
        pt = np.expm1(m.predict(X_te_lgb)) if xf=='log' else m.predict(X_te_lgb)
        oof.setdefault(nm, np.zeros(n_orig))[va] = pv
        tpred[nm] = tpred.get(nm, np.zeros(n_te)) + pt/N_SPLITS

    sc = StandardScaler()
    Xstr = sc.fit_transform(X_cb.iloc[tr]); Xsva = sc.transform(X_cb.iloc[va]); Xste = sc.transform(X_te_cb)

    for nm, y, xf in [('et_log',y_log,'log'),('et_raw',y_raw,'raw')]:
        m = ExtraTreesRegressor(n_estimators=500,max_depth=12,min_samples_leaf=10,random_state=SEED,n_jobs=1)
        m.fit(Xstr, y[tr])
        pv = np.expm1(m.predict(Xsva)) if xf=='log' else m.predict(Xsva)
        pt = np.expm1(m.predict(Xste)) if xf=='log' else m.predict(Xste)
        oof.setdefault(nm, np.zeros(n_orig))[va] = pv
        tpred[nm] = tpred.get(nm, np.zeros(n_te)) + pt/N_SPLITS

    for nm, y, xf in [('lgbet_log',y_log,'log'),('lgbet_raw',y_raw,'raw')]:
        m = lgb.LGBMRegressor(objective='regression',metric='rmse',verbose=-1,random_state=SEED,n_estimators=3000,**LGB_ET_P)
        m.fit(X_lgb.iloc[tr],y[tr],eval_set=[(X_lgb.iloc[va],y[va])],callbacks=[lgb.early_stopping(100,verbose=False)])
        pv = np.expm1(m.predict(X_lgb.iloc[va])) if xf=='log' else m.predict(X_lgb.iloc[va])
        pt = np.expm1(m.predict(X_te_lgb)) if xf=='log' else m.predict(X_te_lgb)
        oof.setdefault(nm, np.zeros(n_orig))[va] = pv
        tpred[nm] = tpred.get(nm, np.zeros(n_te)) + pt/N_SPLITS

    # MLP 6개 변형 (다른 아키텍처/dropout/타겟)
    print(f"    → MLP 학습 중...", flush=True)

    # MLP-A: 기본 (128-64-32, drop=0.3, log)
    pv, pt = train_mlp(Xstr, y_log[tr], Xsva, y_log[va], Xste, [128,64,32], 0.3)
    oof.setdefault('mlp_a_log', np.zeros(n_orig))[va] = np.expm1(pv)
    tpred['mlp_a_log'] = tpred.get('mlp_a_log', np.zeros(n_te)) + np.expm1(pt)/N_SPLITS

    # MLP-A: raw
    pv, pt = train_mlp(Xstr, y_raw[tr], Xsva, y_raw[va], Xste, [128,64,32], 0.3)
    oof.setdefault('mlp_a_raw', np.zeros(n_orig))[va] = pv
    tpred['mlp_a_raw'] = tpred.get('mlp_a_raw', np.zeros(n_te)) + pt/N_SPLITS

    # MLP-B: 넓고 얕은 (256-128, drop=0.4, log)
    pv, pt = train_mlp(Xstr, y_log[tr], Xsva, y_log[va], Xste, [256,128], 0.4, seed=123)
    oof.setdefault('mlp_b_log', np.zeros(n_orig))[va] = np.expm1(pv)
    tpred['mlp_b_log'] = tpred.get('mlp_b_log', np.zeros(n_te)) + np.expm1(pt)/N_SPLITS

    # MLP-B: raw
    pv, pt = train_mlp(Xstr, y_raw[tr], Xsva, y_raw[va], Xste, [256,128], 0.4, seed=123)
    oof.setdefault('mlp_b_raw', np.zeros(n_orig))[va] = pv
    tpred['mlp_b_raw'] = tpred.get('mlp_b_raw', np.zeros(n_te)) + pt/N_SPLITS

    # MLP-C: 깊고 좁은 (64-64-64-64, drop=0.5, log) — 강한 정규화
    pv, pt = train_mlp(Xstr, y_log[tr], Xsva, y_log[va], Xste, [64,64,64,64], 0.5, seed=456)
    oof.setdefault('mlp_c_log', np.zeros(n_orig))[va] = np.expm1(pv)
    tpred['mlp_c_log'] = tpred.get('mlp_c_log', np.zeros(n_te)) + np.expm1(pt)/N_SPLITS

    # MLP-C: raw
    pv, pt = train_mlp(Xstr, y_raw[tr], Xsva, y_raw[va], Xste, [64,64,64,64], 0.5, seed=456)
    oof.setdefault('mlp_c_raw', np.zeros(n_orig))[va] = pv
    tpred['mlp_c_raw'] = tpred.get('mlp_c_raw', np.zeros(n_te)) + pt/N_SPLITS

# unit_price 환산
for k in ['cb_up_log','cb_up_raw','lgb_up_log','lgb_up_raw']:
    oof[k] *= area_train; tpred[k] *= area_test

# =============================================
# 분석
# =============================================
print(f"\n{'='*60}")
print("개별 모델 OOF RMSE")
print("="*60, flush=True)
for nm in sorted(oof):
    rmse = np.sqrt(np.mean((oof[nm] - y_true)**2))
    tag = " ★MLP" if 'mlp' in nm else ""
    print(f"  {nm:15s}: {rmse:,.0f}{tag}")

print(f"\n{'='*60}")
print("MLP 상관관계 (GBDT 대비)")
print("="*60, flush=True)
mlp_names = [k for k in oof if 'mlp' in k]
for mn in mlp_names:
    corrs = {r: np.corrcoef(oof[r], oof[mn])[0,1] for r in ['cb_log','lgb_raw','et_log']}
    print(f"  {mn:15s} → " + ", ".join(f"{k}:{v:.4f}" for k,v in corrs.items()))

print(f"\n{'='*60}")
print("Ridge 스태킹 비교")
print("="*60, flush=True)

def ridge_stack(od, td, yt, names, label=""):
    n, nt = len(yt), len(list(td.values())[0])
    st = np.column_stack([od[k] for k in names])
    ste = np.column_stack([td[k] for k in names])
    best = (1e9, None, None)
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    for a in [0.1,0.5,1,5,10,50,100,200,500]:
        so, sp = np.zeros(n), np.zeros(nt)
        for t,v in kf.split(st):
            m = Ridge(alpha=a); m.fit(st[t],yt[t])
            so[v] = m.predict(st[v]); sp += m.predict(ste)/N_SPLITS
        r = np.sqrt(np.mean((so-yt)**2))
        if r < best[0]: best = (r, a, sp.copy())
    print(f"  [{label}] {len(names)}모델 Ridge(α={best[1]}) → OOF: {best[0]:,.1f}")
    return best[2], best[0]

B12 = ['cb_log','cb_raw','lgb_log','lgb_raw','cb_up_log','cb_up_raw','lgb_up_log','lgb_up_raw','et_log','et_raw','lgbet_log','lgbet_raw']
MLP_A = ['mlp_a_log','mlp_a_raw']
MLP_B = ['mlp_b_log','mlp_b_raw']
MLP_C = ['mlp_c_log','mlp_c_raw']
MLP_ALL = MLP_A + MLP_B + MLP_C

_, r12 = ridge_stack(oof, tpred, y_true, B12, "기존 12모델")
_, r12a = ridge_stack(oof, tpred, y_true, B12+MLP_A, "+MLP-A(128-64-32)")
_, r12b = ridge_stack(oof, tpred, y_true, B12+MLP_B, "+MLP-B(256-128)")
_, r12c = ridge_stack(oof, tpred, y_true, B12+MLP_C, "+MLP-C(64×4)")
_, r12all = ridge_stack(oof, tpred, y_true, B12+MLP_ALL, "+MLP 전체(6개)")

# 최적 MLP 조합 찾기
best_mlp = min([(r12a,'A'),(r12b,'B'),(r12c,'C')], key=lambda x: x[0])
print(f"\n  최선 MLP: {best_mlp[1]} ({best_mlp[0]:,.1f})")
print(f"\n  12 → +MLP-A: {r12-r12a:+.1f}")
print(f"  12 → +MLP-B: {r12-r12b:+.1f}")
print(f"  12 → +MLP-C: {r12-r12c:+.1f}")
print(f"  12 → +MLP전체: {r12-r12all:+.1f}")

print(f"\n{'='*60}")
print("최종 요약")
print("="*60)
print(f"  기존 12모델:   {r12:,.1f}")
print(f"  +MLP-A(2):     {r12a:,.1f} ({r12-r12a:+.1f})")
print(f"  +MLP-B(2):     {r12b:,.1f} ({r12-r12b:+.1f})")
print(f"  +MLP-C(2):     {r12c:,.1f} ({r12-r12c:+.1f})")
print(f"  +MLP 전체(6):  {r12all:,.1f} ({r12-r12all:+.1f})")
print(f"\n  참고: ET 추가 시 -4점 (전략44→45)")
print(f"  참고: 전략53 no-PL2 OOF = 2,229", flush=True)
