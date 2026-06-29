"""
60 ENTITY EMBEDDING MLP
Gu/Dong을 학습 가능한 임베딩 벡터로 변환 → MLP 성능 개선.
목표: 단독 OOF 3,000~4,000 + GBDT 상관 <0.95 → 스태킹 기여.
"""
import os, sys, numpy as np, pandas as pd
os.environ['OMP_NUM_THREADS'] = '1'
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
N_SPLITS = 5; SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)

# === 전처리 (GBDT용) ===
CAT_FEATURES = ['Gu', 'Dong']
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

# === NN용 전처리 (Entity Embedding) ===
# 범주형: Gu, Dong → 정수 인덱스 (임베딩 입력)
# 수치형: 나머지 → StandardScaler

le_gu = LabelEncoder()
le_dong = LabelEncoder()
all_gu = list(train_orig['Gu'].astype(str)) + list(test_orig['Gu'].astype(str))
all_dong = list(train_orig['Dong'].astype(str)) + list(test_orig['Dong'].astype(str))
le_gu.fit(all_gu); le_dong.fit(all_dong)

n_gu = len(le_gu.classes_)
n_dong = len(le_dong.classes_)
print(f"Gu: {n_gu}개, Dong: {n_dong}개", flush=True)

# 임베딩 차원 (경험적 규칙: min(50, n_cat // 2))
emb_gu = min(50, (n_gu + 1) // 2)
emb_dong = min(50, (n_dong + 1) // 2)
print(f"Embedding dims: Gu={emb_gu}, Dong={emb_dong}", flush=True)

nn_tr = preprocess(train_orig)
nn_te = preprocess(test_orig)

gu_tr = le_gu.transform(nn_tr['Gu'].astype(str))
gu_te = le_gu.transform(nn_te['Gu'].astype(str))
dong_tr = le_dong.transform(nn_tr['Dong'].astype(str))
dong_te = le_dong.transform(nn_te['Dong'].astype(str))

NUM_COLS = [c for c in nn_tr.columns if c not in ['Gu', 'Dong', 'Target']]
X_num_tr = nn_tr[NUM_COLS].values.astype(np.float32)
X_num_te = nn_te[NUM_COLS].values.astype(np.float32)
n_num = len(NUM_COLS)
print(f"수치 피처: {n_num}개 {NUM_COLS}", flush=True)


# === Entity Embedding MLP ===
class EmbedMLP(nn.Module):
    def __init__(self, n_num, n_gu, emb_gu, n_dong, emb_dong, layers, drop=0.3):
        super().__init__()
        self.emb_gu = nn.Embedding(n_gu, emb_gu)
        self.emb_dong = nn.Embedding(n_dong, emb_dong)
        self.emb_drop = nn.Dropout(0.2)
        d_in = n_num + emb_gu + emb_dong
        parts = []
        for h in layers:
            parts += [nn.Linear(d_in, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(drop)]
            d_in = h
        parts.append(nn.Linear(d_in, 1))
        self.net = nn.Sequential(*parts)

    def forward(self, x_num, x_gu, x_dong):
        e_gu = self.emb_drop(self.emb_gu(x_gu))
        e_dong = self.emb_drop(self.emb_dong(x_dong))
        x = torch.cat([x_num, e_gu, e_dong], dim=1)
        return self.net(x).squeeze(-1)


def train_embed_mlp(X_num_tr, gu_tr, dong_tr, y_tr,
                    X_num_va, gu_va, dong_va, y_va,
                    X_num_te, gu_te, dong_te,
                    layers=[256, 128, 64], drop=0.3, lr=1e-3, wd=0.01,
                    epochs=800, pat=80, seed=42):
    torch.manual_seed(seed)
    y_mean, y_std = y_tr.mean(), y_tr.std() + 1e-8
    y_tr_n = (y_tr - y_mean) / y_std
    y_va_n = (y_va - y_mean) / y_std

    model = EmbedMLP(X_num_tr.shape[1], n_gu, emb_gu, n_dong, emb_dong, layers, drop)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=100, T_mult=2)

    Xn = torch.FloatTensor(X_num_tr); Gtr = torch.LongTensor(gu_tr); Dtr = torch.LongTensor(dong_tr)
    Yt = torch.FloatTensor(y_tr_n)
    Xnv = torch.FloatTensor(X_num_va); Gva = torch.LongTensor(gu_va); Dva = torch.LongTensor(dong_va)
    Yv = torch.FloatTensor(y_va_n)
    Xnte = torch.FloatTensor(X_num_te); Gte = torch.LongTensor(gu_te); Dte = torch.LongTensor(dong_te)

    best_loss, best_st, wait = 1e9, None, 0
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        loss = nn.MSELoss()(model(Xn, Gtr, Dtr), Yt)
        loss.backward(); opt.step(); sch.step()

        model.eval()
        with torch.no_grad():
            vl = nn.MSELoss()(model(Xnv, Gva, Dva), Yv).item()
        if vl < best_loss:
            best_loss = vl
            best_st = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= pat: break

    model.load_state_dict(best_st); model.eval()
    with torch.no_grad():
        pv = model(Xnv, Gva, Dva).numpy() * y_std + y_mean
        pte = model(Xnte, Gte, Dte).numpy() * y_std + y_mean
    return pv, pte


# =============================================
# 학습
# =============================================
print(f"\n{'='*60}")
print("12 GBDT + Embedding MLP 학습")
print("="*60, flush=True)

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

    # NN 수치 피처 스케일링
    sc_nn = StandardScaler()
    Xn_tr = sc_nn.fit_transform(X_num_tr[tr])
    Xn_va = sc_nn.transform(X_num_tr[va])
    Xn_te = sc_nn.transform(X_num_te)

    print(f"    → Embed-MLP 학습 중...", flush=True)

    # Embed-MLP A: 넓은 (256-128-64, drop=0.3, log)
    pv, pt = train_embed_mlp(Xn_tr, gu_tr[tr], dong_tr[tr], y_log[tr],
                              Xn_va, gu_tr[va], dong_tr[va], y_log[va],
                              Xn_te, gu_te, dong_te,
                              layers=[256,128,64], drop=0.3, seed=42)
    oof.setdefault('emb_a_log', np.zeros(n_orig))[va] = np.expm1(pv)
    tpred['emb_a_log'] = tpred.get('emb_a_log', np.zeros(n_te)) + np.expm1(pt)/N_SPLITS

    # Embed-MLP A: raw
    pv, pt = train_embed_mlp(Xn_tr, gu_tr[tr], dong_tr[tr], y_raw[tr],
                              Xn_va, gu_tr[va], dong_tr[va], y_raw[va],
                              Xn_te, gu_te, dong_te,
                              layers=[256,128,64], drop=0.3, seed=42)
    oof.setdefault('emb_a_raw', np.zeros(n_orig))[va] = pv
    tpred['emb_a_raw'] = tpred.get('emb_a_raw', np.zeros(n_te)) + pt/N_SPLITS

    # Embed-MLP B: 깊은 (128-128-64-32, drop=0.4, log)
    pv, pt = train_embed_mlp(Xn_tr, gu_tr[tr], dong_tr[tr], y_log[tr],
                              Xn_va, gu_tr[va], dong_tr[va], y_log[va],
                              Xn_te, gu_te, dong_te,
                              layers=[128,128,64,32], drop=0.4, seed=123)
    oof.setdefault('emb_b_log', np.zeros(n_orig))[va] = np.expm1(pv)
    tpred['emb_b_log'] = tpred.get('emb_b_log', np.zeros(n_te)) + np.expm1(pt)/N_SPLITS

    # Embed-MLP B: raw
    pv, pt = train_embed_mlp(Xn_tr, gu_tr[tr], dong_tr[tr], y_raw[tr],
                              Xn_va, gu_tr[va], dong_tr[va], y_raw[va],
                              Xn_te, gu_te, dong_te,
                              layers=[128,128,64,32], drop=0.4, seed=123)
    oof.setdefault('emb_b_raw', np.zeros(n_orig))[va] = pv
    tpred['emb_b_raw'] = tpred.get('emb_b_raw', np.zeros(n_te)) + pt/N_SPLITS

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
    tag = " ★EMB" if 'emb' in nm else ""
    print(f"  {nm:15s}: {rmse:,.0f}{tag}")

print(f"\n{'='*60}")
print("Embed-MLP 상관관계")
print("="*60, flush=True)
emb_names = [k for k in oof if 'emb' in k]
for en in emb_names:
    corrs = {r: np.corrcoef(oof[r], oof[en])[0,1] for r in ['cb_log','lgb_raw','et_log']}
    print(f"  {en:15s} → " + ", ".join(f"{k}:{v:.4f}" for k,v in corrs.items()))

print(f"\n  (참고: 이전 MLP는 corr ~0.81, OOF ~5,700)")
print(f"  (참고: ET는 corr ~0.96, OOF ~3,600)")

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
EMB_A = ['emb_a_log','emb_a_raw']
EMB_B = ['emb_b_log','emb_b_raw']
EMB_ALL = EMB_A + EMB_B

_, r12 = ridge_stack(oof, tpred, y_true, B12, "기존 12모델")
_, r12a = ridge_stack(oof, tpred, y_true, B12+EMB_A, "+Emb-A(256-128-64)")
_, r12b = ridge_stack(oof, tpred, y_true, B12+EMB_B, "+Emb-B(128×2-64-32)")
_, r16 = ridge_stack(oof, tpred, y_true, B12+EMB_ALL, "+Emb 전체(4개)")

print(f"\n  12 → +Emb-A: {r12-r12a:+.1f}")
print(f"  12 → +Emb-B: {r12-r12b:+.1f}")
print(f"  12 → +Emb전체: {r12-r16:+.1f}")

print(f"\n{'='*60}")
print("최종 요약")
print("="*60)
print(f"  기존 12모델:   {r12:,.1f}")
print(f"  +Emb-A(2):     {r12a:,.1f} ({r12-r12a:+.1f})")
print(f"  +Emb-B(2):     {r12b:,.1f} ({r12-r12b:+.1f})")
print(f"  +Emb 전체(4):  {r16:,.1f} ({r12-r16:+.1f})")
print(f"\n  성공 기준: OOF < 4,000 AND corr < 0.95")
print(f"  이전 MLP: OOF 5,700 / corr 0.81 → 실패")
print(f"  ET 참고:  OOF 3,600 / corr 0.96 → 성공(-4점)", flush=True)
