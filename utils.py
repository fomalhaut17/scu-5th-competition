import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
import os
import warnings
warnings.filterwarnings('ignore')

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CAT_FEATURES = ['Gu', 'Dong']


def load_data():
    train_path = os.path.join(ROOT_DIR, 'seoul_real_estate_train.csv')
    test_path = os.path.join(ROOT_DIR, 'seoul_real_estate_test.csv')
    sub_path = os.path.join(ROOT_DIR, 'sample_submission.csv')

    if not os.path.exists(train_path):
        import gdown
        gdown.download('https://drive.google.com/uc?id=1Jf2eaIaEA-yfRyYWl_Wk7SfozaaRfaft', train_path, quiet=False)
        gdown.download('https://drive.google.com/uc?id=1WUnwAnuXTSBGu3DpRda-5NgAHys154EK', test_path, quiet=False)
        gdown.download('https://drive.google.com/uc?id=1v3CsMpnKci14OcqYEADPzAJyUcT3U-RA', sub_path, quiet=False)

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample_sub = pd.read_csv(sub_path)
    return train, test, sample_sub


def base_preprocess(df):
    """main.py와 동일한 기본 전처리"""
    df = df.copy()
    df['Distance_to_Subway'] = df['Distance_to_Subway'].fillna(df['Distance_to_Subway'].median())
    df['Age'] = 2026 - df['Year_Built']
    df['Year'] = df['Transaction_YearMonth'] // 100
    df['Month'] = df['Transaction_YearMonth'] % 100
    df = df.drop(columns=['ID', 'Transaction_YearMonth', 'Year_Built'])
    return df


def add_feature_engineering(df):
    """[S01] 추가 피처 4개 생성"""
    df = df.copy()
    df['YearMonth_Seq'] = (df['Year'] - 2024) * 12 + df['Month']
    df['Area_x_Floor'] = df['Exclusive_Area'] * df['Floor']
    df['Floor_per_Area'] = df['Floor'] / df['Exclusive_Area']
    df['Brand_x_Area'] = df['Brand_Apartment'] * df['Exclusive_Area']
    return df


def add_target_encoding(train_df, test_df, target_col='Target', smoothing=10, n_splits=5):
    """K-Fold 기반 타겟 인코딩 (누수 방지). Target 컬럼이 있는 상태에서 호출할 것."""
    train_df = train_df.copy()
    test_df = test_df.copy()

    log_target = np.log1p(train_df[target_col])
    global_mean = log_target.mean()

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for col in CAT_FEATURES:
        new_col = f'{col}_TE'
        train_df[new_col] = np.nan

        for train_idx, val_idx in kf.split(train_df):
            fold_data = log_target.iloc[train_idx].groupby(train_df[col].iloc[train_idx])
            fold_means = fold_data.mean()
            fold_counts = fold_data.count()
            smoothed = (fold_counts * fold_means + smoothing * global_mean) / (fold_counts + smoothing)
            train_df.loc[train_df.index[val_idx], new_col] = (
                train_df[col].iloc[val_idx].map(smoothed).values
            )

        train_df[new_col] = train_df[new_col].fillna(global_mean)

        all_data = log_target.groupby(train_df[col])
        all_means = all_data.mean()
        all_counts = all_data.count()
        all_smoothed = (all_counts * all_means + smoothing * global_mean) / (all_counts + smoothing)
        test_df[new_col] = test_df[col].map(all_smoothed).fillna(global_mean)

    return train_df, test_df


def encode_categoricals(train_df, test_df, as_category=False):
    """범주형 변수 LabelEncoding. as_category=True면 category dtype 변환 (LightGBM/XGBoost용)"""
    train_df = train_df.copy()
    test_df = test_df.copy()
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


def split_data(train_df):
    X = train_df.drop(columns=['Target'])
    y = np.log1p(train_df['Target'])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val


def evaluate(y_val_log, y_pred_raw):
    """RMSE 계산 (y_val_log: log1p된 실제값, y_pred_raw: 원본 스케일 예측값)"""
    y_true = np.expm1(y_val_log)
    return np.sqrt(np.mean((y_pred_raw - y_true) ** 2))


N_SPLITS = 5


def kfold_train_predict(X, y, X_test, model_fn, fit_fn, n_splits=N_SPLITS):
    """
    K-Fold 교차 검증으로 학습 및 예측.

    model_fn: () -> model (모델 생성)
    fit_fn: (model, X_train, y_train, X_val, y_val) -> model (학습)

    Returns: oof_pred (원본 스케일), test_pred (원본 스케일), overall_rmse
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_pred = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"  Fold {fold + 1}/{n_splits}", end=" → ")

        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

        model = model_fn()
        model = fit_fn(model, X_tr, y_tr, X_va, y_va)

        val_pred = np.expm1(model.predict(X_va))
        oof_pred[val_idx] = val_pred
        test_preds += np.expm1(model.predict(X_test)) / n_splits

        fold_rmse = np.sqrt(np.mean((val_pred - np.expm1(y_va.values)) ** 2))
        print(f"RMSE: {fold_rmse:,.0f} 만원")

    overall_rmse = np.sqrt(np.mean((oof_pred - np.expm1(y.values)) ** 2))
    print(f"  ──────────────────────────")
    print(f"  OOF RMSE: {overall_rmse:,.0f} 만원")

    return oof_pred, test_preds, overall_rmse


def oot_split(df, holdout_months=3):
    """Out-of-Time 분할. 마지막 n개월을 검증셋으로 분리.
    Returns: train_idx, val_idx, cutoff_ym"""
    ym = df['Transaction_YearMonth']
    unique_ym = sorted(ym.unique())
    cutoff_ym = unique_ym[-holdout_months]
    train_idx = df.index[ym < cutoff_ym].tolist()
    val_idx = df.index[ym >= cutoff_ym].tolist()
    return train_idx, val_idx, cutoff_ym


def record_result(layer, num, abbr, description, oof_rmse, status='tested'):
    """results.csv에 결과 기록. 같은 layer+num이 있으면 업데이트."""
    csv_path = os.path.join(ROOT_DIR, 'results.csv')
    cols = ['layer', 'num', 'abbr', 'description', 'oof_rmse', 'status', 'date', 'public_rmse']

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, keep_default_na=False)
    else:
        df = pd.DataFrame(columns=cols)

    mask = (df['layer'] == layer) & (df['num'] == num)
    if mask.any():
        df.loc[mask, ['abbr', 'description', 'oof_rmse', 'status']] = [abbr, description, round(oof_rmse), status]
    else:
        new_row = pd.DataFrame([{
            'layer': layer, 'num': num, 'abbr': abbr,
            'description': description, 'oof_rmse': round(oof_rmse),
            'status': status, 'date': '', 'public_rmse': '',
        }])
        df = pd.concat([df, new_row], ignore_index=True)

    df.to_csv(csv_path, index=False)


def save_submission(sample_sub, preds, filename):
    sub = sample_sub.copy()
    sub['Target'] = preds
    filepath = os.path.join(ROOT_DIR, filename)
    sub.to_csv(filepath, index=False)
    print(f"제출 파일 '{filename}' 생성 완료")
