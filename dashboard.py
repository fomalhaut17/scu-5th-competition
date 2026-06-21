import pandas as pd
import os

BASELINE_RMSE = 2787
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.csv')

LAYER_NAMES = {
    'L1': '피처',
    'L2': '모델',
    'L3': '튜닝',
    'L4': '블렌딩',
}


def load_results():
    if not os.path.exists(CSV_PATH):
        print("results.csv가 없습니다.")
        return None
    return pd.read_csv(CSV_PATH, keep_default_na=False)


def print_header(df):
    best_row = df[df['status'] == 'best']
    if best_row.empty:
        best_row = df.loc[df['oof_rmse'].idxmin()]
        best_rmse = int(best_row['oof_rmse'])
    else:
        best_rmse = int(best_row.iloc[0]['oof_rmse'])

    diff = (best_rmse - BASELINE_RMSE) / BASELINE_RMSE * 100
    print("=" * 50)
    print("  SCU 5th Competition Dashboard")
    print(f"  Baseline: {BASELINE_RMSE:,}  |  Best: {best_rmse:,} ({diff:+.1f}%)")
    print("=" * 50)


def print_pipeline(df):
    confirmed = df[df['status'].isin(['confirmed', 'best'])]
    parts = []
    for layer in ['L1', 'L2', 'L3', 'L4']:
        layer_items = confirmed[confirmed['layer'] == layer]['abbr'].tolist()
        if layer_items:
            parts.append(', '.join(layer_items))
    if parts:
        print(f"\n  확정 파이프라인: {' → '.join(parts)}")


def print_layers(df):
    for layer in ['L1', 'L2', 'L3', 'L4']:
        layer_df = df[df['layer'] == layer]
        if layer_df.empty:
            continue

        name = LAYER_NAMES.get(layer, '')
        confirmed_abbrs = layer_df[layer_df['status'].isin(['confirmed', 'best'])]['abbr'].tolist()
        confirmed_str = ', '.join(confirmed_abbrs) if confirmed_abbrs else '미정'

        print(f"\n  [{layer} {name}] 확정: {confirmed_str}")

        for _, row in layer_df.iterrows():
            rmse = int(row['oof_rmse'])
            diff = (rmse - BASELINE_RMSE) / BASELINE_RMSE * 100
            mark = "★" if row['status'] in ('confirmed', 'best') else " "
            status = row['status']
            print(f"    {mark} {row['num']}  {row['abbr']:<10s}  {rmse:>5,}  {diff:>+6.1f}%  {status}")


def print_submissions(df):
    submitted = df[df['date'] != '']
    print(f"\n  {'─' * 40}")
    print("  Kaggle 제출 이력")

    if submitted.empty:
        print("    (아직 제출 없음)")
    else:
        for i, (_, row) in enumerate(submitted.iterrows(), 1):
            public = row['public_rmse'] if row['public_rmse'] != '' else '?'
            print(f"    #{i}  {row['date']}  {row['abbr']}  Public: {public}")


def main():
    df = load_results()
    if df is None:
        return

    print_header(df)
    print_pipeline(df)
    print_layers(df)
    print_submissions(df)
    print()


if __name__ == '__main__':
    main()
