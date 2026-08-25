"""
TT-07 — Gradient Boosting: huấn luyện & lưu model dự đoán thu nhập (Adult Census Income).

Chạy từ thư mục gốc dự án:
    python src/train.py
    python src/train.py --data-dir data --learning-rate 0.05 --n-estimators 500

Script này mirror đúng logic đã kiểm chứng trong notebooks/gradient_boosting_income.ipynb
(bao gồm việc KHÔNG dùng test để chọn siêu tham số — xem README mục "Phương pháp").
"""
import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

COLS = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
        'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
        'hours-per-week', 'native-country', 'income']

RANDOM_STATE = 42


def load_clean(path: Path, skiprows: int = 0) -> pd.DataFrame:
    """Xử lý 4 bẫy kinh điển của Adult Census Income (xem README mục 'Bốn bẫy trong dữ liệu')."""
    df = pd.read_csv(path, header=None, names=COLS, skiprows=skiprows,
                      skipinitialspace=True, na_values='?')
    for c in df.select_dtypes(include='object').columns:
        df[c] = df[c].str.strip()
    df['income'] = df['income'].str.rstrip('.')
    df['income'] = (df['income'] == '>50K').astype(int)
    df = df.drop(columns=['fnlwgt', 'education'])
    return df


def build_pipeline(model, cat_cols, num_cols) -> Pipeline:
    pre = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
        ('num', 'passthrough', num_cols),
    ])
    return Pipeline([('pre', pre), ('clf', model)])


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=Path('data'),
                    help="Thư mục chứa adult.data/adult.test (mặc định: data/)")
    p.add_argument('--reports-dir', type=Path, default=Path('reports'))
    p.add_argument('--models-dir', type=Path, default=Path('models'))
    p.add_argument('--learning-rate', type=float, default=0.05,
                    help="Đã chọn bằng cross-validation ở bước dò lưới trong notebook (không dùng test).")
    p.add_argument('--n-estimators', type=int, default=500)
    p.add_argument('--max-depth', type=int, default=3)
    return p.parse_args()


def main():
    args = parse_args()

    data_dir = args.data_dir
    if not (data_dir / 'adult.data').exists():
        raise FileNotFoundError(
            f"Không tìm thấy adult.data trong '{data_dir}'. "
            f"Chạy script này từ thư mục gốc dự án, hoặc truyền --data-dir đúng.")

    # ---- 1. Đọc & làm sạch dữ liệu (dùng đúng train/test chính thức UCI) ----
    train_df = load_clean(data_dir / 'adult.data')
    test_df = load_clean(data_dir / 'adult.test', skiprows=1)
    print(f"Train: {train_df.shape} | Test: {test_df.shape}")

    y_train, y_test = train_df.pop('income'), test_df.pop('income')
    X_train, X_test = train_df, test_df
    cat_cols = X_train.select_dtypes(include='object').columns.tolist()
    num_cols = X_train.select_dtypes(exclude='object').columns.tolist()

    # ---- 2. Baseline (tham khảo nhanh, không phục vụ chọn model) ----
    for name, model in [('Dummy', DummyClassifier(strategy='most_frequent')),
                         ('DecisionTree', DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE))]:
        pipe = build_pipeline(model, cat_cols, num_cols).fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        print(f"{name}: PR-AUC={average_precision_score(y_test, proba):.3f}, "
              f"ROC-AUC={roc_auc_score(y_test, proba):.3f}")

    # ---- 3. Gradient Boosting với siêu tham số đã chọn bằng CV trong notebook ----
    gb = GradientBoostingClassifier(
        n_estimators=args.n_estimators, learning_rate=args.learning_rate, max_depth=args.max_depth,
        subsample=0.8, validation_fraction=0.1, n_iter_no_change=20, random_state=RANDOM_STATE)
    gb_pipe = build_pipeline(gb, cat_cols, num_cols)

    t0 = time.time()
    gb_pipe.fit(X_train, y_train)
    train_time = time.time() - t0

    proba = gb_pipe.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, proba)
    pr_auc = average_precision_score(y_test, proba)
    print(f"GradientBoosting: ROC-AUC={roc_auc:.3f}, PR-AUC={pr_auc:.3f}, "
          f"train_time={train_time:.2f}s, cây dùng={gb.n_estimators_}/{args.n_estimators}")

    # ---- 4. Lưu model & tóm tắt (luôn tạo thư mục trước khi ghi) ----
    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = args.models_dir / 'gb_pipeline.joblib'
    joblib.dump(gb_pipe, model_path)
    print(f"Đã lưu model tại: {model_path}")

    summary = pd.DataFrame([{
        'model': 'GradientBoosting', 'roc_auc': round(roc_auc, 3),
        'pr_auc': round(pr_auc, 3), 'train_time_s': round(train_time, 2),
    }])
    summary_path = args.reports_dir / 'train_summary.csv'
    summary.to_csv(summary_path, index=False)
    print(f"Đã lưu tóm tắt tại: {summary_path}")


if __name__ == '__main__':
    main()
