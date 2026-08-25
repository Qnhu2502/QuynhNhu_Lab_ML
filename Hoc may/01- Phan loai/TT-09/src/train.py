"""
TT-09 — AdaBoost: huấn luyện & lưu model phát hiện xâm nhập mạng (NSL-KDD, định dạng ARFF).

Chạy từ thư mục gốc dự án:
    python src/train.py
    python src/train.py --data-dir data --n-estimators 300 --learning-rate 0.5

Đọc KDDTrain+.arff / KDDTest+.arff (WEKA ARFF) — KHÔNG phải bản .txt gốc có tên tấn công cụ thể.
Xem README mục "Dữ liệu" để biết vì sao bản ARFF không tách được nhóm DoS/Probe/R2L/U2R.
Mirror đúng logic đã viết trong notebooks/adaboost_ids.ipynb (không dùng test để chọn siêu tham số).
"""
from __future__ import annotations

import argparse
import time
from io import StringIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

COLS_ARFF = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
    'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
    'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
    'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
    'class',  # 'normal' hoặc 'anomaly'
]

RANDOM_STATE = 42


def load_nslkdd_arff(path: Path) -> pd.DataFrame:
    """Đọc 1 file NSL-KDD định dạng ARFF: bỏ qua block '@attribute', đọc CSV từ sau '@data'."""
    with open(path, 'r') as f:
        lines = f.readlines()
    data_start = next(i for i, l in enumerate(lines) if l.strip().lower() == '@data') + 1
    csv_text = ''.join(lines[data_start:])
    df = pd.read_csv(StringIO(csv_text), header=None, names=COLS_ARFF)
    df['attack'] = (df['class'] != 'normal').astype(int)
    df = df.drop(columns=['class'])
    return df


# Nhiều công cụ upload/đồng bộ tự đổi ký tự '+' trong tên file NSL-KDD thành '_'
# (ví dụ 'KDDTrain+.arff' -> 'KDDTrain_.arff'). Dò cả hai kiểu tên để tránh FileNotFoundError.
TRAIN_NAME_CANDIDATES = ['KDDTrain+.arff', 'KDDTrain_.arff', 'KDDTrain+_.arff']
TEST_NAME_CANDIDATES = ['KDDTest+.arff', 'KDDTest_.arff', 'KDDTest+_.arff']


def _find_file(dir_path: Path, candidates: list[str]) -> Path | None:
    for name in candidates:
        p = dir_path / name
        if p.exists():
            return p
    return None


def find_data_files(data_dir: Path) -> tuple[Path, Path]:
    train_path = _find_file(data_dir, TRAIN_NAME_CANDIDATES)
    test_path = _find_file(data_dir, TEST_NAME_CANDIDATES)
    if train_path is None or test_path is None:
        raise FileNotFoundError(
            f"Không tìm thấy đủ KDDTrain+.arff và KDDTest+.arff (hoặc biến thể tên với dấu '_' "
            f"thay vì '+') trong '{data_dir}'. Chạy script này từ thư mục gốc dự án, hoặc truyền "
            f"--data-dir đúng.")
    return train_path, test_path


def build_pipeline(model, cat_cols, num_cols) -> Pipeline:
    pre = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
        ('num', StandardScaler(), num_cols),
    ])
    return Pipeline([('pre', pre), ('clf', model)])


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=Path('data'),
                    help="Thư mục chứa KDDTrain+.arff/KDDTest+.arff (mặc định: data/)")
    p.add_argument('--reports-dir', type=Path, default=Path('reports'))
    p.add_argument('--models-dir', type=Path, default=Path('models'))
    p.add_argument('--n-estimators', type=int, default=300)
    p.add_argument('--learning-rate', type=float, default=0.5)
    p.add_argument('--stump-depth', type=int, default=1,
                    help="max_depth của weak learner. Mặc định=1 (stump, đúng bản chất AdaBoost).")
    return p.parse_args()


def main():
    args = parse_args()

    data_dir = args.data_dir
    train_path, test_path = find_data_files(data_dir)

    # ---- 1. Đọc dữ liệu ----
    train_df = load_nslkdd_arff(train_path)
    test_df = load_nslkdd_arff(test_path)
    print(f"Train: {train_df.shape} | Test (NSL-KDD gốc, có tấn công lạ): {test_df.shape}")

    y_train_full = train_df.pop('attack')
    X_train_full = train_df
    y_test = test_df.pop('attack')
    X_test = test_df

    cat_cols = ['protocol_type', 'service', 'flag']
    num_cols = [c for c in X_train_full.columns if c not in cat_cols]

    # ---- 2. Tách validation từ TRAIN (không đụng test) ----
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=RANDOM_STATE)

    # ---- 3. Baseline tham khảo nhanh (không phục vụ chọn model) ----
    for name, model in [
        ('Dummy', DummyClassifier(strategy='most_frequent')),
        ('1 Stump', DecisionTreeClassifier(max_depth=1, random_state=RANDOM_STATE)),
    ]:
        pipe = build_pipeline(model, cat_cols, num_cols).fit(X_tr, y_tr)
        pred = pipe.predict(X_val)
        print(f"{name}: F1={f1_score(y_val, pred):.3f}, Accuracy={accuracy_score(y_val, pred):.3f}")

    # ---- 4. AdaBoost với siêu tham số từ README (stump, n_estimators, learning_rate) ----
    ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=args.stump_depth, random_state=RANDOM_STATE),
        n_estimators=args.n_estimators, learning_rate=args.learning_rate, random_state=RANDOM_STATE)
    ada_pipe = build_pipeline(ada, cat_cols, num_cols)

    t0 = time.time()
    ada_pipe.fit(X_tr, y_tr)
    train_time = time.time() - t0
    pred_val = ada_pipe.predict(X_val)
    print(f"AdaBoost (validation): F1={f1_score(y_val, pred_val):.3f}, "
          f"Accuracy={accuracy_score(y_val, pred_val):.3f}, train_time={train_time:.2f}s")

    # ---- 5. Refit trên toàn bộ train, đánh giá CHỈ MỘT LẦN trên KDDTest+.arff ----
    ada_final = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=args.stump_depth, random_state=RANDOM_STATE),
        n_estimators=args.n_estimators, learning_rate=args.learning_rate, random_state=RANDOM_STATE)
    ada_final_pipe = build_pipeline(ada_final, cat_cols, num_cols).fit(X_train_full, y_train_full)

    pred_test = ada_final_pipe.predict(X_test)
    f1_test = f1_score(y_test, pred_test)
    acc_test = accuracy_score(y_test, pred_test)
    prec_test = precision_score(y_test, pred_test, zero_division=0)
    rec_test = recall_score(y_test, pred_test, zero_division=0)
    print(f"AdaBoost (KDDTest+.arff, tấn công lạ): F1={f1_test:.3f}, Accuracy={acc_test:.3f}, "
          f"Precision={prec_test:.3f}, Recall={rec_test:.3f}")

    # ---- 6. Lưu model & tóm tắt (luôn tạo thư mục trước khi ghi) ----
    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = args.models_dir / 'adaboost.joblib'
    joblib.dump(ada_final_pipe, model_path)
    print(f"Đã lưu model tại: {model_path}")

    summary = pd.DataFrame([{
        'model': 'AdaBoost', 'f1_validation': round(f1_score(y_val, pred_val), 3),
        'f1_test': round(f1_test, 3), 'accuracy_test': round(acc_test, 3),
        'precision_test': round(prec_test, 3), 'recall_test': round(rec_test, 3),
        'train_time_s': round(train_time, 2),
    }])
    summary_path = args.reports_dir / 'train_summary.csv'
    summary.to_csv(summary_path, index=False)
    print(f"Đã lưu tóm tắt tại: {summary_path}")


if __name__ == '__main__':
    main()
