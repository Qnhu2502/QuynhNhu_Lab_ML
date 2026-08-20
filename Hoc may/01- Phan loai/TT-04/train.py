"""
src/train.py — Huấn luyện Logistic Regression dự đoán nguy cơ bệnh tim.

Cách chạy:
    python src/train.py --data data/heart.csv
    python src/train.py --data data/heart.csv --target-recall 0.90 --test-size 0.2

Yêu cầu: đặt file heart.csv vào thư mục data/ trước khi chạy (xem data/README_DATA.txt).
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (classification_report, precision_recall_curve,
                              precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore", category=ConvergenceWarning)

RANDOM_STATE = 42
TARGET_COL = "target"
CATEGORICAL_COLS = ["cp", "restecg", "slope", "thal", "ca", "sex", "fbs", "exang"]


def load_data(path):
    """Đọc CSV và loại bỏ dòng trùng lặp TRƯỚC khi chia train/test.

    Bản Kaggle của bộ dữ liệu Heart Disease chứa các dòng nhân bản từ ~303 dòng
    gốc (UCI) lên ~1025 dòng. Nếu không loại trùng trước khi split, các dòng
    giống hệt nhau có thể rơi vào cả train và test → rò rỉ dữ liệu (data
    leakage) và đánh giá mô hình lạc quan giả tạo.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Không tìm thấy '{path}'. Hãy tải heart.csv (xem data/README_DATA.txt) "
            f"và đặt đúng đường dẫn, hoặc truyền --data <đường_dẫn>."
        )
    df_raw = pd.read_csv(path)
    n_before = len(df_raw)
    df = df_raw.drop_duplicates().reset_index(drop=True)
    n_after = len(df)
    print(f"Số dòng trước khi loại trùng: {n_before}")
    print(f"Số dòng sau khi loại trùng:  {n_after} (đã loại {n_before - n_after} dòng)")
    return df


def build_pipeline(categorical_cols, numeric_cols, scoring="roc_auc"):
    """Xây Pipeline: ColumnTransformer(OneHotEncoder(drop='first') cho biến
    phân loại + StandardScaler cho biến số) rồi LogisticRegressionCV.

    scoring='roc_auc' (KHÔNG dùng 'recall'): nếu tối ưu trực tiếp recall, CV có
    xu hướng chọn C rất nhỏ (phạt mạnh) vì cách "dễ" nhất để đạt recall cao là
    dự đoán gần như luôn là lớp dương — hệ số co gần về 0, bảng odds ratio trở
    nên vô nghĩa. roc_auc đánh giá khả năng phân biệt trên mọi ngưỡng, tránh
    lỗi suy biến này. Ngưỡng theo recall mong muốn được chọn riêng bằng
    choose_threshold().
    """
    preprocess = ColumnTransformer([
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
        ("num", StandardScaler(), numeric_cols),
    ])
    model = LogisticRegressionCV(
        Cs=10, cv=5, penalty="l2", scoring=scoring,
        max_iter=2000, random_state=RANDOM_STATE,
    )
    return Pipeline([("preprocess", preprocess), ("model", model)])


def odds_ratio_table(pipeline, out_path=None):
    """Tính bảng hệ số & odds ratio từ pipeline đã fit, kèm cảnh báo nếu mô
    hình có dấu hiệu bị co hệ số về 0 (bug cũ do scoring='recall')."""
    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    coefs = pipeline.named_steps["model"].coef_[0]
    table = pd.DataFrame({
        "dac_trung": feature_names,
        "he_so_w": coefs,
        "odds_ratio": np.exp(coefs),
    }).sort_values("odds_ratio", ascending=False)

    max_abs_w = np.abs(coefs).max()
    if max_abs_w < 0.05:
        print(f"CẢNH BÁO: |hệ số| lớn nhất chỉ {max_abs_w:.4f} — mô hình có thể bị "
              "co hệ số về 0 (kiểm tra lại scoring của LogisticRegressionCV).")

    if out_path:
        table.to_csv(out_path, index=False)
    return table


def choose_threshold(pipeline, X_train, y_train, target_recall=0.90, cv=5):
    """Chọn ngưỡng phân loại đạt recall >= target_recall bằng out-of-fold
    prediction trên TẬP TRAIN (cross_val_predict).

    Không dò ngưỡng trực tiếp trên precision_recall_curve của tập test: làm
    vậy tương đương "fit" ngưỡng lên chính dữ liệu dùng để báo cáo hiệu năng
    cuối cùng, khiến precision/recall trên test lạc quan hơn thực tế.
    """
    oof_proba = cross_val_predict(pipeline, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_train, oof_proba)
    valid = recall[:-1] >= target_recall
    if not valid.any():
        raise ValueError(f"Không tìm được ngưỡng nào đạt recall >= {target_recall} (out-of-fold).")
    best_idx = np.where(valid)[0][-1]
    return thresholds[best_idx], recall[best_idx], precision[best_idx]


def vif_table(X_train_numeric, numeric_cols, out_path=None):
    """VIF chỉ tính trên biến số của TẬP TRAIN (đã chuẩn hoá) — không dùng
    toàn bộ df (train+test), để tránh thông tin từ test rò rỉ vào bước phân
    tích đa cộng tuyến."""
    X_num = StandardScaler().fit_transform(X_train_numeric)
    table = pd.DataFrame({
        "bien": numeric_cols,
        "VIF": [variance_inflation_factor(X_num, i) for i in range(len(numeric_cols))],
    }).sort_values("VIF", ascending=False)
    if out_path:
        table.to_csv(out_path, index=False)
    return table


def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện Logistic Regression - Heart Disease")
    parser.add_argument("--data", default="data/heart.csv", help="Đường dẫn file CSV dữ liệu")
    parser.add_argument("--target-recall", type=float, default=0.90, help="Recall tối thiểu mong muốn")
    parser.add_argument("--test-size", type=float, default=0.2, help="Tỉ lệ tập test")
    parser.add_argument("--reports-dir", default="reports", help="Thư mục lưu báo cáo (csv)")
    parser.add_argument("--models-dir", default="models", help="Thư mục lưu mô hình đã huấn luyện")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.reports_dir, exist_ok=True)
    os.makedirs(args.models_dir, exist_ok=True)

    df = load_data(args.data)
    numeric_cols = [c for c in df.columns if c not in CATEGORICAL_COLS + [TARGET_COL]]

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=RANDOM_STATE)
    print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

    pipeline = build_pipeline(CATEGORICAL_COLS, numeric_cols)
    pipeline.fit(X_train, y_train)

    print("\n=== Bảng Odds Ratio ===")
    odds = odds_ratio_table(pipeline, out_path=os.path.join(args.reports_dir, "odds_ratio.csv"))
    print(odds.head(10).to_string(index=False))

    print("\n=== Chọn ngưỡng theo recall mục tiêu ===")
    threshold, oof_recall, oof_precision = choose_threshold(
        pipeline, X_train, y_train, target_recall=args.target_recall)
    y_proba_test = pipeline.predict_proba(X_test)[:, 1]
    y_pred_test = (y_proba_test >= threshold).astype(int)

    print(f"Ngưỡng chọn (out-of-fold, TRAIN): {threshold:.3f}")
    print(f"Recall/Precision out-of-fold (TRAIN): {oof_recall:.3f} / {oof_precision:.3f}")
    print(f"\nTest AUC: {roc_auc_score(y_test, y_proba_test):.3f}")
    print(f"Test recall @ ngưỡng:    {recall_score(y_test, y_pred_test):.3f}")
    print(f"Test precision @ ngưỡng: {precision_score(y_test, y_pred_test):.3f}")
    print(classification_report(y_test, y_pred_test))

    print("=== Bảng VIF (trên train) ===")
    vif = vif_table(X_train[numeric_cols], numeric_cols,
                     out_path=os.path.join(args.reports_dir, "vif_table.csv"))
    print(vif.to_string(index=False))

    import joblib
    model_path = os.path.join(args.models_dir, "logreg_pipeline.pkl")
    joblib.dump(pipeline, model_path)
    print(f"\nĐã lưu mô hình vào {model_path}")
    print(f"Báo cáo (odds_ratio.csv, vif_table.csv) đã lưu vào {args.reports_dir}/")


if __name__ == "__main__":
    main()
