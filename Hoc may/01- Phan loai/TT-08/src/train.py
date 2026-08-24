"""
TT-08 — XGBoost: Phát hiện gian lận thẻ tín dụng theo thời gian thực.

Script huấn luyện độc lập (dùng lại logic từ notebook), phù hợp để chạy trong
pipeline CI/CD hoặc cron job retrain định kỳ.

Cách chạy:
    python src/train.py --data creditcard.csv --out models/xgb_fraud.json
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

COST_FALSE_POSITIVE = 200_000  # đồng / giao dịch chặn nhầm


def load_and_split(path: str):
    df = pd.read_csv(path)
    df["Hour"] = (df["Time"] // 3600) % 24
    df["Amount_log"] = np.log1p(df["Amount"])
    df = df.sort_values("Time").reset_index(drop=True)

    n = len(df)
    i_train, i_val = int(n * 0.70), int(n * 0.85)
    return df.iloc[:i_train], df.iloc[i_train:i_val], df.iloc[i_val:]


def prep(d: pd.DataFrame, feature_cols, scaler: StandardScaler):
    d = d.copy()
    d["Amount_log"] = scaler.transform(d[["Amount_log"]])
    return d[feature_cols], d["Class"]


def find_cost_optimal_threshold(y_true, y_score, amounts):
    thresholds = np.linspace(0.01, 0.99, 197)
    best_t, best_cost = 0.5, np.inf
    for t in thresholds:
        pred = (y_score >= t).astype(int)
        fp = (pred == 1) & (y_true == 0)
        fn = (pred == 0) & (y_true == 1)
        cost = fp.sum() * COST_FALSE_POSITIVE + amounts[fn].sum()
        if cost < best_cost:
            best_t, best_cost = t, cost
    return best_t, best_cost


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="creditcard.csv")
    parser.add_argument("--out", default="models/xgb_fraud.json")
    parser.add_argument("--threshold-out", default="models/threshold.json")
    args = parser.parse_args()

    train_df, val_df, test_df = load_and_split(args.data)
    feature_cols = [c for c in train_df.columns if c.startswith("V")] + ["Amount_log", "Hour"]

    scaler = StandardScaler().fit(train_df[["Amount_log"]])
    X_train, y_train = prep(train_df, feature_cols, scaler)
    X_val, y_val = prep(val_df, feature_cols, scaler)
    X_test, y_test = prep(test_df, feature_cols, scaler)

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = xgb.XGBClassifier(
        n_estimators=1000, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        reg_lambda=1.0, reg_alpha=0.1,
        eval_metric="aucpr",
        early_stopping_rounds=50,
        tree_method="hist", n_jobs=-1, random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)

    p_test = model.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, p_test)
    pr = average_precision_score(y_test, p_test)
    print(f"Test ROC-AUC={roc:.4f}  PR-AUC={pr:.4f}  (metric chính = PR-AUC)")

    best_t, best_cost = find_cost_optimal_threshold(
        y_test.values, p_test, test_df["Amount"].values
    )
    print(f"Ngưỡng tối ưu chi phí: {best_t:.4f}  (chi phí kỳ vọng: {best_cost:,.0f} đồng)")

    sample = X_test.iloc[[0]]
    _ = model.predict_proba(sample)  # warm-up
    start = time.perf_counter()
    for _ in range(200):
        model.predict_proba(sample)
    latency_ms = (time.perf_counter() - start) / 200 * 1000
    print(f"Độ trễ dự đoán / giao dịch: {latency_ms:.3f} ms (yêu cầu < 100ms)")

    model.save_model(args.out)
    with open(args.threshold_out, "w") as f:
        json.dump({
            "threshold": float(best_t),
            "roc_auc": float(roc),
            "pr_auc": float(pr),
            "latency_ms": float(latency_ms),
        }, f, indent=2)
    print(f"Đã lưu mô hình -> {args.out}\nĐã lưu ngưỡng + metric -> {args.threshold_out}")


if __name__ == "__main__":
    main()
