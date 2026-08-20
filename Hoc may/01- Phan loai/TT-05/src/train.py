"""
TT-05 — SVM: Phân loại khối u lành tính / ác tính (Wisconsin Diagnostic Breast Cancer)

Chạy: python src/train.py   (từ thư mục gốc TT-05-SVM-<HoTen>/)

Fix so với bản cũ:
- Không hardcode '../models/...': mọi đường dẫn tính từ vị trí thật của file
  này (Path(__file__).resolve().parent), không phụ thuộc cwd lúc gọi script.
- Tạo thư mục output bằng mkdir(parents=True, exist_ok=True) trước khi ghi.
- Xuất số liệu (metrics) ra reports/metrics.json thay vì chỉ in ra màn hình.
- Nạp dữ liệu từ wdbc.data (file commit kèm bài), không dùng
  sklearn.datasets.load_breast_cancer, để nguồn dữ liệu nhất quán với dữ
  liệu đã nộp.
- GridSearchCV dùng scorer F2 (ưu tiên recall nhưng có ràng buộc precision),
  không dùng recall thuần (tránh baseline "luôn đoán ác tính" thắng ảo).
- Ngưỡng phân loại được chọn bằng out-of-fold prediction trên tập train
  (cross_val_predict), KHÔNG dò trên tập test, để tránh rò rỉ dữ liệu.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, fbeta_score, make_scorer,
                              precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, cross_val_predict,
                                      train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RANDOM_STATE = 42

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "wdbc.data"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

FEATURE_BASES = [
    "radius", "texture", "perimeter", "area", "smoothness",
    "compactness", "concavity", "concave_points", "symmetry", "fractal_dimension",
]
STATS = ["mean", "se", "worst"]
COLS = ["id", "diagnosis"] + [f"{b}_{s}" for s in STATS for b in FEATURE_BASES]
TARGET_NAMES = ["malignant", "benign"]  # 0, 1


def load_data():
    raw = pd.read_csv(DATA_PATH, header=None, names=COLS)
    y = raw["diagnosis"].map({"M": 0, "B": 1})  # 0 = ác tính, 1 = lành tính (tường minh)
    if y.isna().any():
        raise ValueError("Có nhãn không map được trong wdbc.data — kiểm tra lại file.")
    X = raw.drop(columns=["id", "diagnosis"])
    return X, y


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # --- sanity check: recall thuần bị "lừa" bởi baseline luôn đoán ác tính ---
    dummy = DummyClassifier(strategy="constant", constant=0)
    dummy.fit(X_train, y_train)
    dummy_recall = recall_score(y_test, dummy.predict(X_test), pos_label=0)
    dummy_precision = precision_score(y_test, dummy.predict(X_test), pos_label=0)

    # --- GridSearchCV với scorer F2 (không dùng recall thuần) ---
    fbeta2_malignant = make_scorer(fbeta_score, beta=2, pos_label=0)
    pipe_gs = Pipeline([
        ("scale", StandardScaler()),
        ("svm", SVC(probability=True, class_weight="balanced", random_state=RANDOM_STATE)),
    ])
    param_grid = {
        "svm__C": [0.1, 1, 10, 100],
        "svm__gamma": ["scale", 0.001, 0.01, 0.1],
        "svm__kernel": ["linear", "rbf", "poly"],
    }
    gs = GridSearchCV(pipe_gs, param_grid, cv=5, scoring=fbeta2_malignant, n_jobs=-1)
    gs.fit(X_train, y_train)
    best_model = gs.best_estimator_
    svm_best = best_model.named_steps["svm"]

    # --- chọn ngưỡng bằng out-of-fold prediction trên TRAIN (không đụng test) ---
    oof_proba = cross_val_predict(best_model, X_train, y_train, cv=5, method="predict_proba")[:, 0]
    thresholds = np.arange(0.05, 0.55, 0.01)
    rows = []
    for t in thresholds:
        pred_t = np.where(oof_proba >= t, 0, 1)
        rows.append({
            "threshold": t,
            "recall": recall_score(y_train, pred_t, pos_label=0),
            "precision": precision_score(y_train, pred_t, pos_label=0),
        })
    threshold_df = pd.DataFrame(rows)
    candidates = threshold_df[threshold_df["recall"] >= 0.98]
    chosen = (
        candidates.sort_values("precision", ascending=False).iloc[0]
        if len(candidates)
        else threshold_df.sort_values("recall", ascending=False).iloc[0]
    )

    # --- đánh giá MỘT LẦN DUY NHẤT trên test ---
    proba_test = best_model.predict_proba(X_test)[:, 0]
    final_pred = np.where(proba_test >= chosen["threshold"], 0, 1)
    final_recall = recall_score(y_test, final_pred, pos_label=0)
    final_precision = precision_score(y_test, final_pred, pos_label=0)
    cm = confusion_matrix(y_test, final_pred, labels=[0, 1])
    fn = int(cm[0, 1])

    # --- so sánh với Logistic Regression ---
    pipe_lr = Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(class_weight="balanced", max_iter=5000, random_state=RANDOM_STATE)),
    ])
    pipe_lr.fit(X_train, y_train)
    proba_lr = pipe_lr.predict_proba(X_test)[:, 0]
    lr_recall = recall_score(y_test, pipe_lr.predict(X_test), pos_label=0)
    lr_precision = precision_score(y_test, pipe_lr.predict(X_test), pos_label=0)

    # --- lưu model ---
    model_path = MODELS_DIR / "svm_pipeline.joblib"
    joblib.dump(best_model, model_path)

    # --- xuất metrics ---
    metrics = {
        "dummy_all_malignant_baseline": {
            "recall": round(dummy_recall, 4),
            "precision": round(dummy_precision, 4),
            "note": "Minh hoạ vì sao không dùng recall thuần làm scoring cho GridSearchCV",
        },
        "grid_search_best_params": gs.best_params_,
        "grid_search_best_cv_f2": round(gs.best_score_, 4),
        "support_vectors": {
            "per_class": dict(zip(TARGET_NAMES, [int(x) for x in svm_best.n_support_])),
            "total": int(svm_best.n_support_.sum()),
            "train_size": int(len(X_train)),
        },
        "chosen_threshold": round(float(chosen["threshold"]), 2),
        "threshold_selected_on": "out-of-fold predictions on TRAIN (cross_val_predict), not test",
        "final_test_recall_ác_tính": round(final_recall, 4),
        "final_test_precision_ác_tính": round(final_precision, 4),
        "false_negatives_on_test": fn,
        "malignant_in_test": int(cm[0].sum()),
        "roc_auc_svm": round(roc_auc_score(1 - y_test, proba_test), 4),
        "logistic_regression_comparison": {
            "recall_ác_tính": round(lr_recall, 4),
            "precision_ác_tính": round(lr_precision, 4),
            "roc_auc": round(roc_auc_score(1 - y_test, proba_lr), 4),
        },
        "model_path": str(model_path.relative_to(ROOT_DIR)),
    }
    metrics_path = REPORTS_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\nĐã lưu model: {model_path}")
    print(f"Đã lưu metrics: {metrics_path}")


if __name__ == "__main__":
    main()
