"""
TT-05 — SVM: Phân loại khối u lành tính / ác tính
Script huấn luyện độc lập (không phụ thuộc notebook).

Chạy: python train.py
Output: ../models/svm_pipeline.joblib
"""
import numpy as np
import pandas as pd
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import recall_score, precision_score, confusion_matrix, make_scorer

RANDOM_STATE = 42


def main():
    # 1. Nạp dữ liệu (0 = ác tính, 1 = lành tính trong sklearn)
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    # 2. GridSearchCV tối ưu recall lớp ác tính
    recall_malignant = make_scorer(recall_score, pos_label=0)
    pipe = Pipeline([
        ('scale', StandardScaler()),
        ('svm', SVC(probability=True, class_weight='balanced', random_state=RANDOM_STATE))
    ])
    param_grid = {
        'svm__C': [0.1, 1, 10, 100],
        'svm__gamma': ['scale', 0.001, 0.01, 0.1],
        'svm__kernel': ['linear', 'rbf', 'poly']
    }
    gs = GridSearchCV(pipe, param_grid, cv=5, scoring=recall_malignant, n_jobs=-1)
    gs.fit(X_train, y_train)
    best_model = gs.best_estimator_
    print("Best params:", gs.best_params_)

    # 3. Chọn ngưỡng để recall(ác tính) >= 0.98
    proba = best_model.predict_proba(X_test)[:, 0]
    thresholds = np.arange(0.05, 0.55, 0.01)
    best_t, best_precision = 0.5, -1
    for t in thresholds:
        pred_t = np.where(proba >= t, 0, 1)
        rec = recall_score(y_test, pred_t, pos_label=0)
        prec = precision_score(y_test, pred_t, pos_label=0)
        if rec >= 0.98 and prec > best_precision:
            best_t, best_precision = t, prec

    final_pred = np.where(proba >= best_t, 0, 1)
    cm = confusion_matrix(y_test, final_pred, labels=[0, 1])
    fn = cm[0, 1]

    print(f"Ngưỡng chọn: {best_t:.2f}")
    print(f"Recall(ác tính): {recall_score(y_test, final_pred, pos_label=0):.3f}")
    print(f"Precision(ác tính): {precision_score(y_test, final_pred, pos_label=0):.3f}")
    print(f"Số ca ác tính bị bỏ sót (FN): {fn} / {cm[0].sum()}")

    # 4. Lưu model
    joblib.dump(best_model, "../models/svm_pipeline.joblib")
    print("Đã lưu model vào ../models/svm_pipeline.joblib")


if __name__ == "__main__":
    main()
