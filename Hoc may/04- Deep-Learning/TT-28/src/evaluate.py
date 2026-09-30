"""Metric, đo thời gian, lưu kết quả để notebook tổng hợp đọc lại."""
import json
import time

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from .config import RESULTS


def do_thoi_gian_du_doan(du_doan_1_tin, texts, n=100):
    """Thời gian trung bình (ms) để dự đoán từng tin một, đã khởi động trước."""
    du_doan_1_tin(texts[0])
    t0 = time.perf_counter()
    for t in texts[:n]:
        du_doan_1_tin(t)
    return (time.perf_counter() - t0) / n * 1000


def luu_ket_qua(ten, y_true, y_pred, train_s, params, predict_ms, proba=None, extra=None):
    # lớp cân bằng nên accuracy là metric chính; macro-F1 để không bỏ sót lớp yếu
    kq = {"model": ten,
          "accuracy": float(accuracy_score(y_true, y_pred)),
          "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
          "train_seconds": float(train_s), "params": int(params),
          "predict_ms": float(predict_ms), **(extra or {})}
    (RESULTS / f"{ten}.json").write_text(json.dumps(kq, indent=2))
    arrays = {"y_true": y_true, "y_pred": y_pred}
    if proba is not None:
        arrays["proba"] = proba
    np.savez_compressed(RESULTS / f"{ten}.npz", **arrays)
    print(kq)
    return kq
