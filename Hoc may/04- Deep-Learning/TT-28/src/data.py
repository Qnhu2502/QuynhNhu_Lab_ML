"""Nạp AG News và chia train/val/test một lần duy nhất cho mọi mô hình."""
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from .config import SEED, VAL_SIZE


def nap_du_lieu():
    ds = load_dataset("ag_news")
    X, y = ds["train"]["text"], np.array(ds["train"]["label"])
    # val tách từ train (stratify) để chọn siêu tham số / early stopping; test không bị đụng tới
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=VAL_SIZE, stratify=y, random_state=SEED)
    return {
        "texts": {"train": Xtr, "val": Xva, "test": ds["test"]["text"]},
        "y": {"train": ytr, "val": yva, "test": np.array(ds["test"]["label"])},
    }


def kiem_tra_ro_ri(data):
    """Đếm văn bản trùng giữa các tập (rò rỉ dữ liệu)."""
    t = {k: set(v) for k, v in data["texts"].items()}
    return {"train&val": len(t["train"] & t["val"]),
            "train&test": len(t["train"] & t["test"]),
            "val&test": len(t["val"] & t["test"])}
