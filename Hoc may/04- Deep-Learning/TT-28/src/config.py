"""Hằng số dùng chung: đường dẫn, seed, siêu tham số."""
import os
import random
from pathlib import Path

import numpy as np

SEED = 42
ROOT = Path(__file__).resolve().parents[1]      # đường dẫn theo vị trí file, chạy ở đâu cũng đúng
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
RESULTS = REPORTS / "results"                   # json + npz kết quả từng mô hình
for p in (MODELS, RESULTS):
    p.mkdir(parents=True, exist_ok=True)

LABELS = ["World", "Sports", "Business", "Sci/Tech"]
VAL_SIZE = 0.1                                  # tách từ train; test chỉ dùng đánh giá cuối

# nhánh A (tự xây)
MAX_LEN, VOCAB = 64, 20000                      # tin ngắn, 64 token là đủ; 512 chỉ tốn thời gian
D_MODEL, N_HEADS, D_FF, N_BLOCKS = 128, 4, 256, 2


def set_seed(seed=SEED):
    """Cố định seed cho mọi thư viện đang cài."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
