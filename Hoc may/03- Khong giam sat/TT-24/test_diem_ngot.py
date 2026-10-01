"""Kiểm thử các quy tắc nghiệp vụ dễ sai nhất của TT-24 (chạy: pytest)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pca_pipeline import (chon_diem_ngot, gop_pipeline_tuyen_tinh, kiem_tra_chia_theo_nguoi,  # noqa: E402
                              tao_pipeline)


def _df(k_list, acc_list):
    return pd.DataFrame({"K": k_list, "cv_accuracy": acc_list})


def test_diem_ngot_khong_bi_ep_ve_k_lon_nhat():
    # Lỗi cũ: lưới thưa + so với accuracy 561 chiều → trả về K=561 (không giảm chiều nào).
    df = _df([10, 50, 100, 200, 561], [0.83, 0.91, 0.945, 0.955, 0.962])
    assert chon_diem_ngot(df, dung_sai=0.01) == 200


def test_diem_ngot_chon_k_nho_nhat_thoa_dieu_kien():
    df = _df([10, 50, 100, 200, 561], [0.83, 0.953, 0.955, 0.958, 0.962])
    assert chon_diem_ngot(df, dung_sai=0.01) == 50


def test_diem_ngot_tra_ve_k_lon_nhat_neu_khong_k_nao_khac_dat():
    df = _df([10, 50, 561], [0.5, 0.6, 0.96])
    assert chon_diem_ngot(df, dung_sai=0.01) == 561


def test_phat_hien_ro_ri_danh_tinh():
    with pytest.raises(ValueError):
        kiem_tra_chia_theo_nguoi(pd.Series([1, 2, 3]), pd.Series([3, 4]))
    kiem_tra_chia_theo_nguoi(pd.Series([1, 2, 3]), pd.Series([4, 5]))  # không lỗi


@pytest.mark.parametrize("k", [None, 5])
def test_gop_pipeline_tuyen_tinh_khop_decision_function(k):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 561)) * rng.uniform(0.5, 3, 561) + rng.normal(size=561)
    y = rng.integers(0, 3, 200)
    pipe = tao_pipeline(k).fit(x, y)
    w, b = gop_pipeline_tuyen_tinh(pipe)
    assert np.allclose(x @ w + b, pipe.decision_function(x), atol=1e-8)
