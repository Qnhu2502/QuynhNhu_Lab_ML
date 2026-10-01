"""Các hàm vẽ biểu đồ cho TT-24. Mỗi hàm tự lưu PNG (nếu có `luu`) rồi hiển thị."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .pca_pipeline import N_FEATURES, NGUONG_PHUONG_SAI


def _xuat(fig, luu: Path | None) -> None:
    fig.tight_layout()
    if luu is not None:
        Path(luu).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(luu, dpi=150, bbox_inches="tight")
    plt.show()


def ve_scree(pca_full, so_thanh_phan: int = 50, luu: Path | None = None) -> None:
    ty_le = pca_full.explained_variance_ratio_[:so_thanh_phan]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(1, len(ty_le) + 1), ty_le)
    ax.set_xlabel("Thành phần chính"); ax.set_ylabel("Tỉ lệ phương sai")
    ax.set_title(f"Scree plot ({so_thanh_phan} thành phần đầu)")
    _xuat(fig, luu)


def ve_tich_luy(pca_full, nguong_list=NGUONG_PHUONG_SAI, luu: Path | None = None) -> None:
    tich_luy = np.cumsum(pca_full.explained_variance_ratio_)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(range(1, len(tich_luy) + 1), tich_luy)
    for nguong in nguong_list:
        k = int(np.argmax(tich_luy >= nguong)) + 1
        ax.axhline(nguong, color="gray", linestyle="--", linewidth=0.7)
        ax.axvline(k, color="gray", linestyle="--", linewidth=0.7)
        ax.annotate(f"{nguong:.0%}: K={k}", (k, nguong), textcoords="offset points", xytext=(6, -12), fontsize=8)
    ax.set_xlabel("Số thành phần"); ax.set_ylabel("Phương sai tích luỹ")
    ax.set_title("Phương sai tích luỹ (mốc 80/90/95/99%)")
    _xuat(fig, luu)


def ve_danh_doi(df_cv, k_ngot: int, dung_sai: float, luu: Path | None = None) -> None:
    """Accuracy CV (±1 độ lệch chuẩn giữa các fold) và thời gian fit theo K, đánh dấu điểm ngọt."""
    baseline = df_cv.loc[df_cv["K"].idxmax(), "cv_accuracy"]
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(df_cv["K"], df_cv["cv_accuracy"], marker="o", color="tab:blue", label="Accuracy CV (theo người)")
    ax1.fill_between(df_cv["K"], df_cv["cv_accuracy"] - df_cv["cv_accuracy_std"],
                     df_cv["cv_accuracy"] + df_cv["cv_accuracy_std"], color="tab:blue", alpha=0.15)
    ax1.axhline(baseline - dung_sai, color="tab:blue", linestyle=":", linewidth=1,
                label=f"Ngưỡng: accuracy 561 chiều − {dung_sai:.0%}")
    ax1.axvline(k_ngot, color="green", linestyle="--", label=f"Điểm ngọt K={k_ngot}")
    ax1.set_xscale("log"); ax1.set_xticks(list(df_cv["K"])); ax1.set_xticklabels(list(df_cv["K"]), rotation=60, fontsize=8)
    ax1.minorticks_off()
    ax1.set_xlabel("Số chiều K (thang log)"); ax1.set_ylabel("Accuracy", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(df_cv["K"], df_cv["thoi_gian_fit_s"], marker="s", color="tab:red", label="Thời gian fit")
    ax2.set_ylabel("Thời gian fit (giây / fold)", color="tab:red")
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=8)
    ax1.set_title("Đánh đổi số chiều vs accuracy và thời gian (CV theo người trên train)")
    _xuat(fig, luu)


def ve_scatter_hai_bang(emb_trai, y_trai, emb_phai, y_phai, tieu_de_trai: str, tieu_de_phai: str,
                        nhan_truc: tuple[str, str] = ("PC1", "PC2"), luu: Path | None = None) -> None:
    """Hai scatter 2D cạnh nhau, cùng bảng màu theo lớp (dùng cho train/test và PCA/t-SNE)."""
    ten_lop = sorted(set(np.asarray(y_trai)))
    mau = dict(zip(ten_lop, plt.cm.tab10(range(len(ten_lop)))))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, emb, y, tieu_de in ((axes[0], emb_trai, y_trai, tieu_de_trai), (axes[1], emb_phai, y_phai, tieu_de_phai)):
        y = np.asarray(y)
        for lop in ten_lop:
            m = y == lop
            ax.scatter(emb[m, 0], emb[m, 1], s=8, alpha=0.45, color=mau[lop], label=lop)
        ax.set_xlabel(nhan_truc[0]); ax.set_ylabel(nhan_truc[1]); ax.set_title(tieu_de)
    axes[0].legend(markerscale=2, fontsize=8)
    _xuat(fig, luu)


def ve_sai_so_tai_tao(k_list, mse_train, mse_test, luu: Path | None = None) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(k_list, mse_train, marker="o", label="Train")
    ax.plot(k_list, mse_test, marker="s", label="Test (người chưa thấy)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Số thành phần K"); ax.set_ylabel("MSE tái tạo (đặc trưng đã chuẩn hoá)")
    ax.set_title("Sai số tái tạo theo K"); ax.legend()
    _xuat(fig, luu)


def ve_confusion(cm, ten_lop, tieu_de: str, luu: Path | None = None) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(ten_lop))); ax.set_xticklabels(ten_lop, rotation=45, ha="right")
    ax.set_yticks(range(len(ten_lop))); ax.set_yticklabels(ten_lop)
    nguong = cm.max() / 2
    for i in range(len(ten_lop)):
        for j in range(len(ten_lop)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > nguong else "black")
    ax.set_xlabel("Dự đoán"); ax.set_ylabel("Thật"); ax.set_title(tieu_de)
    _xuat(fig, luu)
