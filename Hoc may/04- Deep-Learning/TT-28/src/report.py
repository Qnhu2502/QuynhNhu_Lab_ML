"""Bảng và biểu đồ tổng hợp từ các file kết quả trong reports/results."""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from .config import LABELS, REPORTS, RESULTS

TEN_HIEN_THI = {"tfidf_svc": "TF-IDF + LinearSVC", "lstm": "BiLSTM",
                "transformer_tu_xay": "Transformer tu xay", "distilbert": "DistilBERT"}


def doc_ket_qua(ten):
    kq = json.loads((RESULTS / f"{ten}.json").read_text())
    d = np.load(RESULTS / f"{ten}.npz")
    return kq, d["y_true"], d["y_pred"], (d["proba"] if "proba" in d.files else None)


def bang_ket_qua(mapping):
    rows = []
    for ten, nhan in mapping.items():
        kq = doc_ket_qua(ten)[0]
        rows.append({"Cach": nhan, "Accuracy": round(kq["accuracy"], 4), "Macro-F1": round(kq["macro_f1"], 4),
                     "Train (s)": round(kq["train_seconds"], 1), "So tham so": kq["params"],
                     "Du doan 1 tin (ms)": round(kq["predict_ms"], 2)})
    return pd.DataFrame(rows)


def ve_so_sanh(df, path=REPORTS / "so_sanh_4_cach.png"):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, cot, log in zip(axes, ["Accuracy", "Train (s)", "So tham so", "Du doan 1 tin (ms)"],
                            [False, True, True, False]):
        ax.bar(df["Cach"], df[cot], color="steelblue")
        ax.set_title(cot)
        ax.tick_params(axis="x", rotation=30)
        if log:
            ax.set_yscale("log")
    axes[0].set_ylim(df["Accuracy"].min() - 0.02, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.show()


def ve_confusion(mapping, path=REPORTS / "confusion_4x4.png"):
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for ax, (ten, nhan) in zip(axes.ravel(), mapping.items()):
        _, yt, yp, _ = doc_ket_qua(ten)
        ConfusionMatrixDisplay.from_predictions(yt, yp, display_labels=LABELS, normalize="true",
                                                values_format=".2f", cmap="Blues", colorbar=False, ax=ax)
        ax.set_title(nhan)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.show()


def cap_nham(ten, top=3):
    """Các cặp (thật -> đoán) bị nhầm nhiều nhất."""
    _, yt, yp, _ = doc_ket_qua(ten)
    cm = confusion_matrix(yt, yp)
    np.fill_diagonal(cm, 0)
    idx = np.dstack(np.unravel_index(np.argsort(-cm.ravel()), cm.shape))[0][:top]
    return pd.DataFrame([{"That": LABELS[i], "Bi doan la": LABELS[j], "So tin": int(cm[i, j])} for i, j in idx])


def ve_attention(model, vec, texts, path=REPORTS / "attention_heatmap.png"):
    """Heatmap attention của khối Transformer đầu tiên, trung bình các head."""
    import tensorflow as tf
    from .transformer_block import KhoiTransformer, TokenVaViTri
    emb = next(l for l in model.layers if isinstance(l, TokenVaViTri))
    blk = next(l for l in model.layers if isinstance(l, KhoiTransformer))
    lay_emb = tf.keras.Model(model.input, emb.output)
    vocab = vec.get_vocabulary()
    fig, axes = plt.subplots(len(texts), 1, figsize=(8, 8 * len(texts)))
    for ax, t in zip(np.atleast_1d(axes), texts):
        ids = vec(tf.constant([t]))
        pad = tf.not_equal(ids, 0)
        _, w = blk(lay_emb(ids), pad_mask=pad, return_attn=True)
        n = int(pad.numpy().sum())
        toks = [vocab[i] for i in ids.numpy()[0][:n]]
        ax.imshow(w.numpy()[0].mean(0)[:n, :n], cmap="viridis")
        ax.set_xticks(range(n)); ax.set_xticklabels(toks, rotation=90, fontsize=7)
        ax.set_yticks(range(n)); ax.set_yticklabels(toks, fontsize=7)
        ax.set_xlabel("Tu duoc chu y (key)"); ax.set_ylabel("Tu dang hoi (query)")
        ax.set_title(t[:70])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.show()


def bang_hitl(y_true, proba, nguong=(0.8, 0.9, 0.95, 0.99), tin_moi_ngay=5000, tin_moi_nguoi=400):
    """Human-in-the-loop: tin có độ tin cậy >= ngưỡng thì tự động, còn lại chuyển biên tập viên."""
    conf, dung = proba.max(1), proba.argmax(1) == y_true
    rows = []
    for t in nguong:
        auto = conf >= t
        ty_le = auto.mean()
        cho_bt = (1 - ty_le) * tin_moi_ngay
        rows.append({"Nguong": t, "% tu dong": round(100 * ty_le, 1),
                     "Acc phan tu dong": round(dung[auto].mean(), 4) if auto.any() else np.nan,
                     "Acc model tren phan chuyen BT": round(dung[~auto].mean(), 4) if (~auto).any() else np.nan,
                     "Tin/ngay can BT": int(round(cho_bt)),
                     "So BT can": int(np.ceil(cho_bt / tin_moi_nguoi)),
                     # giả định biên tập viên xử lý đúng phần chuyển sang họ
                     "Acc he thong": round(dung[auto].sum() / len(dung) + (1 - ty_le), 4)})
    return pd.DataFrame(rows)
