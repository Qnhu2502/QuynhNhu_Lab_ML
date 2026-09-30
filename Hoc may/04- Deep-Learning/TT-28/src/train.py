"""Dựng mô hình Keras (LSTM, Transformer) và huấn luyện + đánh giá + lưu kết quả một chỗ."""
import json
import time

import tensorflow as tf
from tensorflow.keras import layers

from .config import (D_FF, D_MODEL, MAX_LEN, MODELS, N_BLOCKS, N_HEADS, VOCAB, set_seed)
from .evaluate import do_thoi_gian_du_doan, luu_ket_qua
from .transformer_block import KhoiTransformer, TokenVaViTri


def tao_vectorizer(texts=None, vocab=None):
    """Chỉ adapt từ điển trên TRAIN (chống leakage). Có thể nạp lại từ file vocab đã lưu."""
    vec = layers.TextVectorization(max_tokens=VOCAB, output_sequence_length=MAX_LEN, vocabulary=vocab)
    if vocab is None:
        vec.adapt(tf.data.Dataset.from_tensor_slices(texts).batch(1024))
    return vec


def luu_vocab(vec):
    (MODELS / "vocab.json").write_text(json.dumps(vec.get_vocabulary()))


def nap_vocab():
    return json.loads((MODELS / "vocab.json").read_text())


def ma_hoa(vec, data):
    return {k: vec(tf.constant(v)).numpy() for k, v in data["texts"].items()}


def _bien_dich(model, lr=5e-4):
    model.compile(tf.keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_lstm():
    set_seed()
    m = tf.keras.Sequential([
        layers.Input((MAX_LEN,), dtype="int32"),
        layers.Embedding(VOCAB, D_MODEL, mask_zero=True),
        layers.Bidirectional(layers.LSTM(64)),
        layers.Dropout(0.2), layers.Dense(4, activation="softmax")])
    return _bien_dich(m, 1e-3)


def build_transformer(use_pos=True, n_heads=N_HEADS):
    set_seed()
    ids = layers.Input((MAX_LEN,), dtype="int32")
    pad = layers.Lambda(lambda t: tf.not_equal(t, 0))(ids)
    x = TokenVaViTri(VOCAB, MAX_LEN, D_MODEL, use_pos)(ids)
    for _ in range(N_BLOCKS):
        x = KhoiTransformer(D_MODEL, n_heads, D_FF)(x, pad_mask=pad)
    x = layers.GlobalAveragePooling1D()(x, mask=pad)       # trung bình chỉ trên token thật
    x = layers.Dropout(0.2)(x)
    return _bien_dich(tf.keras.Model(ids, layers.Dense(4, activation="softmax")(x)))


def chay_keras(ten, model, vec, data, ids, epochs=8, batch=128):
    """Train (early stopping theo val) -> dự đoán test -> đo thời gian -> lưu json/npz/trọng số."""
    t0 = time.perf_counter()
    h = model.fit(ids["train"], data["y"]["train"], validation_data=(ids["val"], data["y"]["val"]),
                  epochs=epochs, batch_size=batch, verbose=2,
                  callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=2,
                                                              restore_best_weights=True)])
    train_s = time.perf_counter() - t0
    proba = model.predict(ids["test"], batch_size=512, verbose=0)
    ms = do_thoi_gian_du_doan(lambda t: model(vec(tf.constant([t])), training=False), data["texts"]["test"])
    model.save_weights(MODELS / f"{ten}.weights.h5")
    return luu_ket_qua(ten, data["y"]["test"], proba.argmax(1), train_s, model.count_params(), ms, proba,
                       extra={"val_accuracy": float(max(h.history["val_accuracy"]))})
