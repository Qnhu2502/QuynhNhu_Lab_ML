"""Các khối tự xây: positional encoding, embedding, khối Transformer Encoder."""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


def ma_hoa_vi_tri(max_len, d_model):
    """Positional encoding sin/cos: attention không biết thứ tự nên phải CỘNG vị trí vào embedding."""
    pos = np.arange(max_len)[:, None]
    i = np.arange(d_model)[None, :]
    ang = pos / np.power(10000, (2 * (i // 2)) / d_model)
    ang[:, 0::2] = np.sin(ang[:, 0::2])
    ang[:, 1::2] = np.cos(ang[:, 1::2])
    return tf.constant(ang[None], dtype=tf.float32)


class TokenVaViTri(layers.Layer):
    """Embedding từ (+ positional encoding nếu use_pos). use_pos=False dùng cho thí nghiệm bỏ vị trí."""

    def __init__(self, vocab, max_len, d_model, use_pos=True, **kw):
        super().__init__(**kw)
        self.emb = layers.Embedding(vocab, d_model)
        self.use_pos = use_pos
        self.pos = ma_hoa_vi_tri(max_len, d_model)
        self.scale = d_model ** 0.5      # nhân sqrt(d) như bài báo gốc để vị trí không lấn át embedding

    def call(self, ids):
        x = self.emb(ids) * self.scale
        return x + self.pos[:, :tf.shape(x)[1]] if self.use_pos else x


class KhoiTransformer(layers.Layer):
    """Multi-head attention -> residual+LayerNorm -> Feed-forward -> residual+LayerNorm."""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, **kw):
        super().__init__(**kw)
        # key_dim = d_model // n_heads để số tham số không đổi khi khảo sát số head
        self.att = layers.MultiHeadAttention(num_heads=n_heads, key_dim=d_model // n_heads)
        self.ffn = tf.keras.Sequential([layers.Dense(d_ff, activation="relu"), layers.Dense(d_model)])
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.do1, self.do2 = layers.Dropout(dropout), layers.Dropout(dropout)

    def call(self, x, pad_mask=None, training=False, return_attn=False):
        # pad_mask (B,T): True ở token thật -> token thật không chú ý vào padding
        att_mask = None if pad_mask is None else tf.tile(pad_mask[:, None, :], [1, tf.shape(x)[1], 1])
        a, w = self.att(x, x, attention_mask=att_mask, return_attention_scores=True)
        x = self.ln1(x + self.do1(a, training=training))
        x = self.ln2(x + self.do2(self.ffn(x), training=training))
        return (x, w) if return_attn else x
