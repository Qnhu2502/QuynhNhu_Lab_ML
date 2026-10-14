import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from chunking import split_cv, windows

WEIGHTS = {'kinh_nghiem': 0.5, 'ky_nang': 0.3, 'du_an': 0.15, 'hoc_van': 0.05}
OTHER_W = 0.1     # trọng số cho phần 'khac'


def load_model(name):
    return SentenceTransformer(name)


def encode(model, texts):
    # normalize=True: tích vô hướng == cosine; batch_size: tránh encode từng CV một
    return model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)


# Mọi hàm score_* trả về ma trận (n_jd, n_cv) để chấm nhiều JD cùng lúc.
def score_tfidf(jds, cvs):
    X = TfidfVectorizer(ngram_range=(1, 2), min_df=2).fit_transform(jds + cvs)
    return cosine_similarity(X[:len(jds)], X[len(jds):])


def score_full(model, jds, cvs):
    # Baseline embedding: đưa nguyên CV vào model (phần vượt max_seq_length bị bỏ)
    return encode(model, jds) @ encode(model, cvs).T


def score_chunk(model, jds, cvs, size=None, weights=WEIGHTS):
    # Cửa sổ tính theo từ, ~0.6 * max_seq_length để chừa chỗ cho subword
    size = size or int(model.max_seq_length * 0.6)
    meta, texts = [], []
    for i, cv in enumerate(cvs):
        for sec, content in split_cv(cv).items():
            for ch in windows(content, size):
                meta.append((i, sec)); texts.append(ch)

    # Encode toàn bộ đoạn của mọi CV trong MỘT lần gọi
    sims = encode(model, jds) @ encode(model, texts).T            # (n_jd, n_chunk)

    out = np.zeros((len(jds), len(cvs)))
    for j in range(len(jds)):
        best = {}                                                 # (cv, mục) -> điểm đoạn tốt nhất
        for key, s in zip(meta, sims[j]):
            best[key] = max(best.get(key, -1.0), s)
        num, den = np.zeros(len(cvs)), np.zeros(len(cvs))
        for (i, sec), s in best.items():
            w = weights.get(sec, OTHER_W)
            num[i] += w * s; den[i] += w
        # Chia cho tổng trọng số các mục CÓ MẶT: CV thiếu mục không bị phạt oan
        out[j] = num / np.maximum(den, 1e-9)
    return out
