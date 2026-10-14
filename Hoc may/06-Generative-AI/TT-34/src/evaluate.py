import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import ndcg_score


def ndcg_at_k(scores, labels, k=10):
    """scores, labels: (n_jd, n_cv). Trung bình NDCG@k qua các JD."""
    return float(np.mean([ndcg_score([l], [s], k=k) for s, l in zip(scores, labels)]))


def precision_at_k(scores, labels, k=10, thr=2):
    """Tỷ lệ CV trong top-k có nhãn vàng >= thr (coi là phù hợp)."""
    return float(np.mean([(l[np.argsort(s)[::-1][:k]] >= thr).mean() for s, l in zip(scores, labels)]))


def synonym_pairs(encode_fn, pairs, corpus, k=10, hi=0.5, lo=0.1):
    """Giữ các cặp embedding giống (>=hi) mà TF-IDF gần như không thấy (<=lo)."""
    a, b = [p[0] for p in pairs], [p[1] for p in pairs]
    emb = (encode_fn(a) * encode_fn(b)).sum(1)
    tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2).fit(corpus)   # cùng cấu hình baseline
    tfv = np.asarray(tf.transform(a).multiply(tf.transform(b)).sum(1)).ravel()  # hàng đã chuẩn hoá L2
    df = pd.DataFrame({'a': a, 'b': b, 'embedding': emb, 'tfidf': tfv})
    return df[(df.embedding >= hi) & (df.tfidf <= lo)].sort_values('embedding', ascending=False).head(k)


def bias_check(score_fn, jds, cvs, name_pairs):
    """Gắn tên nam / tên nữ vào đầu cùng một CV (nội dung giữ nguyên), so điểm và thứ hạng."""
    tag = lambda k: [f"Name: {name_pairs[i % len(name_pairs)][k]}\n{cv}" for i, cv in enumerate(cvs)]
    a, b = score_fn(jds, tag(0)), score_fn(jds, tag(1))            # (n_jd, n_cv)
    rank = lambda s: (-s).argsort(1).argsort(1) + 1
    df = pd.DataFrame({'diem_nam': a.mean(0), 'diem_nu': b.mean(0), 'chenh_lech': (b - a).mean(0),
                       'doi_hang_tb': np.abs(rank(b) - rank(a)).mean(0)})
    p = wilcoxon((b - a).ravel()).pvalue if np.any(a != b) else 1.0
    summary = {'chenh_lech_tb': float((b - a).mean()),
               'chenh_lech_tuyet_doi_tb': float(np.abs(b - a).mean()),
               'so_voi_do_lech_chuan_cua_diem': float(np.abs(b - a).mean() / a.std()),
               'doi_hang_tb': float(df.doi_hang_tb.mean()),
               'p_wilcoxon': float(p)}
    return df, summary
