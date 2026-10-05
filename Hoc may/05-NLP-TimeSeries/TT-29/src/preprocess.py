"""Tải và làm sạch dữ liệu 20 Newsgroups cho bài toán phân loại ticket."""
from sklearn.datasets import fetch_20newsgroups

RANDOM_STATE = 42


def tai_20newsgroups(bo_remove: bool = True):
    """bo_remove=True -> remove headers/footers/quotes (bản SẠCH, dùng chính thức).

    QUAN TRỌNG: bo_remove=False chỉ dùng để minh chứng rò rỉ (xem Bước 1 notebook),
    KHÔNG dùng để train model thật.
    """
    remove = ('headers', 'footers', 'quotes') if bo_remove else ()
    train = fetch_20newsgroups(subset='train', remove=remove, random_state=RANDOM_STATE)
    test = fetch_20newsgroups(subset='test', remove=remove, random_state=RANDOM_STATE)
    return train, test
