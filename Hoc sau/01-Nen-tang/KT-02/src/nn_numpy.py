"""MLP (sigmoid + MSE) bằng NumPy thuần: forward, backward theo quy tắc dây chuyền, SGD."""
import numpy as np

XOR_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
XOR_Y = np.array([[0], [1], [1], [0]], dtype=float)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class MLP:
    """sizes=[2,4,1] là mạng 2->4->1; sizes=[2,1] là mạng KHÔNG có tầng ẩn."""

    def __init__(self, sizes, seed=0):
        rng = np.random.default_rng(seed)
        self.W = [rng.standard_normal((i, o)) for i, o in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(o) for o in sizes[1:]]

    def forward(self, X):
        self.A = [X]                                   # lưu activation từng tầng cho backward
        for W, b in zip(self.W, self.b):
            self.A.append(sigmoid(self.A[-1] @ W + b))
        return self.A[-1]

    def loss(self, X, y):
        return np.mean((self.forward(X) - y) ** 2)

    def backward(self, y):
        """Gọi sau forward/loss. Trả về (gW, gb) cùng cấu trúc với (W, b)."""
        gW, gb = [None] * len(self.W), [None] * len(self.b)
        a = self.A[-1]
        d = 2 * (a - y) / y.size * a * (1 - a)         # dL/dz tầng cuối (nhớ chia N)
        for l in reversed(range(len(self.W))):
            gW[l] = self.A[l].T @ d
            gb[l] = d.sum(axis=0)
            if l > 0:                                  # lan sai số ngược về tầng trước
                a = self.A[l]
                d = (d @ self.W[l].T) * a * (1 - a)
        return gW, gb

    def step(self, gW, gb, lr):
        for p, g in zip(self.W + self.b, gW + gb):     # cập nhật tại chỗ
            p -= lr * g

    def fit(self, X, y, lr=2.0, epochs=10000):
        hist = np.empty(epochs)
        for e in range(epochs):
            hist[e] = self.loss(X, y)
            self.step(*self.backward(y), lr)
        return hist
