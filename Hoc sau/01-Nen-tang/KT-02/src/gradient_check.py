"""Kiểm chứng gradient: so gradient giải tích với gradient số (sai phân trung tâm)."""
import numpy as np


def kiem_chung_gradient(f, x, grad_giai_tich, eps=1e-5, ten=""):
    """f() trả loss và đọc x tại chỗ; x bị sửa tạm rồi khôi phục. Sai lệch phải < 1e-7."""
    grad_so = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        goc = x[i]
        x[i] = goc + eps; f_cong = f()
        x[i] = goc - eps; f_tru = f()
        x[i] = goc
        grad_so[i] = (f_cong - f_tru) / (2 * eps)
        it.iternext()
    sai_lech = np.abs(grad_so - grad_giai_tich).max()
    print(f"{ten:>3} sai lệch lớn nhất: {sai_lech:.2e} -> {'ĐÚNG' if sai_lech < 1e-7 else 'SAI'}")
    return sai_lech


def check_mlp(net, X, y, eps=1e-5):
    """Kiểm chứng toàn bộ W, b của một MLP; trả sai lệch lớn nhất."""
    net.loss(X, y)
    gW, gb = net.backward(y)                           # tính gradient giải tích trước
    f = lambda: net.loss(X, y)
    params = [(f"W{k+1}", p, g) for k, (p, g) in enumerate(zip(net.W, gW))] \
           + [(f"b{k+1}", p, g) for k, (p, g) in enumerate(zip(net.b, gb))]
    return max(kiem_chung_gradient(f, p, g, eps, ten) for ten, p, g in params)
