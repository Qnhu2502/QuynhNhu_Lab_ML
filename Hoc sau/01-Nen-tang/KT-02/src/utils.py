"""Tiện ích PyTorch dùng chung cho notebook 02 và 03 (dữ liệu, dựng mạng, huấn luyện, đo gradient)."""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ACTS = {
    "Sigmoid": nn.Sigmoid, "Tanh": nn.Tanh, "ReLU": nn.ReLU,
    "LeakyReLU": lambda: nn.LeakyReLU(0.01), "ELU": nn.ELU, "GELU": nn.GELU, "SiLU": nn.SiLU,
}

INITS = {
    "zeros": nn.init.zeros_,
    "random_nho": lambda w: nn.init.normal_(w, 0.0, 0.01),
    "xavier": nn.init.xavier_normal_,                                   # Var = 2/(n_in+n_out)
    "he": lambda w: nn.init.kaiming_normal_(w, nonlinearity="relu"),    # Var = 2/n_in
}


def load_mnist(root="data"):
    """Trả (Xtr, ytr, Xte, yte); toàn bộ nằm sẵn trên DEVICE để huấn luyện nhanh."""
    from torchvision import datasets
    out = []
    for is_train in (True, False):
        d = datasets.MNIST(root, train=is_train, download=True)
        x = ((d.data.float() / 255 - 0.1307) / 0.3081).view(len(d), -1)
        out += [x.to(DEVICE), d.targets.to(DEVICE)]
    return out


def make_mlp(widths, act, init="he", bn=False, seed=0):
    """Dựng MLP; cùng seed => cùng bộ số ngẫu nhiên khi chỉ đổi activation (so sánh công bằng)."""
    torch.manual_seed(seed)
    layers = []
    for i, (a, b) in enumerate(zip(widths[:-1], widths[1:])):
        lin = nn.Linear(a, b)
        INITS[init](lin.weight)
        nn.init.zeros_(lin.bias)
        layers.append(lin)
        if i < len(widths) - 2:
            if bn:
                layers.append(nn.BatchNorm1d(b))
            layers.append(act())
    return nn.Sequential(*layers)


def train(model, data, epochs, lr, bs=128, momentum=0.9, seed=0):
    """SGD + cross-entropy. Trả (lịch sử từng epoch, số giây). Dừng sớm nếu loss không hữu hạn."""
    Xtr, ytr, Xte, yte = data
    model.to(DEVICE)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    gen = torch.Generator().manual_seed(seed)          # cùng thứ tự batch cho mọi cấu hình
    hist, t0 = [], time.perf_counter()
    for ep in range(1, epochs + 1):
        model.train()
        total = torch.zeros((), device=DEVICE)
        for idx in torch.randperm(len(Xtr), generator=gen).to(DEVICE).split(bs):
            loss = F.cross_entropy(model(Xtr[idx]), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.detach() * len(idx)
        model.eval()
        with torch.no_grad():
            acc = (model(Xte).argmax(1) == yte).float().mean().item()
        hist.append(dict(epoch=ep, loss=total.item() / len(Xtr), acc=acc))
        if not np.isfinite(hist[-1]["loss"]):
            break
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    return hist, time.perf_counter() - t0


def layer_grad_norms(model, data, n_batches=10, bs=256, seed=0):
    """||grad W|| của từng Linear (từ tầng gần đầu vào đến tầng cuối), trung bình qua vài batch."""
    Xtr, ytr = data[0], data[1]
    model.to(DEVICE).train()
    linears = [m for m in model if isinstance(m, nn.Linear)]
    gen = torch.Generator().manual_seed(seed)
    norms = np.zeros(len(linears))
    for _ in range(n_batches):
        idx = torch.randint(0, len(Xtr), (bs,), generator=gen).to(DEVICE)
        model.zero_grad()
        F.cross_entropy(model(Xtr[idx]), ytr[idx]).backward()
        norms += [l.weight.grad.norm().item() for l in linears]
    return norms / n_batches


@torch.no_grad()
def dead_fraction(model, X):
    """Tỉ lệ neuron 'chết': tiền kích hoạt z <= 0 với MỌI mẫu (đúng cho ReLU và LeakyReLU)."""
    dead = []
    hooks = [m.register_forward_pre_hook(lambda mod, inp: dead.append((inp[0] <= 0).all(0)))
             for m in model if isinstance(m, (nn.ReLU, nn.LeakyReLU))]
    model.eval()
    model(X)
    for h in hooks:
        h.remove()
    return torch.cat(dead).float().mean().item()


@torch.no_grad()
def activations_per_layer(model, X):
    """Danh sách activation (đã flatten) sau mỗi ReLU, theo thứ tự tầng."""
    outs = []
    hooks = [m.register_forward_hook(lambda mod, i, o: outs.append(o.flatten().cpu()))
             for m in model if isinstance(m, nn.ReLU)]
    model.eval()
    model(X)
    for h in hooks:
        h.remove()
    return outs
