import copy
import numpy as np, torch
from sklearn.metrics import average_precision_score

def predict(model, loader, device, eval_mode=True):
    # eval_mode=False chỉ dùng cho thí nghiệm "quên model.eval()"
    model.train(not eval_mode)
    ps, ys = [], []
    with torch.no_grad():
        for xb, yb in loader:
            ps.append(torch.sigmoid(model(xb.to(device))).cpu())   # sigmoid ở bước predict
            ys.append(yb)
    return torch.cat(ps).numpy().ravel(), torch.cat(ys).numpy().ravel()

def fit(model, train_loader, val_loader, optimizer, criterion, device,
        epochs=100, patience=10, zero_grad=True):
    hist = {"train_loss": [], "val_loss": [], "val_prauc": []}
    best, best_state, wait = -1.0, None, 0
    for ep in range(epochs):
        model.train()                                    # bật Dropout / BatchNorm train
        tl, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            if zero_grad:
                optimizer.zero_grad()                    # nếu thiếu: gradient cộng dồn qua các batch
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            tl += loss.item() * len(xb); n += len(xb)

        model.eval()                                     # tắt Dropout, BatchNorm dùng thống kê chạy
        vl, vn, ps, ys = 0.0, 0, [], []
        with torch.no_grad():                            # không tính gradient khi đánh giá
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                vl += criterion(out, yb).item() * len(xb); vn += len(xb)
                ps.append(torch.sigmoid(out).cpu()); ys.append(yb.cpu())
        p = np.nan_to_num(torch.cat(ps).numpy().ravel(), nan=0.5)    # tránh lỗi khi loss nổ thành NaN
        pr = average_precision_score(torch.cat(ys).numpy().ravel(), p)
        hist["train_loss"].append(tl / n); hist["val_loss"].append(vl / vn); hist["val_prauc"].append(pr)

        if pr > best:                                    # early stopping thủ công theo PR-AUC val
            best, wait, best_state = pr, 0, copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)                # trả về trọng số tốt nhất (checkpoint)
    return hist
