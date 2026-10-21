# KT-02 — BÊN TRONG MẠNG NƠ-RON
## Tự code backpropagation và so sánh 7 hàm kích hoạt

| | |
|---|---|
| 🎓 **Khoá** | HỌC SÂU · [Buổi 2](https://github.com/TruongTanNghia/Training-Deep-Learning/tree/main/Buoi-02-MangNoron-ChuyenSau) |
| 🧠 **Nhóm** | Nền tảng · Hiểu bản chất |
| 🔧 **Nội dung** | 7 activation · Backprop thủ công · Khởi tạo trọng số |
| 🏭 **Lĩnh vực** | Nghiên cứu · Nền tảng cho mọi bài sau |
| ⏱ **Thời lượng** | 8–12 giờ |
| 📈 **Độ khó** | ⭐⭐⭐⭐ |

---

## 1. BÀI NÀY KHÁC MỌI BÀI KHÁC

Đây là bài **không có sản phẩm kinh doanh** — sản phẩm là **sự hiểu biết**. Mọi kiến
trúc từ KT-04 trở đi (ResNet, Transformer, Diffusion) đều đứng trên 3 thứ ở bài này:

```
   ① Vì sao cần PHI TUYẾN     → không có nó, 100 tầng = 1 tầng
   ② Backprop hoạt động ra sao → quy tắc dây chuyền lan sai số ngược
   ③ Khởi tạo trọng số quan trọng thế nào → sai là mạng không học được
```

> ⚠️ Nếu bỏ qua bài này, bạn vẫn train được model — nhưng khi model **không** học
> được, bạn sẽ không biết vì sao và chỉ có thể thử mò.

---

## 2. "BÀI TOÁN THỰC TẾ" CỦA BÀI NÀY

```
   Tình huống thật hay gặp trong công việc:

   "Loss = NaN sau 3 epoch"                  → vì sao? sửa thế nào?
   "Loss đứng yên không giảm"                → vanishing gradient? lr sai? khởi tạo sai?
   "Train accuracy 99%, val accuracy 60%"    → overfit — nhưng cơ chế nào gây ra?
   "Đổi ReLU sang sigmoid thì mạng chết"     → vì sao?

   → Người hiểu bài này chẩn đoán được trong 10 phút.
     Người không hiểu sẽ thử mò cả ngày.
```

---

## 3. DỮ LIỆU

| Giai đoạn | Dữ liệu | Vì sao |
|-----------|---------|--------|
| **Phần 1** | XOR (4 điểm) | Nhỏ nhất có thể, đủ chứng minh cần tầng ẩn |
| **Phần 2** | `make_moons` (1.000 điểm) | Vẽ được ranh giới quyết định trên 2D |
| **Phần 3** | MNIST (60.000 ảnh) | Đủ lớn để thấy khác biệt giữa các activation |

Tất cả đều có sẵn (`sklearn.datasets`, `torchvision.datasets`) — không cần tải thủ công.

---

## 4. HƯỚNG ĐI ĐÚNG

### 4.1. PHẦN 1 — Tự code mạng bằng NumPy thuần

```
   Kiến trúc 2 → 4 → 1 giải XOR, KHÔNG dùng PyTorch/TensorFlow.
   Chỉ NumPy. Tự viết forward, tự viết backward.
```

```python
import numpy as np

# --- FORWARD ---
z1 = X  @ W1 + b1;  a1 = 1 / (1 + np.exp(-z1))
z2 = a1 @ W2 + b2;  a2 = 1 / (1 + np.exp(-z2))
loss = np.mean((a2 - y) ** 2)

# --- BACKWARD (quy tắc dây chuyền) ---
d_a2 = 2 * (a2 - y) / len(y)          # ∂L/∂a2
d_z2 = d_a2 * a2 * (1 - a2)           # qua đạo hàm sigmoid: σ(1−σ)
d_W2 = a1.T @ d_z2
d_b2 = d_z2.sum(axis=0)

d_a1 = d_z2 @ W2.T                    # sai số "chảy ngược" về tầng ẩn
d_z1 = d_a1 * a1 * (1 - a1)
d_W1 = X.T @ d_z1
d_b1 = d_z1.sum(axis=0)

# --- UPDATE ---
for p, g in [(W1, d_W1), (b1, d_b1), (W2, d_W2), (b2, d_b2)]:
    p -= LR * g
```

### 4.2. ⭐ Kiểm chứng gradient — kỹ thuật chuyên nghiệp

Cách duy nhất để **chắc chắn** backprop tự viết là đúng:

```python
def kiem_chung_gradient(f, x, grad_giai_tich, eps=1e-5):
    """So gradient giải tích với gradient số. Sai lệch phải < 1e-7."""
    grad_so = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:
        i = it.multi_index
        goc = x[i]
        x[i] = goc + eps;  f_cong = f()
        x[i] = goc - eps;  f_tru  = f()
        x[i] = goc
        grad_so[i] = (f_cong - f_tru) / (2 * eps)
        it.iternext()
    sai_lech = np.abs(grad_so - grad_giai_tich).max()
    print(f"Sai lệch lớn nhất: {sai_lech:.2e} → {'ĐÚNG' if sai_lech < 1e-7 else 'SAI'}")
```

### 4.3. PHẦN 2 — So sánh 7 hàm kích hoạt

| # | Hàm | Công thức | Đạo hàm tối đa | Vấn đề |
|---|-----|-----------|---------------|--------|
| 1 | Sigmoid | 1/(1+e⁻ᶻ) | **0,25** | Vanishing gradient |
| 2 | Tanh | (eᶻ−e⁻ᶻ)/(eᶻ+e⁻ᶻ) | 1,0 | Vẫn tiêu biến |
| 3 | **ReLU** | max(0,z) | 1 (vùng dương) | Dying ReLU |
| 4 | LeakyReLU | max(0.01z, z) | 1 | Chữa dying ReLU |
| 5 | ELU | z nếu z>0, α(eᶻ−1) nếu z≤0 | 1 | Chậm hơn ReLU |
| 6 | GELU | z·Φ(z) | ~1 | ⭐ Chuẩn của Transformer |
| 7 | SiLU/Swish | z·σ(z) | ~1 | Dùng trong EfficientNet, YOLO |

### 4.4. ⭐ Chứng minh vanishing gradient bằng số

```
   Đạo hàm sigmoid lớn nhất = 0,25.
   Backprop nhân dồn qua từng tầng:

     10 tầng sigmoid → gradient ≈ 0,25¹⁰ ≈ 0,00000095
     → tầng đầu gần như KHÔNG nhận được tín hiệu học

   ReLU có đạo hàm = 1 ở vùng dương → nhân bao nhiêu lần vẫn = 1 → hết tiêu biến.
```

**Thí nghiệm bắt buộc:** xây mạng 15 tầng, đo **độ lớn gradient tại từng tầng** với
sigmoid vs ReLU, vẽ biểu đồ theo chiều sâu.

```python
for ten, tham_so in model.named_parameters():
    if tham_so.grad is not None:
        print(f"{ten}: |grad| = {tham_so.grad.norm().item():.2e}")
```

### 4.5. PHẦN 3 — Khởi tạo trọng số

```
   Khởi tạo TẤT CẢ W = 0 (hoặc mọi neuron cùng giá trị):
     → mọi neuron trong 1 tầng tính ra CÙNG một số
     → nhận CÙNG gradient → cập nhật GIỐNG HỆT nhau mãi mãi
     → 128 neuron nhưng chỉ học được đúng 1 thứ ("đối xứng không phá vỡ được")

   Đúng cách:
     Xavier/Glorot  — cho tanh, sigmoid:  Var(W) = 2/(n_in + n_out)
     He/Kaiming     — cho ReLU:           Var(W) = 2/n_in     ⭐ mặc định hiện nay
```

---

## 5. CÁC BƯỚC THỰC HIỆN

```
   PHẦN 1 — NumPy thuần (4h)
   ☐ 1. Cài mạng 2→4→1 giải XOR, chỉ dùng NumPy
   ☐ 2. ⭐ Kiểm chứng gradient — phải đạt sai lệch < 1e-7
   ☐ 3. Train 10.000 epoch, vẽ loss → chỉ ra "khúc gãy" kinh điển
   ☐ 4. Vẽ ranh giới quyết định, xác nhận giải đúng XOR
   ☐ 5. ⭐ Bỏ tầng ẩn (chỉ 2→1) → chứng minh KHÔNG giải được XOR

   PHẦN 2 — Activation (4h)
   ☐ 6. Vẽ đồ thị 7 hàm và đạo hàm của chúng trên cùng hệ trục
   ☐ 7. Train MNIST với từng activation (giữ nguyên mọi thứ khác)
        → bảng: accuracy · số epoch tới hội tụ · thời gian
   ☐ 8. ⭐ Mạng 15 tầng: đo |gradient| theo tầng với sigmoid vs ReLU
        → biểu đồ log-scale chứng minh vanishing gradient
   ☐ 9. ⭐ Tạo hiện tượng DYING RELU: đặt lr lớn → đếm % neuron chết
        (neuron chết = luôn xuất 0 với mọi đầu vào) → LeakyReLU cứu được không?

   PHẦN 3 — Khởi tạo (3h)
   ☐ 10. So sánh 4 cách khởi tạo: zeros · random nhỏ · Xavier · He
         → biểu đồ loss theo epoch
   ☐ 11. ⭐ Khởi tạo zeros → in trọng số 5 neuron sau vài epoch, chứng minh chúng GIỐNG HỆT
   ☐ 12. Vẽ phân phối activation qua từng tầng với từng cách khởi tạo
   ☐ 13. Thêm BatchNorm → có làm mạng bớt nhạy với khởi tạo không?
   ☐ 14. ✍️ Viết CẨM NANG CHẨN ĐOÁN: gặp lỗi X → nguyên nhân có thể → cách sửa
```

---

## 6. TIÊU CHÍ HOÀN THÀNH

```
   ☐ ⭐ Backprop NumPy vượt kiểm chứng gradient (sai lệch < 1e-7)
   ☐ Mạng XOR đạt 100% (dự đoán ≈ [0,1,1,0])
   ☐ Chứng minh mạng không tầng ẩn KHÔNG giải được XOR
   ☐ Có đồ thị 7 activation + đạo hàm
   ☐ Có bảng so sánh 7 activation trên MNIST
   ☐ ⭐ Có biểu đồ |gradient| theo tầng chứng minh vanishing gradient
   ☐ ⭐ Có thí nghiệm dying ReLU + tỉ lệ neuron chết
   ☐ Có so sánh 4 cách khởi tạo + bằng chứng "đối xứng không phá vỡ được"
   ☐ ⭐ Có CẨM NANG CHẨN ĐOÁN ≥ 6 lỗi thường gặp
```

---

## 7. CẠM BẪY

| Cạm bẫy | Hậu quả |
|---------|---------|
| Không kiểm chứng gradient | Backprop sai mà không biết, debug hàng giờ |
| `eps` quá nhỏ khi kiểm chứng (1e-10) | Sai số làm tròn của số thực lấn át |
| Quên chia N trong đạo hàm loss | Gradient tỉ lệ với batch size |
| So sánh activation mà đổi cả lr | Không biết cải thiện do đâu |
| Kết luận "ReLU luôn tốt nhất" | GELU thắng trong Transformer, SiLU trong CNN hiện đại |

---

## 8. SẢN PHẨM NỘP & MỞ RỘNG

```
KT-02-NNInternals-<HoTen>/
├── README.md          ← ⭐ có CẨM NANG CHẨN ĐOÁN
├── notebooks/{01_backprop_numpy.ipynb, 02_activations.ipynb, 03_initialization.ipynb}
├── src/{nn_numpy.py, gradient_check.py}
├── reports/{7_activations.png, vanishing_gradient.png, dying_relu.csv, init_comparison.png}
└── requirements.txt
```

**Mở rộng:**
1. Tự cài **Adam** bằng NumPy → so sánh với SGD trên cùng mạng XOR
2. Tự cài **BatchNorm** (cả forward lẫn backward) → kiểm chứng gradient
3. Vẽ **bề mặt loss** 3D của mạng 2 tham số → nhìn thấy cực tiểu cục bộ

**Tham khảo:** [Buổi 2 — Mạng nơ-ron chuyên sâu](https://github.com/TruongTanNghia/Training-Deep-Learning/tree/main/Buoi-02-MangNoron-ChuyenSau/Tai-Lieu)
