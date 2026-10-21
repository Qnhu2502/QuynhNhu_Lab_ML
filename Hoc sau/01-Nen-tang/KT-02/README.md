# KT-02 - Bên trong mạng nơ-ron

Tự code backpropagation bằng NumPy và so sánh 7 hàm kích hoạt, các cách khởi tạo trọng số.

## Cấu trúc

```
KT-02-NNInternals-HoTen/
├── README.md
├── notebooks/{01_backprop_numpy, 02_activations, 03_initialization}.ipynb
├── src/{nn_numpy.py, gradient_check.py, utils.py}
├── reports/   (do notebook tự sinh ra)
└── requirements.txt
```

Cách chạy: `pip install -r requirements.txt`, rồi chạy lần lượt 3 notebook (mở từ thư mục `notebooks/`).
`src/utils.py` chứa phần PyTorch dùng chung (nạp MNIST, dựng MLP, vòng train, đo gradient, đếm neuron chết) để hai notebook sau không lặp code.

Quy ước: các ô `[...]` trong phần Nhận xét là số liệu điền sau khi chạy. Phần giải thích cơ chế là kết luận cần kiểm chứng bằng số liệu của chính bạn.

---

## Phần 1 - NumPy thuần (XOR)

**Bước 1. Mạng 2-4-1.** `src/nn_numpy.py`: lớp `MLP` có `forward`, `backward` (quy tắc dây chuyền), `step` (SGD). Sigmoid + MSE, đạo hàm loss chia cho số phần tử để gradient không phụ thuộc kích thước batch.

**Bước 2. Kiểm chứng gradient.** `src/gradient_check.py`: sai phân trung tâm `(f(x+eps) - f(x-eps)) / 2eps`, `eps = 1e-5`, so với gradient giải tích cho mọi `W`, `b`.
- Nhận xét: sai lệch lớn nhất `[~1e-12]`, nhỏ hơn ngưỡng 1e-7, nên backprop đúng. `eps = 1e-5` cân bằng giữa sai số xấp xỉ (lớn khi eps lớn) và sai số làm tròn (lớn khi eps cỡ 1e-10).

**Bước 3. Train 10.000 epoch.** Ra `reports/xor_loss.png`.
- Nhận xét: loss đứng ở cao nguyên gần 0,25 (mạng đang xuất ~0,5 cho mọi điểm) rồi tụt đột ngột ở khoảng epoch `[...]`. Đó là "khúc gãy": gradient quanh điểm yên ngựa rất nhỏ, mạng phải phá được đối xứng giữa các neuron ẩn mới thoát ra. Loss cuối `[...]`, dự đoán `[≈ 0,1,1,0]`.
- Có vài seed kẹt ở cực tiểu cục bộ nên notebook thử seed lần lượt đến khi đạt 100%. Seed dùng: `[...]`.

**Bước 4. Ranh giới quyết định.** Ra `reports/xor_boundary.png` (ô trái).
- Nhận xét: ranh giới cong chia 4 điểm thành 2 vùng đúng nhãn. Hai nửa mặt phẳng ghép lại được là nhờ tầng ẩn phi tuyến.

**Bước 5. Bỏ tầng ẩn (2-1).** Ra `reports/xor_boundary.png` (ô phải).
- Nhận xét: qua 20 seed, accuracy tốt nhất là `[75%]`, không seed nào đạt 100%. Mạng không tầng ẩn chỉ tạo được ranh giới tuyến tính, mà XOR không tách được bằng một đường thẳng. Đây là lý do cần phi tuyến: không có nó, 100 tầng tuyến tính vẫn chỉ tương đương 1 tầng.

---

## Phần 2 - So sánh 7 hàm kích hoạt

**Bước 6. Đồ thị 7 hàm và đạo hàm.** Ra `reports/7_activations.png`; đạo hàm lấy bằng autograd. Thêm `reports/moons_boundaries.png`: ranh giới quyết định của từng activation trên `make_moons`.
- Nhận xét: sigmoid có đạo hàm tối đa 0,25 và tiến về 0 ở hai đầu. Tanh tối đa 1 nhưng cũng bão hòa. ReLU/LeakyReLU có đạo hàm 1 ở vùng dương. GELU/SiLU trơn, đạo hàm xấp xỉ 1 ở vùng dương. ELU trơn ở vùng âm.

**Bước 7. MNIST, 7 activation.** Mạng 784-256-128-10, init He, SGD momentum 0,9, lr 0,05, 10 epoch. Cùng seed khởi tạo và cùng thứ tự batch cho mọi activation, chỉ đổi activation. Ra `reports/activation_mnist.csv` (accuracy tốt nhất, số epoch tới 97%, loss cuối, thời gian).
- Nhận xét: `[điền bảng]`. Kỳ vọng: nhóm ReLU/LeakyReLU/GELU/SiLU/ELU gần nhau về accuracy. Sigmoid hội tụ chậm nhất. ReLU nhanh nhất về thời gian vì phép tính đơn giản nhất. Chênh lệch nhỏ giữa nhóm đầu không phải kết luận "X luôn tốt nhất" (mạng nông, 1 seed). Muốn kết luận chắc thì cần chạy nhiều seed.

**Bước 8. Vanishing gradient, mạng 15 tầng.** Mỗi activation dùng init phù hợp của nó (Xavier cho sigmoid, He cho ReLU) để so sánh theo hướng "mỗi bên ở điều kiện tốt nhất". Ra `reports/vanishing_gradient.png` (log-scale), có đường tham chiếu 0,25^k.
- Nhận xét: với sigmoid, `|grad|` tầng 1 / tầng 15 là `[...]`, giảm theo cấp số nhân khi đi về đầu vào, bám đường 0,25^k (lý thuyết 0,25^14 ≈ 4e-9). Với ReLU, `|grad|` các tầng gần như phẳng (tỉ lệ `[...]`). Tầng đầu của mạng sigmoid gần như không nhận được tín hiệu học dù đã dùng init đúng cách, nên nguyên nhân nằm ở bản thân đạo hàm của sigmoid.

**Bước 9. Dying ReLU.** Mạng 784-256-256-256-10, SGD thuần (không momentum), lr ∈ {0,01; 0,1; 0,5; 1; 2; 5}, 3 epoch. Neuron chết = tiền kích hoạt `z <= 0` với mọi mẫu train. Ra `reports/dying_relu.csv`.
- Nhận xét: lr nhỏ thì tỉ lệ chết `[~0%]`; lr lớn thì một bước cập nhật quá mạnh đẩy bias/trọng số về vùng âm, neuron luôn xuất 0, gradient bằng 0 nên không tự hồi phục. Tỉ lệ chết cao nhất trước khi phân kỳ: `[...]%` tại lr `[...]`. LeakyReLU: tỉ lệ "chết" `[...]%`. Neuron LeakyReLU vẫn nhận gradient 0,01 nên về nguyên tắc còn cứu được, nhưng khi lr quá lớn thì cả hai đều phân kỳ (NaN) và LeakyReLU không cứu được.

---

## Phần 3 - Khởi tạo trọng số

Mạng 784-(256 x 5)-10, ReLU, lr 0,05, 5 epoch cho các bước 10 và 13.

**Bước 10. 4 cách khởi tạo.** zeros, random nhỏ (std 0,01), Xavier, He. Ra `reports/init_comparison.png`.
- Nhận xét: zeros: loss đứng ở ≈ 2,30 (= ln 10, đoán ngẫu nhiên), accuracy ≈ 10%. Random nhỏ: tín hiệu co dần qua 5 tầng, học rất chậm hoặc không học (`[...]`). Xavier: học được nhưng với ReLU thì phương sai tín hiệu giảm nhẹ mỗi tầng. He: học nhanh và ổn định nhất (`[...]`).

**Bước 11. Chứng minh đối xứng không phá vỡ được.** Mạng 784-128-10 dùng sigmoid vì với ReLU, z = 0 làm đạo hàm bằng 0 nên trọng số đứng yên hoàn toàn và không thấy được cơ chế. In 5 neuron đầu và chênh lệch lớn nhất giữa 128 neuron.
- Nhận xét: sau 3 epoch với init zeros, trọng số khác 0 nhưng 128 neuron giống hệt nhau (chênh lệch `[~0]`). Với init He, chênh lệch `[...]`. Mọi neuron nhận cùng gradient nên cập nhật giống nhau: 128 neuron chỉ học được đúng 1 đặc trưng.

**Bước 12. Phân phối activation qua từng tầng.** Ngay sau khởi tạo, trước khi train. Ra `reports/init_activation_dist.png`.
- Nhận xét: zeros: toàn bộ activation = 0 (cột đơn ở 0). Random nhỏ: std giảm theo cấp số nhân qua các tầng, phân phối co về 0. Xavier: std giảm nhẹ qua các tầng. He: std ổn định ở mọi tầng. Đây là lý do He được dùng mặc định cho ReLU: `Var(W) = 2/n_in` bù đúng việc ReLU cắt mất một nửa phương sai.

**Bước 13. BatchNorm.** Cùng 4 cách init, có và không BatchNorm. Ra `reports/batchnorm_init.csv` và in độ lệch accuracy giữa các cách init.
- Nhận xét: không BN, độ lệch accuracy giữa các init là `[...]`; có BN là `[...]`. BN chuẩn hóa lại phân phối ở mỗi tầng nên giảm mạnh độ nhạy với random nhỏ và Xavier. Ngoại lệ là zeros: mọi neuron vẫn giống nhau nên BN không phá được đối xứng, mạng vẫn không học (`[...]`).

---

## Bước 14. Cẩm nang chẩn đoán

| Triệu chứng | Nguyên nhân có thể | Cách kiểm tra và sửa |
|---|---|---|
| Loss = NaN sau vài epoch | lr quá lớn làm gradient bùng nổ; `log(0)` hoặc chia cho 0 trong loss; dữ liệu chứa NaN/inf; `exp` tràn số | Giảm lr 10 lần; clip gradient (`clip_grad_norm_`); dùng loss ổn định số học (`CrossEntropyLoss` nhận logits, không tự softmax rồi log); `torch.isfinite(X).all()` |
| Loss đứng yên, không giảm | lr quá nhỏ; vanishing gradient (sigmoid/tanh sâu); khởi tạo sai (zeros, random quá nhỏ); chưa gọi `zero_grad`/`step`; dữ liệu chưa chuẩn hóa | In `\|grad\|` theo tầng (bước 8); đổi sang ReLU/GELU + He; chuẩn hóa đầu vào; thử overfit 1 batch nhỏ, nếu không overfit được thì lỗi nằm ở mô hình hoặc code |
| Đổi ReLU sang sigmoid thì mạng "chết" | Sigmoid làm gradient nhân dồn 0,25 mỗi tầng, tầng sâu gần như không học | Dùng Xavier cho sigmoid/tanh, giảm số tầng, thêm BatchNorm hoặc skip-connection, hoặc quay lại ReLU |
| Train accuracy 99%, val accuracy 60% (overfit) | Mô hình quá lớn so với dữ liệu; train quá lâu; dữ liệu train/val không cùng phân phối; rò rỉ dữ liệu | Thêm dropout/weight decay/augmentation; early stopping; giảm kích thước mô hình; kiểm tra cách chia train/val |
| Accuracy kẹt ≈ 10% (MNIST), loss ≈ 2,30 | Init zeros, hoặc nhãn bị xáo trộn so với ảnh, hoặc lr quá lớn làm neuron chết hàng loạt | Đếm % neuron chết (bước 9); in trọng số vài neuron để xem có giống nhau (bước 11); kiểm tra `X` và `y` còn khớp nhau |
| Nhiều neuron luôn xuất 0 (Dying ReLU) | lr quá lớn hoặc bias bị đẩy về âm | Giảm lr, dùng LeakyReLU/ELU/GELU, dùng init He, thêm BatchNorm |
| Gradient tầng đầu nhỏ hơn tầng cuối nhiều bậc | Vanishing gradient: activation bão hòa, mạng sâu, init quá nhỏ | Đổi activation, init He/Xavier đúng cặp, BatchNorm, residual connection |
| Gradient tự viết cho kết quả lạ, loss không giảm đúng | Backprop sai (thiếu đạo hàm activation, sai chiều ma trận, quên chia N) | Chạy kiểm chứng gradient; sai lệch phải < 1e-7. Dùng `eps ≈ 1e-5`, không dùng 1e-10 |
| Loss giảm rồi tăng vọt hoặc dao động mạnh | lr quá lớn; batch quá nhỏ; thiếu BatchNorm ở mạng sâu | Giảm lr hoặc dùng lr schedule/warmup; tăng batch; dùng Adam/clip gradient |
| Kết quả thay đổi mỗi lần chạy, khó so sánh | Không cố định seed, hoặc đổi nhiều thứ cùng lúc | Cố định seed và thứ tự batch; chỉ đổi 1 biến mỗi lần thí nghiệm (như bước 7) |
