# KT-01 - MLP với PyTorch: dự đoán khách hàng rời bỏ

Dữ liệu: Telco Customer Churn (7.043 dòng, nhãn lệch). Chia 70/15/15 có stratify, số đặc trưng sau one-hot: 45, số tham số MLP: 14,593.

## Các bước thực hiện và nhận xét

**1. Tiền xử lý.** Ép kiểu `TotalCharges` (11 dòng trống thành NaN, điền median), bỏ `customerID`, chuẩn hóa biến số, one-hot biến phân loại, tất cả trong sklearn Pipeline chỉ fit trên tập train.

**2. Dataset và DataLoader.** `ChurnDataset` trả về `(x, y)` với `y` đã `unsqueeze(1)`. Batch mẫu có shape x = (128, 45), y = (128, 1).

**3. Mô hình.** MLP 128-64-1 có BatchNorm và Dropout; lớp cuối là logit, không có sigmoid. Dùng `BCEWithLogitsLoss(pos_weight=2.77)` để xử lý lệch lớp.

**4. Vòng lặp huấn luyện.** `src/train.py` có đủ `zero_grad`, `train()`, `eval()`, `no_grad`. Chỉ số theo dõi là PR-AUC vì nhãn lệch.

**5. Thí nghiệm sai 1: bỏ `zero_grad()`.** Sau 30 epoch, train loss là 0.6525 khi đúng và 0.7467 khi bỏ; PR-AUC val tốt nhất lần lượt 0.6288 và 0.6291. Lý do: gradient của các batch cộng dồn, hướng cập nhật bị nhiễm gradient cũ nên đường loss kém ổn định và model khó hội tụ. Xem `reports/exp_zero_grad.png`.

**6. Thí nghiệm sai 2: bỏ `model.eval()`.** Đánh giá 20 lần cùng một model: chế độ eval cho PR-AUC 0.6313 (độ lệch chuẩn 0.00000); chế độ train cho 0.6131 (độ lệch chuẩn 0.00851, biên độ 0.0360). Dropout còn bật và BatchNorm dùng thống kê của từng batch nên điểm dao động ngẫu nhiên, đồng thời thường lệch thấp hơn.

**7. Learning curve.** `reports/learning_curve.png` (optimizer SGD). Train loss và val loss cho thấy mức độ overfit; PR-AUC val dùng để chọn epoch tốt nhất.

**8. Early stopping và checkpoint.** Dừng khi PR-AUC val không cải thiện sau 10 epoch, trả về trọng số tốt nhất và lưu `models/best.pt` bằng `state_dict`.

**9. So sánh 3 optimizer.** Chọn theo PR-AUC val, không chọn theo test. Xem `reports/optimizer_comparison.png`.

| Optimizer | Số epoch | PR-AUC val | PR-AUC test | Thời gian (s) |
|---|---|---|---|---|
| SGD | 20 | 0.6313 | 0.6540 | 17.0 |
| Adam | 18 | 0.6286 | 0.6508 | 19.0 |
| AdamW | 20 | 0.6288 | 0.6433 | 19.2 |

Optimizer được chọn: **SGD**.

**10. So sánh với XGBoost** (`reports/vs_xgboost.csv`).

| Mô hình | PR-AUC val | PR-AUC test | Thời gian train (s) | Số dòng code |
|---|---|---|---|---|
| MLP PyTorch (SGD) | 0.6313 | 0.654 | 17.0389 | 74 |
| XGBoost | 0.6405 | 0.6592 | 2.1668 | 2 |

Kết luận: trên PR-AUC test, XGBoost cao hơn MLP (chênh lệch XGBoost trừ MLP = +0.0052). Tiêu chí "không thua XGBoost quá 0,03": đạt. Với dữ liệu bảng nhỏ, MLP cần nhiều code và nhiều bước tinh chỉnh hơn nhưng không mang lại lợi thế rõ so với mô hình cây; XGBoost là lựa chọn thực dụng hơn cho bài toán này. Giá trị của bài là hiểu vòng lặp huấn luyện để dùng cho các kiến trúc sau.

**11. Lưu và nạp model.** Lưu `state_dict`, nạp vào MLP mới, gọi `eval()`. Xác suất dự đoán trùng khớp với model gốc: True.

**12. CPU và GPU.** 10 epoch: chỉ có CPU: 2.4s (không có GPU để so sánh). Với dữ liệu nhỏ như thế này, GPU thường không nhanh hơn đáng kể vì chi phí chuyển dữ liệu và khởi chạy chiếm phần lớn.

## Cấu trúc thư mục
`notebooks/mlp_pytorch_churn.ipynb`, `src/{dataset,model,train}.py`, `models/best.pt`, `reports/{learning_curve.png, optimizer_comparison.png, exp_zero_grad.png, vs_xgboost.csv}`, `requirements.txt`.
