# TT-04 — Logistic Regression: Chẩn đoán nguy cơ bệnh tim

Dự đoán nguy cơ mắc bệnh tim (Heart Disease) bằng Logistic Regression, có so sánh với SVM và
Random Forest.

## Cấu trúc thư mục

```
TT-04/
├── data/
│   └── README_DATA.txt      # hướng dẫn tải heart.csv (không commit dữ liệu thô)
├── src/
│   └── train.py              # script huấn luyện, chạy độc lập qua dòng lệnh
├── logistic_heart.ipynb      # notebook phân tích, biểu đồ, giải thích từng bước
├── reports/                  # sinh ra khi chạy notebook/train.py (odds_ratio.csv, vif_table.csv, hình ảnh)
├── models/                   # sinh ra khi chạy train.py (logreg_pipeline.pkl)
└── README.md                 # file này
```

## Cách chạy

1. Tải `heart.csv` theo hướng dẫn trong `data/README_DATA.txt`, đặt vào `data/heart.csv`.
2. Chạy script huấn luyện:
   ```
   python src/train.py --data data/heart.csv
   ```
   hoặc mở `logistic_heart.ipynb` để xem phân tích đầy đủ kèm biểu đồ.

`src/train.py` được tổ chức thành các hàm độc lập, dễ tái sử dụng/test:
`load_data`, `build_pipeline`, `odds_ratio_table`, `choose_threshold`, `vif_table`.

## Phương pháp

- **Loại trùng lặp trước khi split:** bản Kaggle của bộ dữ liệu này là bản nhân bản từ dữ liệu
  gốc UCI, nên `drop_duplicates()` được thực hiện **trước** `train_test_split` để tránh rò rỉ dữ
  liệu (nếu không, các dòng giống hệt nhau có thể vừa nằm ở train vừa nằm ở test).
- **Tiền xử lý:** `ColumnTransformer` gồm `OneHotEncoder(drop='first')` cho các biến phân loại
  không có thứ tự (`cp`, `thal`, `ca`, `slope`, `restecg`, `sex`, `fbs`, `exang`) và
  `StandardScaler` cho các biến số, gói trong một `Pipeline` cùng mô hình.
- **Chọn C bằng `LogisticRegressionCV(scoring='roc_auc')`** — không dùng `scoring='recall'`.
  Dùng recall làm tiêu chí CV khiến mô hình học cách "gian lận" (predict gần như luôn dương để
  recall cao), hệ số co gần về 0, bảng odds ratio trở nên vô nghĩa. Đây là lỗi nghiêm trọng nhất
  ở bản trước và đã được sửa — xem chi tiết trong notebook (mục 6, 7).
- **Chọn ngưỡng đạt recall ≥ 0.90** bằng out-of-fold prediction (`cross_val_predict`) trên tập
  train, sau đó áp dụng cố định lên test — không dò ngưỡng trực tiếp trên tập test.
- **VIF** chỉ tính trên biến số của tập train (không dùng toàn bộ train+test).
- So sánh với SVM và Random Forest bằng ROC AUC, precision/recall tại ngưỡng 0.5.
- Diễn giải kết quả cho một bệnh nhân cụ thể, kèm lưu ý đạo đức về giới hạn của mô hình.

## Kết quả (trên `data/heart.csv`, bản Kaggle 1025 dòng)

- Sau `drop_duplicates()`: **1025 → 302 dòng** (loại 723 dòng trùng lặp).
- Train/test: 241 / 61 dòng (stratified, test_size=0.2).
- `LogisticRegressionCV(scoring='roc_auc')` chọn `C ≈ 0.359`; |hệ số| lớn nhất = 1.21 (không còn
  bị co về 0 như bản lỗi cũ dùng `scoring='recall'`).
- **Top yếu tố nguy cơ (odds ratio cao nhất):** `cp_2` (OR≈2.92), `cp_3` (OR≈2.31), `thalach`
  (OR≈1.92), `thal_2` (OR≈1.76), `cp_1` (OR≈1.51) — xem đầy đủ ở `reports/odds_ratio.csv`.
- So sánh L1/L2: L2 AUC=0.906 (0 hệ số=0), L1 AUC=0.902 (2 hệ số=0) — cả hai đều hoạt động tốt,
  không còn hiện tượng "mô hình chết" (AUC=0.5) như bản lỗi cũ.
- **Ngưỡng đạt recall ≥ 0.90** (chọn bằng out-of-fold trên train): ngưỡng = 0.430, recall/precision
  out-of-fold = 0.901 / 0.797. Áp dụng lên test: AUC=0.906, recall=0.879, precision=0.853.
- **VIF** (trên train): cao nhất là `thalach` ≈ 1.31 — không có biến nào > 10, không đáng lo về
  đa cộng tuyến.
- **So sánh mô hình** (ROC AUC): LogisticRegression 0.906, RandomForest 0.887, SVM 0.878 —
  Logistic Regression tốt nhất và có ưu điểm diễn giải được qua odds ratio.

File chi tiết: `reports/odds_ratio.csv`, `reports/vif_table.csv`, `reports/eda.png`,
`reports/roc_pr_curve.png` (sinh ra khi chạy `src/train.py` hoặc notebook).

## Giới hạn & lưu ý đạo đức

Đây là công cụ hỗ trợ tham khảo, không thay thế chẩn đoán y khoa. Dữ liệu huấn luyện có quy mô
nhỏ và chưa được kiểm định lâm sàng độc lập; xem mục 13 trong notebook để biết chi tiết đầy đủ.
