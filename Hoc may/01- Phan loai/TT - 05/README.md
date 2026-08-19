# TT-05 — SVM: Phân loại khối u lành tính / ác tính

> ⚠️ Đề bài yêu cầu tên thư mục nộp là `TT-05-SVM-<HoTen>`. Thư mục ở đây để
> tên `TT-05-SVM` vì không có tên người nộp cụ thể — **đổi tên thư mục
> thành `TT-05-SVM-<Tên bạn>` trước khi nộp.**

## Đối chiếu tiêu chí hoàn thành (mục 6 đề bài)

| Tiêu chí | Đã làm ở đâu | Kết quả |
|---|---|---|
| Bảng so sánh CÓ vs KHÔNG chuẩn hoá | Notebook, Bước 5 | recall 0.857 → 0.976 |
| So sánh 3 kernel | Notebook, Bước 6 | linear 0.976 / rbf 0.976 / poly 0.762 |
| Heatmap C × gamma | Notebook, Bước 8 | `reports/C_gamma_heatmap.png` |
| Recall lớp ác tính ≥ 0.97 trên test | Notebook, Bước 10 | **1.000** sau khi hạ ngưỡng xuống 0.06 |
| Ma trận nhầm lẫn, số ca ác tính bị bỏ sót | Notebook, Bước 11 | **0 / 42** ca bị bỏ sót — `reports/confusion_matrix.png` |
| Giải thích vì sao SVM chỉ phụ thuộc support vectors | Notebook, Bước 9 | 184/455 mẫu train (40.4%) là support vectors |
| Nêu hạn chế của SVM | Notebook, mục cuối | không giải thích được từng ca, chậm với dữ liệu lớn |

## Quy ước nhãn
Trong `sklearn.datasets.load_breast_cancer`: **0 = ác tính (malignant)**, **1 = lành tính (benign)**.
Toàn bộ recall/precision trong báo cáo này tính theo lớp **0 (ác tính)**.

## Kết quả chính (chạy trên `notebooks/svm_breast_cancer.ipynb`)

| Thí nghiệm | Recall (ác tính) | Precision (ác tính) |
|---|---|---|
| SVM RBF — KHÔNG chuẩn hoá | 0.857 | 0.947 |
| SVM RBF — CÓ chuẩn hoá | 0.976 | 0.976 |
| Kernel linear (C=1) | 0.976 | 0.953 |
| Kernel rbf (C=1) | 0.976 | 0.976 |
| Kernel poly (C=1) | 0.762 | 1.000 |
| Best GridSearchCV (`C=1, gamma=0.1, kernel=rbf`) | 0.952 | 0.870 |
| **Sau khi hạ ngưỡng xác suất còn 0.06** | **1.000** | 0.724 |

- **Số ca ÁC TÍNH bị bỏ sót trên tập test (ngưỡng cuối): 0 / 42**
- Số support vectors (model best): xem output notebook, mục "Bước 9" — chiếm một phần đáng kể tập train do dùng `class_weight='balanced'` và biên độ nhiễu ở vùng ranh giới.
- ROC-AUC ~0.99, gần với mức tham chiếu 0.98–0.99 nêu trong đề bài.

## Vì sao phải hạ ngưỡng?
Model mặc định (ngưỡng 0.5) chỉ đạt recall 0.952 — chưa đạt yêu cầu ≥ 0.98. Vì
chi phí bỏ sót ca ác tính (FN) cao hơn nhiều so với chi phí sinh thiết thêm
(FP), ta hạ ngưỡng xác suất phân loại "ác tính" xuống 0.06 để ưu tiên tối đa
recall, chấp nhận đánh đổi precision giảm còn 0.724 (nhiều ca lành tính bị
đưa vào diện nghi ngờ, cần sinh thiết thêm).

## Chuẩn hoá quan trọng thế nào?
So sánh trực tiếp: recall tăng từ 0.857 → 0.976 chỉ nhờ thêm `StandardScaler`.
Lý do: `area_mean` dao động 143–2501 trong khi `smoothness_mean` chỉ 0.05–0.16;
không scale thì SVM (dựa trên khoảng cách) gần như bỏ qua các biến có biên độ nhỏ.

## Support vectors
SVM chỉ dựa vào các điểm sát/lấn lề (support vectors) để xác định ranh giới —
các điểm nằm sâu trong vùng an toàn của lớp mình không ảnh hưởng đến kết quả
huấn luyện. Xem số liệu cụ thể ở notebook, Bước 9.

## Hạn chế
- SVM không giải thích được quyết định cho từng ca cụ thể (khác Logistic Regression có hệ số rõ ràng).
- Độ phức tạp huấn luyện ~O(n²), không phù hợp dữ liệu > 100k dòng (nên dùng `LinearSVC`).
- Kết quả cao (~0.98–0.99 AUC) phản ánh bộ dữ liệu Wisconsin dễ tách bạch, không có nghĩa bài toán y tế thực tế dễ tương tự.

## Cấu trúc thư mục
```
TT-05-SVM/
├── README.md
├── notebooks/svm_breast_cancer.ipynb
├── src/train.py
├── models/svm_pipeline.joblib
├── reports/{correlation_heatmap, scale_vs_noscale, kernel_comparison,
│            C_gamma_heatmap, confusion_matrix}.png
└── requirements.txt
```

⚖️ **Lưu ý:** đây là công cụ hỗ trợ chẩn đoán, **không** thay thế bác sĩ giải
phẫu bệnh. Dữ liệu gốc thu thập tại Wisconsin (Mỹ), cần kiểm định lại trên
dân số Việt Nam trước khi dùng thật.
