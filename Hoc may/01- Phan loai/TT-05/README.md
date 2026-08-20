# TT-05-SVM-HoTen — SVM: Phân loại khối u lành tính / ác tính

> Đổi `HoTen` trong tên thư mục thành tên bạn trước khi nộp (không để dấu cách
> trong tên thư mục — dùng gạch nối, ví dụ `TT-05-SVM-NguyenVanA`).

## Bản này đã sửa gì so với bản nộp trước

| # | Vấn đề | Sửa thế nào |
|---|---|---|
| 1 | Cell nạp dữ liệu dùng `X`, `y` **trước khi gán** → `NameError` khi chạy lại notebook từ đầu | Gán `X`, `y` ngay trong cùng cell nạp dữ liệu, trước khi dùng |
| 2 | `savefig('../reports/...')` ghi ra ngoài thư mục bài nếu chạy sai `cwd` | Tính `ROOT_DIR`/`REPORTS_DIR`/`MODELS_DIR` bằng `pathlib`, tự `mkdir(parents=True, exist_ok=True)`, không phụ thuộc `cwd` |
| 3 | Ngưỡng xác suất dò trực tiếp trên tập **test** → recall 1.000/FN=0 là số đã tối ưu cho test, không phải ước lượng khách quan | Dò ngưỡng bằng xác suất **out-of-fold** (`cross_val_predict`, cv=5) trên **tập train**; tập test chỉ dùng **một lần duy nhất** ở bước đánh giá cuối |
| 4 | `GridSearchCV` dùng `scoring=recall` thuần, không ràng buộc precision → một model "luôn đoán ác tính" đạt recall=1.0 vẫn có thể thắng | Thêm `DummyClassifier` (luôn đoán ác tính) để **chứng minh** vấn đề (recall 1.0, precision 0.368 trên test), rồi đổi `scoring` sang `fbeta_score(beta=2)` — ưu tiên recall nhưng có phạt precision |
| 5 | Commit `wdbc.data` nhưng code nạp từ `sklearn.datasets` — hai nguồn dữ liệu không nhất quán | Notebook và `src/train.py` đều nạp trực tiếp từ `wdbc.data` (parse theo `wdbc.names`), không dùng `sklearn.datasets.load_breast_cancer` nữa |
| 6 | `src/train.py` hardcode `'../models/...'`, không `makedirs`, không xuất metric | Viết lại: đường dẫn tính từ `Path(__file__).resolve().parent`, `mkdir(parents=True, exist_ok=True)`, xuất `reports/metrics.json`. Đã test chạy từ `cwd` khác vẫn đúng |
| 7 | Tên thư mục `TT - 05` có dấu cách | Đổi thành `TT-05-SVM-<HoTen>` (không dấu cách) |

## Đối chiếu tiêu chí hoàn thành (mục 6 đề bài)

| Tiêu chí | Đã làm ở đâu | Kết quả |
|---|---|---|
| `StandardScaler` luôn trong `Pipeline` | Notebook Bước 5; `src/train.py` | Không rò rỉ scale từ CV/test vào fit |
| `GridSearchCV(cv=5)` với scorer tự định nghĩa, xử lý đúng nhãn ngược (0 = ác tính), xác nhận bằng `target_names` | Notebook Bước 1 (xác nhận nhãn), Bước 7 (`make_scorer(fbeta_score, beta=2, pos_label=0)`) | Best: `C=1, gamma=scale, kernel=rbf` |
| Bảng so sánh CÓ vs KHÔNG chuẩn hoá bằng số thật | Notebook Bước 5 | recall **0.857 → 0.976** |
| So sánh 3 kernel kèm số support vectors | Notebook Bước 6 | linear: recall 0.976, **32 SV**; rbf: recall 0.976, **97 SV**; poly: recall 0.762, **145 SV** |
| Heatmap C × gamma dựng từ `gs.cv_results_` (điểm CV, không phải test) | Notebook Bước 8 | `reports/C_gamma_heatmap.png` |
| Chọn ngưỡng để recall lớp ác tính cao, đo trung thực | Notebook Bước 10–11 | Ngưỡng 0.40 (chọn qua OOF trên train) → **test recall 0.976**, FN = **1/42** |
| Ma trận nhầm lẫn | Notebook Bước 11 | `reports/confusion_matrix.png` |
| Giải thích vì sao SVM chỉ phụ thuộc support vectors | Notebook Bước 9 | 108/455 mẫu train (23.7%) là support vectors |
| Nêu hạn chế của SVM | Notebook, mục cuối | Không giải thích được từng ca; chậm với dữ liệu lớn; recall cao không tự động đáng tin nếu tối ưu/đo sai cách |

## Quy ước nhãn
Dữ liệu gốc `wdbc.data` dùng `M` (malignant) / `B` (benign). Trong code, ánh xạ
tường minh **`M → 0` (ác tính), `B → 1` (lành tính)** — trùng với quy ước của
`sklearn.datasets.load_breast_cancer`, nhưng ở đây được xác nhận bằng `print()`
ngay sau khi nạp dữ liệu chứ không giả định ngầm.
Toàn bộ recall/precision trong báo cáo này tính theo lớp **0 (ác tính)**.

## Kết quả chính (chạy trên `notebooks/svm_breast_cancer.ipynb`, cũng in ra ở `reports/metrics.json`)

| Thí nghiệm | Recall (ác tính) | Precision (ác tính) |
|---|---|---|
| SVM RBF — KHÔNG chuẩn hoá | 0.857 | 0.947 |
| SVM RBF — CÓ chuẩn hoá | 0.976 | 0.976 |
| Kernel linear (C=1), 32 support vectors | 0.976 | 0.953 |
| Kernel rbf (C=1), 97 support vectors | 0.976 | 0.976 |
| Kernel poly (C=1), 145 support vectors | 0.762 | 1.000 |
| Baseline "luôn đoán ác tính" (minh hoạ lỗi recall thuần) | 1.000 | 0.368 |
| Best GridSearchCV theo F2 (`C=1, gamma=scale, kernel=rbf`) | — (xem dưới) | — |
| **Sau khi chọn ngưỡng 0.40 (dò trên OOF của train, đo trên test)** | **0.976** | **0.891** |
| Logistic Regression (so sánh) | 0.976 | 0.911 |

- **Số ca ÁC TÍNH bị bỏ sót trên tập test (ngưỡng cuối): 1 / 42**
- Số support vectors (model best): **108 / 455 mẫu train (23.7%)** — xem `reports/metrics.json`.
- ROC-AUC ≈ 0.994 (SVM), 0.995 (Logistic Regression).

## Vì sao KHÔNG còn "recall 1.000, FN = 0" như bản trước?
Bản trước dò ngưỡng phân loại **trực tiếp trên tập test**, nên con số recall
1.000/FN=0 là ngưỡng *đã được chọn để làm đẹp chính tập đó* — không phải ước
lượng khách quan cho dữ liệu mới. Bản này chọn ngưỡng bằng xác suất
**out-of-fold** trên tập **train** (mô hình không thấy nhãn thật của mẫu đang
dự đoán trong mỗi fold của `cross_val_predict`), sau đó áp ngưỡng đó lên test
**đúng một lần**. Kết quả (recall 0.976, FN=1/42) thấp hơn con số cũ nhưng là
số **đáng tin để báo cáo**.

## Vì sao không dùng `scoring='recall'` thuần cho GridSearchCV?
Vì recall thuần không phạt việc dự đoán sai lớp lành tính: một model "luôn
đoán ác tính" đạt recall = 1.000 (xem dòng baseline ở bảng trên) dù precision
chỉ 0.368 — vô dụng trên lâm sàng vì đưa mọi bệnh nhân vào diện nghi ngờ. Notebook
Bước 7 dựng một `DummyClassifier` để chứng minh điều này trước khi đổi sang
scorer `fbeta_score(beta=2, pos_label=0)`, vẫn ưu tiên recall nhưng có ràng
buộc precision.

## Chuẩn hoá quan trọng thế nào?
So sánh trực tiếp: recall tăng từ 0.857 → 0.976 chỉ nhờ thêm `StandardScaler`.
Lý do: `area_mean` dao động 143–2501 trong khi `smoothness_mean` chỉ 0.05–0.16;
không scale thì SVM (dựa trên khoảng cách) gần như bỏ qua các biến có biên độ nhỏ.

## Support vectors
SVM chỉ dựa vào các điểm sát/lấn lề (support vectors) để xác định ranh giới —
các điểm nằm sâu trong vùng an toàn của lớp mình không ảnh hưởng đến kết quả
huấn luyện. Xem số liệu cụ thể ở notebook, Bước 9, hoặc `reports/metrics.json`.

## Hạn chế
- SVM không giải thích được quyết định cho từng ca cụ thể (khác Logistic Regression có hệ số rõ ràng).
- Độ phức tạp huấn luyện ~O(n²)–O(n³), không phù hợp dữ liệu > 100k dòng (nên dùng `LinearSVC`).
- Recall cao **không tự động đáng tin**: nếu tối ưu recall thuần hoặc dò ngưỡng/tham số trực tiếp trên tập test, con số báo cáo có thể là ảo tưởng do rò rỉ dữ liệu — hai lỗi đã sửa ở bản này.
- Kết quả cao (~0.98–0.99 AUC) phản ánh bộ dữ liệu Wisconsin dễ tách bạch, không có nghĩa bài toán y tế thực tế dễ tương tự.

## Cấu trúc thư mục
```
TT-05-SVM-HoTen/
├── README.md
├── wdbc.data                # dữ liệu gốc — nguồn DUY NHẤT mà notebook/script dùng
├── wdbc.names                # mô tả cột, dùng để parse wdbc.data
├── notebooks/svm_breast_cancer.ipynb
├── src/train.py              # script độc lập, chạy được từ bất kỳ cwd nào
├── models/svm_pipeline.joblib
├── reports/
│   ├── metrics.json          # toàn bộ số liệu, xuất tự động (không chỉ in ra màn hình)
│   └── {correlation_heatmap, scale_vs_noscale, kernel_comparison,
│         C_gamma_heatmap, confusion_matrix}.png
└── requirements.txt
```

Chạy script độc lập (không cần Jupyter):
```bash
pip install -r requirements.txt
python src/train.py
```

⚖️ **Lưu ý:** đây là công cụ hỗ trợ chẩn đoán, **không** thay thế bác sĩ giải
phẫu bệnh. Dữ liệu gốc thu thập tại Wisconsin (Mỹ), cần kiểm định lại trên
dân số Việt Nam trước khi dùng thật.
