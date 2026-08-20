# TT-07 — Gradient Boosting: Dự đoán thu nhập (Adult Census Income)

Chấm điểm khả năng tài chính (>50K$/năm hay không) hỗ trợ duyệt hồ sơ vay tiêu dùng.

## Cách chạy
```bash
pip install -r requirements.txt
python src/train.py          # hoặc mở notebooks/gradient_boosting_income.ipynb
```
Dữ liệu đặt sẵn tại `data/adult.data` (train) và `data/adult.test` (test).

## Kết quả chính

| Model | ROC-AUC | PR-AUC | Thời gian train |
|---|---|---|---|
| Dummy (baseline) | 0.500 | 0.236 | — |
| Decision Tree (depth=6) | 0.895 | 0.723 | — |
| Random Forest | 0.894 | 0.743 | 49.3s |
| AdaBoost | 0.912 | 0.787 | 2.9s |
| **Gradient Boosting** | **0.925** | **0.820** | 8.1s |
| HistGradientBoosting | 0.927 | — | ~4.1s (nhanh hơn GB ~2.3×) |

Đạt mức tham chiếu ROC-AUC ~0,92–0,93.

## BAGGING vs BOOSTING
- **Random Forest (Bagging)** dùng cây **sâu** (mỗi cây học độc lập, song song), giảm **variance** bằng cách bỏ phiếu đa số. Cây cần đủ mạnh vì không có cơ chế sửa lỗi lẫn nhau.
- **Gradient Boosting** dùng cây **nông** (`max_depth=3`) học **tuần tự**, mỗi cây sửa phần dư (residual) của các cây trước, giảm **bias**. Nếu cây sâu, một cây đã khớp gần hết dữ liệu → không còn gì để cây sau học, dẫn tới overfit ngay từ đầu và mất ý nghĩa "học yếu, cộng dồn từ từ" của boosting.
- Kết quả grid dò `learning_rate × n_estimators` xác nhận quan hệ nghịch: `learning_rate` nhỏ (0.01) cần nhiều cây hơn mới đạt AUC tốt (0.915 ở 500 cây), trong khi `learning_rate` lớn (0.3) đạt đỉnh sớm rồi giảm dần khi thêm cây (dấu hiệu overfit) — xem `reports/lr_vs_nestimators.png`.
- Đường train/validation loss (`reports/loss_theo_so_cay.png`) với early-stopping (`n_iter_no_change=20`) cho thấy validation loss vẫn giảm ổn định gần hết quá trình nhờ `learning_rate` nhỏ + `subsample=0.8`, không có overfit rõ rệt trong khoảng cây đã huấn luyện.

## ⚖️ THIÊN LỆCH (Bias) theo giới tính / chủng tộc

| Nhóm | n | Tỉ lệ >50K thực tế | Tỉ lệ >50K dự đoán | Accuracy |
|---|---|---|---|---|
| Nam | 10,860 | 30.0% | 25.1% | 84.1% |
| Nữ | 5,421 | 10.9% | 7.9% | 93.8% |
| Black | 1,561 | 11.5% | 8.1% | 93.1% |
| White | 13,946 | 25.0% | 20.7% | 86.6% |
| Asian-Pac-Islander | 480 | 27.7% | 24.0% | 85.8% |

Model dự đoán nam giới có thu nhập >50K gấp hơn 3 lần nữ giới, phản ánh đúng (và khuếch đại nhẹ) chênh lệch có sẵn trong dữ liệu điều tra dân số 1994. Nhóm Black có tỉ lệ dự đoán >50K thấp hơn đáng kể so với White dù cùng cỡ mẫu lớn. Accuracy cao hơn ở nhóm thiểu số (Nữ, Black) chủ yếu vì lớp `<=50K` chiếm đa số ở các nhóm này — accuracy cao không đồng nghĩa công bằng.

> ⚠️ Đây là dữ liệu điều tra dân số Mỹ 1994, chứa định kiến lịch sử rõ rệt. Model **học và khuếch đại** các định kiến đó qua các biến thay thế (occupation, hours-per-week, marital-status…) ngay cả khi không dùng trực tiếp `sex`/`race` để dự đoán. **Không dùng model này cho quyết định thật về con người.**

## Cấu trúc thư mục
```
TT-07-GradientBoosting/
├── README.md
├── notebooks/gradient_boosting_income.ipynb
├── src/train.py
├── models/gb_pipeline.joblib
├── reports/{loss_theo_so_cay.png, lr_vs_nestimators.png, bias_by_group.png,
│            model_comparison.csv, bias_by_group.csv}
├── data/{adult.data, adult.test}
└── requirements.txt
```
