# TT-08 — XGBoost: Phát hiện gian lận thẻ tín dụng theo thời gian thực

Dữ liệu: [Credit Card Fraud Detection (Kaggle)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284.807 giao dịch, **492 ca gian lận (0,1727%)**.

## Cấu trúc dự án

```
TT-08-XGBoost/
├── README.md
├── notebooks/xgboost_fraud.ipynb   ← toàn bộ pipeline, đã chạy sẵn (có output/biểu đồ)
├── src/train.py                    ← script huấn luyện độc lập, dùng lại cho retrain/CI
├── models/xgb_fraud.json           ← mô hình XGBoost đã huấn luyện
├── reports/
│   ├── eda_overview.png
│   ├── pr_vs_roc.png
│   ├── chi_phi_theo_nguong.png
│   └── feature_importance.png
└── requirements.txt
```

Chạy lại: `pip install -r requirements.txt` rồi mở `notebooks/xgboost_fraud.ipynb`, hoặc `python src/train.py --data creditcard.csv`.

## Phương pháp

- **Chia dữ liệu theo THỜI GIAN** (sắp xếp theo `Time`, không shuffle): 70% train · 15% validation · 15% test. Lý do: gian lận có tính thời điểm (tấn công theo đợt) — chia ngẫu nhiên sẽ làm mô hình "nhìn thấy tương lai" và điểm đánh giá bị ảo.
- **Feature engineering**: `Time` → `Hour = (Time // 3600) % 24` (giờ trong ngày); `Amount` → `log1p` rồi `StandardScaler` (fit chỉ trên train). `V1`–`V28` giữ nguyên vì đã PCA hoá sẵn.
- **XGBoost**: `scale_pos_weight` cân bằng lớp thiểu số, `eval_metric='aucpr'`, `early_stopping_rounds=50` trên 1000 cây.
- **Đánh giá**: PR-AUC là metric chính; ROC-AUC chỉ mang tính tham khảo.
- **Ngưỡng phân loại**: chọn theo chi phí kinh doanh (chặn nhầm = 200.000đ, bỏ lọt = số tiền giao dịch), không dùng mặc định 0.5.

## VÌ SAO ROC-AUC ĐÁNH LỪA

Ở tỉ lệ lệch 0,17%, số giao dịch hợp lệ (True Negative) áp đảo tuyệt đối. ROC-AUC dựa trên **False Positive Rate = FP / (FP + TN)** — vì mẫu số TN cực lớn, dù mô hình bắn ra khá nhiều cảnh báo sai (FP) thì FPR vẫn gần 0, khiến đường ROC và ROC-AUC trông "đẹp" một cách giả tạo.

PR-AUC dựa trên **Precision = TP / (TP + FP)** — không có TN trong công thức — nên phản ánh đúng câu hỏi thực tế: *trong số cảnh báo gian lận mà hệ thống bắn ra, bao nhiêu phần trăm là đúng?* Đây là con số ảnh hưởng trực tiếp đến trải nghiệm khách hàng (bị chặn nhầm) và chi phí vận hành.

**Kết quả đo được trên tập test (chia theo thời gian):**

| Mô hình | ROC-AUC | PR-AUC |
|---|---|---|
| Dummy (ngẫu nhiên) | ~0,50 | 0,0012 |
| Logistic Regression (balanced) | 0,9778 | 0,6948 |
| **XGBoost** | **0,9826** | **0,7620** |

Chênh lệch 0,2206 điểm giữa ROC-AUC (0,9826) và PR-AUC (0,7620) của XGBoost minh chứng rõ cho việc ROC-AUC đánh giá quá lạc quan so với năng lực thật của mô hình.

## Kết quả chính khác

- **Ngưỡng đạt Precision ≥ 0,90**: 0,9854 → Precision = 0,9024, Recall = 0,7115.
- **Ngưỡng tối ưu theo chi phí** (chặn nhầm 200.000đ / bỏ lọt = số tiền giao dịch): 0,99 → chi phí kỳ vọng ~402.626đ trên tập test, so với ~40.802.362đ nếu dùng ngưỡng mặc định 0,5 (giảm >100 lần).
- **Độ trễ dự đoán**: ~1,6 ms / giao dịch — đạt ràng buộc < 100ms rất thoải mái.
- **scale_pos_weight** dùng trong train: 518,18 (tỉ lệ mẫu hợp lệ / gian lận trong tập train).

## Hạn chế

- `V1`–`V28` đã được PCA hoá để bảo mật dữ liệu gốc → không thể diễn giải ý nghĩa nghiệp vụ của từng đặc trưng, không làm được feature engineering theo domain knowledge.
- Dữ liệu chỉ trải dài 2 ngày → chưa đủ để đánh giá độ ổn định theo mùa vụ/tuần/tháng.
- Chi phí chặn nhầm (200.000đ) là giả định cố định, thực tế có thể thay đổi theo phân khúc khách hàng.

## Theo dõi concept drift

Kẻ gian liên tục đổi chiêu thức nên phân phối dữ liệu và hiệu năng mô hình sẽ suy giảm theo thời gian. Đề xuất:
1. Ghi log mọi dự đoán (điểm số + nhãn thực khi có phản hồi).
2. Theo dõi PR-AUC theo cửa sổ trượt (rolling window) hàng tuần.
3. Cảnh báo khi PR-AUC giảm quá X% so với baseline lúc deploy.
4. Retrain định kỳ hoặc khi phát hiện drift.

**Mở rộng khả thi**: mô phỏng train ngày 1 / test ngày 2 để đo trực tiếp mức suy giảm điểm số theo thời gian; thử Isolation Forest như lớp phát hiện bất thường không giám sát bổ trợ; deploy FastAPI với ngưỡng cấu hình được.
