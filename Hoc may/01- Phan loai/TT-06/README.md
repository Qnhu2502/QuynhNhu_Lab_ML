# TT-06 — Naive Bayes — Lọc tin nhắn rác

## Dữ liệu

- Nguồn: `spam.csv` (5.572 dòng, encoding `latin-1`).
- Sau khi loại **403 dòng trùng lặp** → còn **5.169 dòng** (ham 87,4% / spam 12,6%).
- Chia train/test 80/20, `stratify` theo nhãn, vector hoá được `fit` riêng trên tập train.

## Bảng so sánh 3 tổ hợp (trên tập test, alpha=0.1, ngram=(1,2))

| Tổ hợp | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| TF-IDF + MultinomialNB | 0,9833 | 0,9008 | 0,9402 | 0,9855 |
| Count + MultinomialNB | 0,9524 | 0,9160 | 0,9339 | 0,9836 |
| TF-IDF + BernoulliNB | 1,0000 | 0,8550 | 0,9218 | 0,9816 |

Baseline `DummyClassifier` (luôn đoán ham): accuracy 0,8733 — cao nhưng vô dụng, minh hoạ vì sao không dùng accuracy làm thước đo chính.

TF-IDF + MultinomialNB thắng về F1. BernoulliNB cho precision tuyệt đối nhưng recall thấp nhất — với SMS ngắn, chỉ biết "có/không có từ" mất nhiều thông tin hơn tần suất/TF-IDF.

## Dò alpha & ngram_range (GridSearchCV, 5-fold, scoring=f1)

Tốt nhất: **alpha=0,01, ngram_range=(1,1)** — f1 CV ≈ 0,938. Trên test: precision 0,9746 / recall 0,8779 / f1 0,9237.

## Top 20 từ đặc trưng của spam

`free, call, txt, mobile, text, reply, claim, stop, prize, urgent, ur, cash, win, nokia, service, tone, please, guaranteed, apply, code` (nhóm từ quảng cáo, kêu gọi hành động, khuyến mãi — hợp lý với domain SMS rác).

Chi tiết + biểu đồ: `reports/top_tu_spam.png`.

## Ngưỡng đạt precision ≥ 0,98

Chọn threshold = 0,571 trên xác suất dự đoán → **precision 0,9829, recall 0,8779**.

Confusion matrix (test set, 1.034 tin):

| | Dự đoán ham | Dự đoán spam |
|---|---|---|
| **Thực tế ham** | 901 | 2 |
| **Thực tế spam** | 16 | 115 |

Chỉ 2 tin thật bị chặn nhầm (FP) trên 903 tin ham — đạt mục tiêu ưu tiên precision như README yêu cầu.

## Phân tích lỗi (10 ca sai)

Gần hết là lỗi FN: spam bị đoán nhầm thành ham. Điểm chung là các tin spam "cá nhân hoá" — không dùng từ khoá rõ ràng như "free"/"prize", viết tắt nhiều, giọng văn giống tin nhắn thật (ví dụ giả làm lời mời hẹn hò, thông báo cuộc gọi nhỡ). Đây là đánh đổi chấp nhận được vì mục tiêu là precision cao, không phải recall cao — thà lọt vài tin rác còn hơn chặn nhầm OTP.

## So với Logistic Regression + TF-IDF

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Naive Bayes (best) | 0,9746 | 0,8779 | 0,9237 |
| Logistic Regression | 0,9516 | 0,9008 | 0,9255 |

LogReg nhỉnh hơn chút về F1/recall, nhưng NB train nhanh và đơn giản hơn nhiều — hợp làm baseline production trước khi đầu tư vào model nặng hơn.

## Vì sao alpha = 0 gây lỗi

Nếu một từ chưa từng xuất hiện trong lớp ham (hoặc spam) trong tập train, `P(từ | lớp) = 0`. Vì Naive Bayes nhân xác suất các từ với nhau, một số hạng bằng 0 sẽ kéo cả tích về 0, bất kể các từ khác trong câu nói lên điều gì. Laplace smoothing (`alpha > 0`) cộng thêm một lượng nhỏ vào mọi số đếm để không có xác suất nào bằng 0 tuyệt đối.

## Thời gian train/predict

- Train: ~0,05s
- Dự đoán 1 tin: ~0,6ms (đạt yêu cầu < 5ms/tin)

## Cấu trúc thư mục

```
TT-06-NaiveBayes/
├── README.md
├── requirements.txt
├── spam.csv
├── notebooks/naive_bayes_sms.ipynb
├── src/train.py
├── models/nb_pipeline.joblib
└── reports/
    ├── bang_so_sanh.csv
    ├── top_tu_spam.png
    ├── confusion_matrix.png
    └── do_dai_tin.png
```

## Chạy lại

```bash
pip install -r requirements.txt
python src/train.py
```
