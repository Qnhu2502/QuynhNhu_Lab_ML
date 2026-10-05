# TT-29 — TF-IDF
## Tự động phân loại ticket hỗ trợ khách hàng về đúng bộ phận

| | |
|---|---|
| 🎓 **Khoá** | HỌC MÁY · [Buổi 8](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-08-NLP) |
| 🧠 **Nhóm** | NLP · Biểu diễn văn bản |
| 🔧 **Thuật toán** | TF-IDF + LinearSVC / Logistic Regression |
| 🏭 **Lĩnh vực** | CSKH · Vận hành dịch vụ |
| ⏱ **Thời lượng** | 5–7 giờ |
| 📈 **Độ khó** | ⭐⭐ |

---

## 1. THUẬT TOÁN NÀY LÀ GÌ

```
   TF  (Term Frequency)     — từ này xuất hiện nhiều thế nào TRONG văn bản này?
   IDF (Inverse Doc Freq)   — từ này có HIẾM trên toàn kho không?

        TF-IDF(t,d) = TF(t,d) × log( N / df(t) )

   Một câu tóm gọn:
   "Từ nào nhiều Ở ĐÂY nhưng hiếm Ở KHẮP NƠI thì đó là từ đáng chú ý."

   Ví dụ trong kho 1.000 ticket:
     "hoá đơn" có mặt trong 30 ticket → IDF = log(1000/30) = 3,5 → TF-IDF CAO
     "và"      có mặt trong 990 ticket → IDF = log(1000/990) = 0,01 → ≈ 0
```

---

## 2. BÀI TOÁN THỰC TẾ

```
   Trung tâm CSKH nhận 2.000 ticket/ngày. Nhân viên tiếp nhận phải đọc rồi
   chuyển về đúng bộ phận: Kỹ thuật · Thanh toán · Tài khoản · Khiếu nại.

   Chuyển SAI bộ phận → ticket bị đá qua đá lại → SLA vỡ, khách bực.
   Thống kê thực tế: ~18% ticket bị chuyển sai ít nhất 1 lần.

   → Model phân loại tự động + cơ chế: nếu độ tin cậy thấp thì để người quyết định.

   ⚠️ Vì sao chọn TF-IDF chứ không phải BERT?
      • Chạy trên CPU, dự đoán < 5 ms → nhúng thẳng vào hệ thống ticket
      • Train lại trong 30 giây khi có nhãn mới
      • Giải thích được: chỉ ra ĐÚNG TỪ nào khiến ticket được xếp vào bộ phận đó
```

---

## 3. BỘ DỮ LIỆU

| | |
|---|---|
| **Dùng để luyện** | 20 Newsgroups — `sklearn.datasets.fetch_20newsgroups` |
| **Kích thước** | 18.846 văn bản × 20 chủ đề |
| **Vì sao dùng** | Có sẵn trong sklearn, không cần tải, cấu trúc giống bài toán ticket |

```python
from sklearn.datasets import fetch_20newsgroups

# ⭐ BẮT BUỘC bỏ headers/footers/quotes — nếu không sẽ RÒ RỈ
train = fetch_20newsgroups(subset='train',
                           remove=('headers', 'footers', 'quotes'))
```

### 🚨 Bẫy rò rỉ kinh điển của bộ này

```
   Nếu KHÔNG bỏ headers:
     mỗi văn bản chứa dòng "Newsgroups: comp.graphics"
     → chính là ĐÁP ÁN nằm ngay trong đầu vào!
     → accuracy vọt lên ~95% (giả tạo), thực tế chỉ ~70%.

   Footer chứa chữ ký email, quotes chứa trích dẫn bài trước → cũng rò rỉ.
```

**Dữ liệu thật của công ty:** xuất ticket từ hệ thống (Zendesk/Jira/Freshdesk),
**bắt buộc ẩn danh** tên khách, số điện thoại, email, số tài khoản trước khi dùng.

---

## 4. HƯỚNG ĐI ĐÚNG

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.calibration import CalibratedClassifierCV

pipe = make_pipeline(
    TfidfVectorizer(lowercase=True, ngram_range=(1, 2),
                    min_df=3,           # bỏ từ xuất hiện < 3 văn bản (lỗi chính tả)
                    max_df=0.8,         # bỏ từ có mặt > 80% văn bản (vô dụng)
                    sublinear_tf=True,  # 1+log(TF) — giảm ảnh hưởng từ lặp nhiều
                    max_features=50000),
    CalibratedClassifierCV(LinearSVC(C=1.0), cv=3),   # ⭐ để có XÁC SUẤT
)
```

> 💡 `LinearSVC` thường **thắng** Naive Bayes và Logistic Regression trên văn bản
> nhiều chiều, nhưng nó **không có `predict_proba`**. Bọc `CalibratedClassifierCV`
> để lấy xác suất phục vụ cơ chế ngưỡng tin cậy.

### Tiếng Việt cần thêm bước tách từ

```python
from underthesea import word_tokenize
def tach_tu_vn(s):
    return word_tokenize(s, format="text")     # "học sinh" → "học_sinh"
# rồi truyền vào TfidfVectorizer(tokenizer=..., preprocessor=...)
```

---

## 5. CÁC BƯỚC THỰC HIỆN

```
   ☐ 1. ⭐ Chạy 2 lần: CÓ và KHÔNG có remove=('headers','footers','quotes')
        → lập bảng so sánh accuracy → viết nhận xét về rò rỉ
        → từ bước 2 CHỈ dùng bản đã remove
   ☐ 2. EDA: số văn bản mỗi lớp, độ dài văn bản, từ phổ biến nhất
   ☐ 3. Baseline: DummyClassifier + Naive Bayes (TT-06)
   ☐ 4. TF-IDF + LinearSVC
   ☐ 5. Khảo sát tham số vectorizer:
        ngram_range (1,1) vs (1,2) · min_df ∈ {1,3,5} · sublinear_tf True/False
        → bảng: số đặc trưng sinh ra × accuracy
   ☐ 6. So sánh 3 bộ phân loại trên cùng TF-IDF: NB · LogReg · LinearSVC
   ☐ 7. ⭐ Với mỗi lớp, in TOP 15 từ có trọng số cao nhất
        → kiểm tra bằng mắt: có hợp lý không? có từ rò rỉ nào lọt vào không?
   ☐ 8. Ma trận nhầm lẫn → nhóm chủ đề nào hay nhầm nhau?
   ☐ 9. ⭐ Cơ chế tin cậy: nếu max(proba) < 0,6 → chuyển người xử lý
        → bảng: % tự động · accuracy của phần tự động · % cần người
   ☐ 10. Đo thời gian: train toàn bộ + dự đoán 1 ticket (phải < 5 ms)
   ☐ 11. Phân tích 10 ca sai → nguyên nhân là gì?
```

---

## 6. TIÊU CHÍ HOÀN THÀNH

```
   ☐ ⭐ Có bảng chứng minh rò rỉ do headers/footers
   ☐ Có khảo sát tham số vectorizer (ít nhất 3 tham số)
   ☐ Có so sánh 3 bộ phân loại
   ☐ ⭐ Có danh sách top 15 từ đặc trưng cho mỗi lớp
   ☐ ⭐ Có bảng cơ chế ngưỡng tin cậy
   ☐ Đo được thời gian dự đoán 1 văn bản < 5 ms
   ☐ Có phân tích ca sai
   ☐ Nêu hạn chế: TF-IDF không hiểu NGHĨA ("tuyệt vời" ≠ "hay" với nó)
```

**Mức tham chiếu (đã remove headers):** accuracy ~0,69–0,73 với 20 lớp.
Con số này **thấp hơn nhiều** so với ~0,95 khi chưa remove — đó chính là bài học.

---

## 7. CẠM BẪY

| Cạm bẫy | Hậu quả |
|---------|---------|
| Quên `remove=(...)` | Rò rỉ nhãn qua header → accuracy ảo |
| `fit` vectorizer trên toàn bộ dữ liệu | Rò rỉ từ vựng và IDF |
| Dùng LinearSVC mà cần xác suất | Không có `predict_proba` → phải calibrate |
| Bỏ stopword chứa từ phủ định | Nguy hiểm với bài phân tích cảm xúc |
| Quên tách từ tiếng Việt | Tokenizer cắt sai từ ghép |
| `min_df=1` | Giữ cả lỗi chính tả → số chiều bùng nổ, overfit |

---

## 8. SẢN PHẨM NỘP & MỞ RỘNG

```
TT-29-TFIDF-<HoTen>/
├── README.md          ← ⭐ có bảng chứng minh rò rỉ + top từ mỗi lớp
├── notebooks/tfidf_ticket_classification.ipynb
├── src/{preprocess.py, train.py}
├── models/tfidf_pipeline.joblib
├── reports/{leakage_comparison.png, top_tu_moi_lop.png, confusion_matrix.png, nguong_tin_cay.png}
└── requirements.txt
```

**Mở rộng:**
1. **SVD/LSA** trên ma trận TF-IDF → giảm chiều, có tăng accuracy không?
2. So sánh với **PhoBERT** (KT-14): chênh lệch accuracy vs chênh lệch tốc độ
3. Học trực tuyến với `SGDClassifier.partial_fit()` → cập nhật model khi có ticket mới
   mà không train lại toàn bộ

**Tham khảo:** [Buổi 8 — TF-IDF & NLP](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-08-NLP/Tai-Lieu/ly_thuyet_chi_tiet_buoi_08.md)
