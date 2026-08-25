# TT-09 — ADABOOST
## Phát hiện xâm nhập mạng trong hệ thống giám sát an ninh

| | |
|---|---|
| 🎓 **Khoá** | HỌC MÁY · [Buổi 6](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-06-Ensemble-EndToEnd) |
| 🧠 **Nhóm** | Phân loại · Boosting (thế hệ đầu) |
| 🔧 **Thuật toán** | AdaBoost (Adaptive Boosting) |
| 🏭 **Lĩnh vực** | An ninh mạng · SOC |
| ⏱ **Thời lượng** | 5–7 giờ |
| 📈 **Độ khó** | ⭐⭐⭐ |

---

## 1. THUẬT TOÁN NÀY LÀ GÌ

AdaBoost (1995) là thuật toán boosting **đầu tiên**. Khác Gradient Boosting ở chỗ:
thay vì học phần dư, nó **đánh trọng số lại các MẪU**.

```
   Vòng 1: mọi mẫu trọng số bằng nhau → train cây cụt (stump, depth=1)
           → những mẫu bị phân SAI được TĂNG trọng số
   Vòng 2: cây mới buộc phải chú ý vào các mẫu khó đó
           → lại tăng trọng số mẫu vẫn sai
   ...
   Kết quả cuối = tổng có trọng số của tất cả cây
                  (cây nào chính xác hơn được tiếng nói lớn hơn)

        α_m = ½·ln((1 − err_m) / err_m)      ← trọng số của cây thứ m
```

| | AdaBoost | Gradient Boosting |
|---|---|---|
| Cơ chế | Đánh trọng số **MẪU** | Học **PHẦN DƯ** |
| Weak learner | Stump (depth = 1) | Cây nông (depth = 3) |
| Nhạy với nhiễu/outlier | ⚠️ **RẤT nhạy** | Ít nhạy hơn |
| Còn dùng nhiều? | Ít — chủ yếu để hiểu nền tảng | ✅ Phổ biến |

---

## 2. BÀI TOÁN THỰC TẾ

```
   Trung tâm điều hành an ninh (SOC) nhận hàng triệu gói tin/phút.
   Cần phân loại: kết nối BÌNH THƯỜNG hay TẤN CÔNG.

   ⚠️ Đặc thù an ninh mạng:
      • Bỏ sót 1 cuộc tấn công → có thể mất toàn bộ dữ liệu công ty  → RECALL quan trọng
      • Báo động giả quá nhiều → nhân viên SOC "mệt mỏi cảnh báo"
        (alert fatigue) rồi bỏ qua cả cảnh báo thật → PRECISION cũng quan trọng
   → Cân bằng bằng F1 / F2-score.
```

---

## 3. BỘ DỮ LIỆU

| | |
|---|---|
| **Tên** | NSL-KDD (bản cải tiến của KDD Cup 99) |
| **Link** | https://www.unb.ca/cic/datasets/nsl.html |
| **Định dạng dùng trong bài này** | **ARFF** (WEKA) — `KDDTrain+.arff`, `KDDTest+.arff` |
| **Kích thước** | Train ~125.973 dòng, Test (`KDDTest+.arff`) 22.544 dòng × 42 cột (41 đặc trưng + `class`) |
| **Nhãn** | `class` = `normal` / `anomaly` — **đã nhị phân hoá sẵn trong bản ARFF** |

> ⚠️ **Khác biệt quan trọng so với bản `.txt` gốc:** bản ARFF chỉ có 42 cột (41 đặc trưng + `class`
> nhị phân), **không có** cột `label` với tên tấn công cụ thể (`neptune`, `smurf`,
> `guess_passwd`...) và **không có** `difficulty_level` như bản `.txt` 43 cột. Vì vậy bài này
> **không tách được** nhóm tấn công DoS/Probe/R2L/U2R — EDA và báo cáo chỉ dừng ở mức nhị phân
> normal/attack. Nếu cần phân tích chi tiết theo nhóm tấn công, phải dùng bản `KDDTrain+.txt`/
> `KDDTest+.txt` gốc (có cột `label` cụ thể + `difficulty_level`).

**Nguồn thay thế nhẹ hơn:** `sklearn.datasets.fetch_kddcup99(subset='SA')` — tải trực tiếp, không
cần đăng ký. Lưu ý đây là **KDD Cup 99 gốc**, không phải NSL-KDD, nên **không có** đặc tính "test
chứa tấn công lạ" — nếu dùng nguồn này, mục 9 (chênh lệch CV vs test do zero-day) không áp dụng được.

### ⚠️ Bẫy dữ liệu

```
   1. Bộ NSL-KDD có tập TEST (KDDTest+) chứa các LOẠI TẤN CÔNG KHÔNG có trong train
      → đây là CỐ Ý (mô phỏng tấn công zero-day)
      → điểm trên tập test sẽ THẤP hơn CV/validation rất nhiều — đó là điều ĐÚNG, không phải lỗi
      → bản ARFF không cho biết CỤ THỂ tấn công nào là mới (nhãn đã nhị phân hoá), nhưng cơ chế
        gây chênh lệch điểm vẫn không đổi

   2. Lớp U2R cực hiếm (~0,04%) trong bản .txt gốc → gần như không học được nếu giữ đa lớp
      → bài này dùng luôn `class` nhị phân sẵn có trong ARFF (normal vs attack)

   3. 3 cột phân loại: protocol_type, service (70 mức!), flag
      → one-hot làm số chiều tăng mạnh; cần handle_unknown='ignore' vì service ở test
        có thể có giá trị hiếm không xuất hiện ở train

   4. File ARFF có block header '@attribute'/'@data' — đọc thẳng bằng pd.read_csv sẽ lỗi
      → phải bỏ qua mọi dòng trước '@data', chỉ parse phần dữ liệu CSV phía sau
```

### 3.1. ⚠️ Nguyên tắc phương pháp: KHÔNG chọn siêu tham số trên test (bài học từ TT-07)

`KDDTest+.arff` chỉ được dùng **đúng một lần**, ở bước đánh giá cuối (mục 9 trong notebook) — vì
đây chính là con số cần để đo mức độ mô hình sụt điểm trước tấn công lạ (điều README nhấn mạnh ở
mục 6). Nếu dùng test để chọn `n_estimators`/`learning_rate` hay vẽ đường F1 chẩn đoán, chênh lệch
CV-vs-test báo cáo ra sẽ không còn đáng tin — vì một phần chênh lệch đó có thể do rò rỉ, không phải
do tấn công lạ. Mọi lựa chọn siêu tham số, đường F1 theo n_estimators (mục 6), và thí nghiệm nhiễu
nhãn (mục 7) đều dùng tập **validation tách riêng 20% từ `KDDTrain+.arff`**.

---

## 4. HƯỚNG ĐI ĐÚNG
from sklearn.tree import DecisionTreeClassifier

ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),   # ⭐ STUMP — đúng bản chất AdaBoost
    n_estimators=300,
    learning_rate=0.5,        # quan hệ NGƯỢC với n_estimators
    random_state=42,
)
```

**Ba tham số:**

| Tham số | Ý nghĩa | Khuyến nghị |
|---------|---------|-------------|
| `estimator` | Weak learner | `max_depth=1` (stump) hoặc 2–3 nếu underfit |
| `n_estimators` | Số vòng | 100–500 |
| `learning_rate` | Đóng góp mỗi cây | 0,1–1,0 (nhỏ thì cần nhiều cây hơn) |

> ⚠️ **Điểm yếu chí mạng cần thí nghiệm:** AdaBoost rất nhạy với **NHÃN SAI**.
> Mẫu bị gán nhãn sai sẽ liên tục bị phân sai → trọng số tăng vô hạn → model
> dồn hết sức học một điểm rác. Bài này yêu cầu **chứng minh** hiện tượng đó.

---

## 5. CÁC BƯỚC THỰC HIỆN

```
   ☐ 1. Nạp dữ liệu ARFF (bỏ qua block @attribute, đọc CSV sau @data), nhãn class đã nhị phân sẵn
   ☐ 2. One-hot 3 cột phân loại; scale các cột số
   ☐ 3. EDA: tỉ lệ normal/attack (bản ARFF không tách được nhóm DoS/Probe/R2L/U2R — xem mục 3)
   ☐ 4. Tách 20% của TRAIN làm validation (KHÔNG đụng KDDTest+.arff) — dùng cho bước 6, 7
   ☐ 5. Baseline: DummyClassifier + 1 stump đơn lẻ (depth=1); rồi AdaBoost 300 stump so với 1 stump
        → cho thấy 1 stump YẾU tới mức nào, và boosting mạnh lên ra sao khi cộng dồn
   ☐ 6. Vẽ đường accuracy/F1 theo n_estimators = 1..300 — đo trên VALIDATION, không phải test
   ☐ 7. ⭐ THÍ NGHIỆM NHIỄU: đảo ngẫu nhiên 5% nhãn tập train (chỉ trong phần dùng để fit)
        → chạy lại AdaBoost và Random Forest, đo trên validation nhãn sạch
        → lập bảng: cái nào tụt điểm nhiều hơn? Giải thích.
   ☐ 8. So sánh AdaBoost vs Gradient Boosting (TT-07) vs Random Forest (TT-03) trên validation
   ☐ 9. Đánh giá trên KDDTest+.arff gốc (có tấn công lạ) — CHỈ MỘT LẦN → phân tích chênh lệch
   ☐ 10. Ma trận nhầm lẫn + tính tỉ lệ báo động giả mỗi ngày (alert fatigue)
   ☐ 11. Lưu model & bảng tổng kết
```

---

## 6. TIÊU CHÍ HOÀN THÀNH

```
   ☐ So sánh 1 stump vs 300 stump → chứng minh boosting hiệu quả
   ☐ Có biểu đồ F1 theo số vòng lặp
   ☐ ⭐ Có thí nghiệm NHIỄU NHÃN + bảng so sánh AdaBoost vs Random Forest
   ☐ Có bảng so sánh 3 thuật toán ensemble
   ☐ Giải thích được chênh lệch điểm giữa CV và tập test NSL-KDD
   ☐ Ước tính số cảnh báo giả/ngày ở ngưỡng đã chọn
```

**Mức tham chiếu:** F1 ~0,95+ trên cross-validation, nhưng chỉ ~0,75–0,80 trên tập
test gốc (do có tấn công chưa từng thấy) — **chênh lệch này là kết quả đúng**.

---

## 7. CẠM BẪY

| Cạm bẫy | Hậu quả |
|---------|---------|
| Dùng cây sâu làm weak learner | Mất bản chất AdaBoost, overfit |
| Bỏ qua nhiễu nhãn | Model dồn sức học điểm rác |
| Chỉ đánh giá bằng CV | Không thấy được điểm yếu với tấn công lạ |
| Giữ nguyên đa lớp với U2R | Lớp 0,04% không học nổi |
| Không tính số báo động giả | Hệ thống không dùng được thực tế |

---

## 8. SẢN PHẨM NỘP & MỞ RỘNG

```
TT-09-AdaBoost-<HoTen>/
├── README.md          ← có mục "Dữ liệu" (3, ARFF), "Phương pháp" (3.1), "THÍ NGHIỆM NHIỄU NHÃN" (7)
├── .gitignore         ← không commit data/ thô và artifact sinh ra
├── data/               ← KHÔNG commit; tự tải KDDTrain+.arff/KDDTest+.arff vào đây
├── notebooks/adaboost_ids.ipynb
├── src/train.py       ← chạy độc lập: python src/train.py (tự tạo models/, reports/)
├── models/adaboost.joblib
├── reports/
│   ├── eda_attack_distribution.png
│   ├── f1_theo_vong_lap.png
│   ├── thi_nghiem_nhieu.png / .csv
│   ├── so_sanh_ensemble.png / .csv
│   ├── confusion_matrix_test.png
│   └── final_summary.csv
└── requirements.txt
```

> **Dữ liệu:** đặt `KDDTrain+.arff` và `KDDTest+.arff` vào `data/` (cùng cấp `notebooks/`, `src/`)
> trước khi chạy notebook hoặc `src/train.py`. `data/` đã có trong `.gitignore`.

**Mở rộng:**
1. Bài toán ĐA LỚP: phân loại đúng 5 nhóm tấn công (dùng `SAMME`) — cần đổi sang bản `.txt` gốc có
   nhãn cụ thể, vì bản ARFF nhị phân trong bài này không còn giữ thông tin đó
2. Phát hiện bất thường không giám sát (Isolation Forest) để bắt tấn công zero-day
3. Học trực tuyến: cập nhật model khi có mẫu tấn công mới mà không train lại

**Tham khảo:** [Buổi 6 — Ensemble & Boosting](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-06-Ensemble-EndToEnd/Tai-Lieu/ly_thuyet_chi_tiet_buoi_06.md)
