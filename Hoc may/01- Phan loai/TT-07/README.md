# TT-07 — GRADIENT BOOSTING
## Dự đoán mức thu nhập để chấm điểm hồ sơ vay tiêu dùng

| | |
|---|---|
| 🎓 **Khoá** | HỌC MÁY · [Buổi 6](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-06-Ensemble-EndToEnd) |
| 🧠 **Nhóm** | Phân loại · Ensemble (Boosting) |
| 🔧 **Thuật toán** | Gradient Boosting Classifier |
| 🏭 **Lĩnh vực** | Tài chính · Tín dụng tiêu dùng |
| ⏱ **Thời lượng** | 6–8 giờ |
| 📈 **Độ khó** | ⭐⭐⭐ |

---

## 1. THUẬT TOÁN NÀY LÀ GÌ

```
   BAGGING (Random Forest — TT-03)     BOOSTING (bài này)
     Cây 1 ─┐                            Cây 1 → sai ─→ Cây 2 → sai ─→ Cây 3
     Cây 2 ─┼─▶ BỎ PHIẾU                   ↓        ↓        ↓
     Cây 3 ─┘                            Kết quả = Cây1 + Cây2 + Cây3
     (học SONG SONG, độc lập)            (học TUẦN TỰ, cây sau sửa lỗi cây trước)
     Giảm VARIANCE                        Giảm BIAS
```

**Cơ chế:** mỗi cây mới học để dự đoán **PHẦN DƯ (residual)** của các cây trước.

```
   Dự đoán ban đầu = giá trị trung bình
   Lặp lại:
     ① Tính residual = thực tế − dự đoán hiện tại
     ② Train 1 cây NÔNG (max_depth 3) để dự đoán residual đó
     ③ Dự đoán mới = dự đoán cũ + learning_rate × cây mới
```

**Vì sao Boosting dùng cây nông còn Random Forest dùng cây sâu?** Bagging (RF) giảm *variance*
bằng cách trung bình hoá nhiều cây mạnh, không tương quan — mỗi cây cần đủ sâu để tự nó dự đoán
tốt. Boosting giảm *bias* bằng cách cộng dồn nhiều mô hình yếu (*weak learner*) học phần lỗi còn
lại — nếu cây đã sâu/mạnh ngay từ đầu, nó sẽ khớp gần hết residual của vòng đó, các vòng sau gần
như không còn gì để sửa và mô hình overfit rất nhanh. Cây nông giữ cho mỗi bước đóng góp nhỏ,
`learning_rate` kiểm soát mức đóng góp đó.

---

## 2. BÀI TOÁN THỰC TẾ

```
   Công ty tài chính tiêu dùng cần ước lượng KHẢ NĂNG TÀI CHÍNH của khách
   trước khi duyệt khoản vay trả góp, nhưng khách thường KHÔNG khai thu nhập
   hoặc khai không chính xác.

   → Dùng thông tin có thể kiểm chứng (nghề nghiệp, học vấn, giờ làm/tuần,
     tình trạng hôn nhân) để dự đoán khách có thu nhập > 50.000$/năm hay không.

   → Kết quả là 1 trong các đầu vào của mô hình chấm điểm tín dụng.
```

---

## 3. BỘ DỮ LIỆU

| | |
|---|---|
| **Tên** | Adult Census Income |
| **Link** | https://archive.ics.uci.edu/dataset/2/adult |
| **Kích thước** | `adult.data` (train) = 32.561 dòng, `adult.test` (test) = 16.281 dòng — dùng đúng bộ train/test chính thức của UCI, không tự split, để số liệu so được với mức tham chiếu công khai |
| **Nhãn** | `income` (`<=50K` / `>50K`) — **24,1%** là `>50K` trên train |

**Các cột dùng để huấn luyện (sau khi bỏ `fnlwgt`, `education`):** `age`, `workclass`,
`education-num`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `capital-gain`,
`capital-loss`, `hours-per-week`, `native-country`

### ⚠️ Bốn bẫy trong dữ liệu (đã xử lý trong `load_clean()`)

```
   1. Giá trị thiếu ghi bằng ' ?' (có DẤU CÁCH đứng trước!)
      → na_values='?' + skipinitialspace=True (KHÔNG chỉ replace('?'), vì dấu
        cách nằm TRƯỚC dấu '?' — replace thường không bắt được)

   2. Mọi giá trị chuỗi đều có dấu cách thừa ở đầu → .str.strip() trên mọi cột object

   3. adult.test có 1 dòng header rác đầu file VÀ nhãn kết thúc bằng dấu chấm
      ('>50K.' thay vì '>50K') — rất dễ làm sai nhãn nếu bỏ qua
      → skiprows=1 khi đọc adult.test, và .str.rstrip('.') trước khi so sánh nhãn

   4. education và education-num là CÙNG MỘT THÔNG TIN (một dạng chữ, một dạng số)
      → giữ cả hai là trùng lặp → bỏ education, giữ education-num (đã là số thứ tự)

   5. fnlwgt là trọng số thống kê dân số, KHÔNG liên quan tới cá nhân
      → bỏ, giữ lại chỉ gây nhiễu
```

---

## 4. HƯỚNG ĐI ĐÚNG

### 4.1. Ba siêu tham số then chốt

```
   learning_rate (η) — mỗi cây đóng góp bao nhiêu
        η nhỏ (0,05) + n_estimators LỚN (500)  = ⭐ TỐT NHẤT, ổn định
        η lớn (0,3)  + n_estimators nhỏ (100)  = nhanh nhưng dễ overfit

   n_estimators — số cây (quan hệ NGƯỢC với learning_rate)

   max_depth = 3 — cây phải NÔNG (weak learner)
        ⚠️ Khác hoàn toàn Random Forest (cây sâu, mạnh) — xem giải thích ở mục 1
```

> 💡 **Quy tắc vàng:** `learning_rate` NHỎ + `n_estimators` LỚN. Nếu train quá lâu,
> hãy chuyển sang **HistGradientBoostingClassifier** (nhanh hơn nhiều lần) hoặc LightGBM.

### 4.2. Code

```python
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier

gb = GradientBoostingClassifier(
    n_estimators=500, learning_rate=0.05, max_depth=3,
    subsample=0.8,              # Stochastic GB — chống overfit
    validation_fraction=0.1, n_iter_no_change=20,   # dừng sớm nội bộ, dùng train, không đụng test
    random_state=42,
)

# Bản NHANH cho dữ liệu lớn (xử lý được cả NaN và biến phân loại)
hgb = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05,
                                     early_stopping=True, random_state=42)
```

### 4.3. ⚠️ Nguyên tắc phương pháp: KHÔNG chọn siêu tham số trên test

Bản trước của bài này dò `learning_rate × n_estimators` và vẽ đường "validation loss" đều bằng
cách chấm điểm trực tiếp trên `X_test/y_test` — nghĩa là quyết định mô hình cuối cùng đã "nhìn
thấy" tập test trước khi báo cáo kết quả, làm ROC-AUC lạc quan hơn thực tế và không phản ánh đúng
hiệu năng khi triển khai với dữ liệu hoàn toàn mới. Bài này sửa lại theo nguyên tắc:

```
   Train (32.561 dòng)                                    Test (16.281 dòng)
   ├── dùng CROSS-VALIDATION (StratifiedKFold) để          ├── KHÔNG động tới cho tới
   │   dò lưới learning_rate × n_estimators                │   khi mọi lựa chọn đã chốt
   ├── tách riêng 20% làm X_val để vẽ đường train/val       ├── chỉ dùng ĐÚNG MỘT LẦN để
   │   loss (chẩn đoán overfit)                             │   báo cáo ROC-AUC/PR-AUC cuối
   └── validation_fraction nội bộ của GB (early stopping)   └── và để minh hoạ áp dụng
       dùng chính train, tách tự động, không phải rò rỉ         nghiệp vụ (ngưỡng theo chi phí)
```

---

## 5. CÁC BƯỚC THỰC HIỆN

```
   ☐ 1. Đọc dữ liệu, .str.strip() mọi cột chuỗi, na_values='?', skiprows=1 cho adult.test,
        rstrip('.') cho nhãn test
   ☐ 2. Bỏ fnlwgt và education (giữ education-num)
   ☐ 3. EDA: tỉ lệ >50K theo học vấn, tình trạng hôn nhân; capital-gain lệch cực đoan (91,7% = 0)
   ☐ 4. Tách 20% của TRAIN làm validation (KHÔNG đụng test) để dùng ở bước 6, 8
   ☐ 5. Pipeline: OneHot(cat) + passthrough(num)  [Boosting KHÔNG cần scale]
   ☐ 6. Baseline: DummyClassifier + Decision Tree (đánh giá trên test — hợp lệ vì không tuning)
   ☐ 7. ⭐ Dò learning_rate × n_estimators bằng CROSS-VALIDATION trên train (không phải trên test)
   ☐ 8. Gradient Boosting cuối cùng với siêu tham số đã chọn, fit trên toàn bộ train
   ☐ 9. ⭐ Vẽ TRAIN & VALIDATION loss theo số cây trên tập validation tách riêng
        (a) cấu hình đã regularize — đối chiếu xem có thực sự overfit trong phạm vi đã chọn không
        (b) cấu hình cố ý không regularize — minh hoạ overfit THẬT để so sánh
   ☐ 10. Đánh giá GradientBoosting trên test — CHỈ MỘT LẦN, sau khi đã chốt mọi lựa chọn
   ☐ 11. So sánh 3 thuật toán trên CÙNG dữ liệu:
         Random Forest (TT-03) vs Gradient Boosting vs AdaBoost (TT-09)
         → bảng: PR-AUC, ROC-AUC, thời gian train
   ☐ 12. Đo thời gian: GradientBoosting vs HistGradientBoosting (trong CÙNG một lần chạy)
   ☐ 13. ⭐ Gắn với nghiệp vụ: ma trận nhầm lẫn + chọn ngưỡng xác suất theo chi phí
         duyệt-nhầm/từ-chối-nhầm (không dùng mặc định 0.5)
   ☐ 14. ⚖️ Kiểm tra THIÊN LỆCH theo `sex` và `race`
```

```python
# Vẽ loss trên VALIDATION (tách từ train), không phải test
train_loss = [log_loss(y_tr, p) for p in fitted_gb.staged_predict_proba(Xtr_t)]
val_loss = [log_loss(y_val, p) for p in fitted_gb.staged_predict_proba(Xval_t)]
```

---

## 6. TIÊU CHÍ HOÀN THÀNH

```
   ☐ Xử lý đúng bẫy ' ?', dấu cách thừa, skiprows/rstrip('.') cho adult.test
   ☐ Có biểu đồ train/validation loss theo số cây, vẽ trên validation TÁCH RIÊNG khỏi test,
     có đối chiếu cấu hình regularize (chưa overfit) vs không regularize (overfit thật)
   ☐ Có bảng lưới learning_rate × n_estimators dò bằng CROSS-VALIDATION trên train
   ☐ Có bảng so sánh Bagging vs Boosting vs AdaBoost (kèm thời gian train, cùng 1 lần chạy)
   ☐ PR-AUC > baseline rõ rệt, test chỉ dùng đúng 1 lần cho báo cáo cuối
   ☐ Có ma trận nhầm lẫn và ngưỡng quyết định gắn với chi phí nghiệp vụ (duyệt vay)
   ☐ ⚖️ Có phân tích thiên lệch theo giới tính / chủng tộc
   ☐ Giải thích được vì sao Boosting dùng cây NÔNG còn RF dùng cây SÂU
```

**Mức tham chiếu:** ROC-AUC ~0,92–0,93 · Accuracy ~0,87. Chạy `notebooks/gradient_boosting_income.ipynb`
hoặc `src/train.py` để lấy số liệu thật trên máy của bạn (xem mục 7 — kết quả tham khảo).

---

## 7. KẾT QUẢ THAM KHẢO

Bộ code trong bản này **chưa được chạy** (chỉ giao mã nguồn, không kèm output/đường dẫn môi
trường). Các số liệu dưới đây là kết quả tham khảo từ một lần chạy thử nghiệm trước đó trên cùng
pipeline — khi bạn tự chạy `notebooks/gradient_boosting_income.ipynb` hoặc `src/train.py`, số liệu
có thể lệch nhẹ (do CV shuffle, thời gian train phụ thuộc máy...) nhưng về cơ bản sẽ nằm trong mức
tương tự.

### 7.1. Baseline vs Gradient Boosting (đánh giá trên test, 16.281 dòng)

| Model | PR-AUC | ROC-AUC |
|---|---|---|
| Dummy (most_frequent) | ~0,24 | 0,50 |
| Decision Tree (depth=6) | ~0,72 | ~0,90 |
| **Gradient Boosting** | **~0,82** | **~0,92–0,93** |
| HistGradientBoosting | ~0,83 | ~0,93 |

### 7.2. Dò lưới learning_rate × n_estimators (ROC-AUC trung bình, 3-fold CV trên **train**)

`learning_rate` nhỏ (0,05) cần `n_estimators` lớn (500) mới đạt hiệu năng tương đương
`learning_rate` lớn (0,3) với ít cây hơn — xác nhận quan hệ nghịch giữa hai tham số. Xem
`reports/lr_vs_nestimators.png` sau khi chạy để có bảng/heatmap số thật.

### 7.3. Bagging vs Boosting vs AdaBoost (đánh giá trên test)

Gradient Boosting thường vượt Random Forest về cả PR-AUC lẫn ROC-AUC với thời gian train ngắn hơn
đáng kể trên dữ liệu này; AdaBoost nhanh nhất nhưng độ chính xác thấp hơn GB. Bảng số liệu thật nằm
ở `reports/model_comparison.csv` sau khi chạy.

### 7.4. GradientBoosting vs HistGradientBoosting

HistGradientBoosting thường nhanh hơn GradientBoosting vài lần với ROC-AUC tương đương — con số cụ
thể phụ thuộc máy chạy, xem output notebook hoặc `reports/train_summary.csv`.

### 7.5. Đường train/validation loss (trên validation tách từ train, không phải test)

Notebook vẽ 2 kịch bản cạnh nhau để tránh gán nhãn "overfit" sai:
- **(a) Cấu hình đã regularize** (giống model cuối, `subsample=0.8`, `learning_rate` nhỏ): kỳ vọng
  loss validation vẫn còn giảm ở cây cuối — tức trong phạm vi `n_estimators` đã chọn, mô hình
  **chưa chắc đã overfit**; code tự tính `best_iter` và in ra vị trí thật thay vì giả định trước.
- **(b) Cấu hình cố ý KHÔNG regularize** (`learning_rate=0,3`, `subsample=1.0`, 1000 cây): dùng để
  minh hoạ overfit THẬT — loss validation dự kiến chạm đáy rồi tăng trở lại khi số cây tiếp tục
  tăng, đối lập với (a).

### 7.6. Ma trận nhầm lẫn & ngưỡng theo chi phí nghiệp vụ

Code dò ngưỡng xác suất tối thiểu hoá `chi phí = C_FP × FP + C_FN × FN` (mặc định `C_FP=1, C_FN=3`
— giả định từ chối nhầm khách tốt tốn kém hơn duyệt nhầm khách yếu; **cần thay bằng số thật từ đội
rủi ro tín dụng**). Kết quả cụ thể (ngưỡng tối ưu, 2 ma trận nhầm lẫn) nằm ở
`reports/threshold_cost.png` sau khi chạy.

### 7.7. Kiểm tra thiên lệch theo `sex` / `race` (trên test, ngưỡng 0,5)

Theo đặc điểm đã biết của bộ dữ liệu Adult Census 1994: nam giới và người White có tỉ lệ `>50K`
thực tế lẫn dự đoán cao hơn rõ rệt so với nữ giới và người Black. **Accuracy cao ở một nhóm không
đồng nghĩa với công bằng** — cần nhìn cả `positive_rate_pred` so với `positive_rate_true` của từng
nhóm, không chỉ accuracy. Số liệu thật nằm ở `reports/bias_by_group.csv` sau khi chạy.

---

## 8. CẠM BẪY

| Cạm bẫy | Hậu quả |
|---------|---------|
| `replace('?')` không có dấu cách | Không bắt được giá trị thiếu |
| Quên `skiprows=1` / `rstrip('.')` cho `adult.test` | Đọc nhầm dòng header, nhãn `'>50K.'` không khớp `'>50K'` → mọi mẫu >50K bị gán nhãn sai |
| Dò siêu tham số / vẽ loss bằng chính `X_test` | Rò rỉ dữ liệu — kết quả lạc quan hơn thực tế, không có cross-validation nào để đối chứng |
| Gán nhãn "overfit" cho điểm loss thấp nhất nằm gần cuối dải | Hiểu sai: loss còn đang giảm nghĩa là CHƯA overfit, không phải "bắt đầu overfit" |
| `max_depth` lớn (8–10) | Overfit ngay — sai bản chất boosting |
| `learning_rate` lớn + nhiều cây | Overfit nặng |
| Giữ cả `education` và `education-num` | Trùng lặp thông tin |
| Giữ `fnlwgt` | Nhiễu, không liên quan tới cá nhân |
| Ghi file (`joblib.dump`, `to_csv`) vào thư mục chưa tồn tại | `FileNotFoundError` khi chạy script/notebook trên máy sạch — luôn `mkdir(parents=True, exist_ok=True)` trước khi ghi |
| Commit dữ liệu thô (`adult.data`/`adult.test`, ~6MB) vào git | Repo phình to, dữ liệu có thể tải lại từ UCI — dùng `.gitignore` |
| Bỏ qua kiểm tra thiên lệch | Rủi ro đạo đức & pháp lý nghiêm trọng |
| Chỉ dùng ngưỡng mặc định 0,5 khi kết quả phục vụ quyết định duyệt/từ chối vay | Bỏ qua chi phí khác nhau giữa hai loại sai lầm (duyệt nhầm vs từ chối nhầm) |

---

## 9. SẢN PHẨM NỘP

```
TT-07-GradientBoosting-<HoTen>/
├── README.md                       ← có mục "Phương pháp" (mục 4.3), "Kết quả" (mục 7) và "Thiên lệch" (7.7)
├── .gitignore                      ← không commit data/ thô và artifact sinh ra
├── data/                           ← KHÔNG commit; tự tải adult.data/adult.test từ UCI vào đây
├── notebooks/gradient_boosting_income.ipynb
├── src/train.py                    ← chạy độc lập: python src/train.py (tự tạo models/, reports/)
├── models/gb_pipeline.joblib
├── reports/
│   ├── loss_theo_so_cay.png        (train/val loss — 2 kịch bản regularize vs không)
│   ├── lr_vs_nestimators.png       (heatmap CV trên train)
│   ├── threshold_cost.png          (ngưỡng theo chi phí + 2 ma trận nhầm lẫn)
│   ├── bias_by_group.png / .csv
│   ├── model_comparison.csv        (RF vs GB vs AdaBoost)
│   └── final_summary.csv / train_summary.csv
└── requirements.txt
```

> **Dữ liệu:** tải `adult.data` và `adult.test` từ https://archive.ics.uci.edu/dataset/2/adult,
> đặt vào `data/` (cùng cấp `notebooks/`, `src/`) trước khi chạy notebook hoặc `src/train.py`.
> Thư mục `data/` đã được thêm vào `.gitignore` — không commit dữ liệu thô vào git.

> ⚖️ **Cảnh báo đạo đức bắt buộc:** bộ dữ liệu này từ điều tra dân số Mỹ 1994,
> chứa **định kiến lịch sử** rõ rệt về giới tính và chủng tộc. Model sẽ **học và
> khuếch đại** các định kiến đó. Bài tập yêu cầu đo và báo cáo mức chênh lệch —
> tuyệt đối không dùng model này cho quyết định thật về con người.

---

## 10. MỞ RỘNG

```
   1. Thử LightGBM và so sánh: nhanh hơn bao nhiêu lần với cùng độ chính xác?
   2. Dùng SHAP để giải thích 3 hồ sơ cụ thể quanh ngưỡng quyết định (0,22)
   3. Thử ràng buộc công bằng: bỏ hẳn cột `sex`, `race` → điểm giảm bao nhiêu?
      Model có còn thiên lệch không? (gợi ý: vẫn có, qua biến thay thế như occupation)
   4. Thay chi phí giả định ở mục 7.6 bằng số liệu thật từ đội rủi ro tín dụng,
      dựng đường cong lợi nhuận kỳ vọng theo ngưỡng thay vì chỉ chi phí đơn thuần
```

**Tham khảo:** [Buổi 6 — Ensemble & Boosting](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-06-Ensemble-EndToEnd/Tai-Lieu/ly_thuyet_chi_tiet_buoi_06.md)
