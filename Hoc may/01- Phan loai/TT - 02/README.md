# TT-02 — DECISION TREE (CÂY QUYẾT ĐỊNH)
## Dự đoán nhân viên nghỉ việc & rút ra QUY TẮC cho phòng Nhân sự

| | |
|---|---|
| 🎓 **Khoá** | HỌC MÁY · [Buổi 3](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-03-Feature-Eng-Tree) |
| 🧠 **Nhóm** | Phân loại có giám sát |
| 🔧 **Thuật toán** | Decision Tree (CART) |
| 🏭 **Lĩnh vực** | Nhân sự · Quản trị nhân tài |
| ⏱ **Thời lượng** | 5–7 giờ |
| 📈 **Độ khó** | ⭐⭐ |

---

## 1. THUẬT TOÁN NÀY LÀ GÌ

```
   Cây hỏi liên tiếp các câu hỏi CÓ/KHÔNG để chia nhỏ dữ liệu:

                  Làm thêm giờ?
                 ╱            ╲
              CÓ                KHÔNG
             ╱                     ╲
      Thâm niên < 2 năm?         [Ở LẠI 92%]
       ╱          ╲
     CÓ            KHÔNG
   [NGHỈ 68%]    [Ở LẠI 71%]

   Mỗi lần chia, cây chọn câu hỏi làm giảm "độ hỗn loạn" nhiều nhất
   → đo bằng GINI hoặc ENTROPY.
```

**Giá trị lớn nhất của thuật toán này:** nó cho ra **LUẬT rõ ràng** mà người không
biết kỹ thuật cũng đọc được. Với phòng Nhân sự, một cây sâu 3 tầng có ích hơn
một model chính xác hơn 3% nhưng không giải thích được.

---

## 2. BÀI TOÁN THỰC TẾ

```
   Công ty 1.470 nhân viên, tỉ lệ nghỉ việc 16%/năm.
   Chi phí thay thế 1 nhân viên ≈ 6–9 tháng lương.

   Phòng Nhân sự KHÔNG cần một con số xác suất.
   Họ cần biết: "NHÓM NÀO đang có nguy cơ và VÌ SAO?"
   → để sửa chính sách: lương, tăng ca, lộ trình thăng tiến.

   👉 Sản phẩm cuối cùng của bài này là 5 QUY TẮC bằng tiếng Việt,
      không phải file .joblib.
```

---

## 3. BỘ DỮ LIỆU

| | |
|---|---|
| **Tên** | IBM HR Analytics Employee Attrition |
| **Link** | https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset |
| **Kích thước** | 1.470 dòng × 35 cột |
| **Nhãn** | `Attrition` (Yes/No) — khoảng **16% Yes** |

**Nhóm cột:** thu nhập (`MonthlyIncome`, `PercentSalaryHike`), công việc (`JobRole`,
`JobLevel`, `OverTime`), thâm niên (`YearsAtCompany`, `YearsSinceLastPromotion`),
hài lòng (`JobSatisfaction`, `WorkLifeBalance`), nhân khẩu (`Age`, `Gender`,
`MaritalStatus`)

### ⚠️ Bẫy: 4 cột vô dụng phải bỏ

```
   EmployeeCount      = 1 cho mọi dòng      → 0 thông tin
   StandardHours      = 80 cho mọi dòng     → 0 thông tin
   Over18             = 'Y' cho mọi dòng    → 0 thông tin
   EmployeeNumber     = mã định danh        → cây sẽ HỌC THUỘC mã nhân viên!

   → Luôn kiểm tra: df.nunique() để tìm cột chỉ có 1 giá trị.
```

---

## 4. HƯỚNG ĐI ĐÚNG

### 4.1. Cây KHÔNG cần chuẩn hoá — nhưng cần chống overfit

```
   Cây để mặc định (max_depth=None) sẽ mọc tới khi mỗi lá 1 mẫu
   → accuracy TRAIN = 100%, accuracy TEST tệ → OVERFIT kinh điển.

   3 cách cắt tỉa:
     max_depth        = 3–6      ← quan trọng nhất, cũng giúp cây ĐỌC ĐƯỢC
     min_samples_leaf = 20–50    ← mỗi lá phải đủ người mới đáng tin
     ccp_alpha        > 0        ← cắt tỉa theo chi phí-độ phức tạp
```

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text

tree = DecisionTreeClassifier(
    max_depth=4,               # đủ nông để in ra giấy A4 cho HR đọc
    min_samples_leaf=30,
    class_weight='balanced',   # xử lý lệch 16/84
    random_state=42,
)
```

### 4.2. Xuất LUẬT — phần quan trọng nhất

```python
print(export_text(tree, feature_names=list(X.columns)))   # dạng văn bản

import matplotlib.pyplot as plt
plt.figure(figsize=(22, 10))
plot_tree(tree, feature_names=X.columns, class_names=['Ở lại','Nghỉ'],
          filled=True, rounded=True, fontsize=9)
plt.savefig('reports/cay_quyet_dinh.png', dpi=150, bbox_inches='tight')
```

### 4.3. Gini vs Entropy

```
   Gini    = 1 − Σpᵢ²        tính nhanh hơn      ← MẶC ĐỊNH, dùng cái này
   Entropy = −Σpᵢ·log₂(pᵢ)   nhạy hơn một chút

   Thực tế 2 cái cho kết quả gần như giống nhau. Đừng tốn thời gian tranh cãi.
```

---

## 5. CÁC BƯỚC THỰC HIỆN

```
   ☐ 1. df.nunique() → phát hiện & bỏ 4 cột vô dụng
   ☐ 2. Mã hoá biến phân loại (OneHotEncoder trong Pipeline)
   ☐ 3. EDA: tỉ lệ nghỉ theo OverTime, JobRole, YearsAtCompany
   ☐ 4. Baseline: DummyClassifier
   ☐ 5. Cây KHÔNG giới hạn độ sâu → ghi accuracy train & test
        ⚠️ Phải thấy train ≈ 100%, test thấp → CHỨNG MINH overfit
   ☐ 6. Vẽ đường accuracy train/test theo max_depth = 1..20
        → chỉ ra điểm bắt đầu overfit
   ☐ 7. GridSearchCV: max_depth, min_samples_leaf, criterion (scoring='f1')
   ☐ 8. Vẽ cây tốt nhất, xuất export_text
   ☐ 9. Feature importance top 10
   ☐ 10. ✍️ VIẾT 5 QUY TẮC bằng tiếng Việt cho phòng Nhân sự
   ☐ 11. So sánh với Random Forest (TT-03) → mất gì, được gì?
```

---

## 6. TIÊU CHÍ HOÀN THÀNH

```
   ☐ Có biểu đồ train/test accuracy theo max_depth → chỉ rõ vùng overfit
   ☐ Có hình cây quyết định ĐỌC ĐƯỢC (max_depth ≤ 5)
   ☐ Có ≥ 5 quy tắc viết bằng tiếng Việt, mỗi quy tắc kèm % và số người
   ☐ Có bảng feature importance
   ☐ F1 trên tập test > baseline
   ☐ Nêu được hạn chế: cây đơn KHÔNG ỔN ĐỊNH (đổi chút dữ liệu → cây khác hẳn)
```

**Ví dụ quy tắc đạt yêu cầu:**
> *"Nhân viên **làm thêm giờ** + **thâm niên dưới 2 năm** + **mức lương thuộc nhóm thấp nhất**
> có tỉ lệ nghỉ việc **68%** (34/50 người trong nhóm này đã nghỉ). Đề xuất: rà soát lại
> khối lượng công việc và khung lương của nhóm nhân viên mới."*

---

## 7. CẠM BẪY

| Cạm bẫy | Hậu quả |
|---------|---------|
| Để `max_depth=None` | Overfit nặng, cây hàng trăm nút không ai đọc nổi |
| Giữ `EmployeeNumber` | Cây học thuộc mã nhân viên |
| Quên `class_weight` | Cây dự đoán "ở lại" hết vẫn được 84% accuracy |
| Tin tuyệt đối vào 1 cây | Cây rất không ổn định — đổi seed là ra cây khác |
| Nhầm tương quan với nhân quả | "Làm thêm giờ → nghỉ việc" chỉ là liên hệ, không phải nguyên nhân đã chứng minh |

---

## 8. SẢN PHẨM NỘP

```
TT-02-DecisionTree-<HoTen>/
├── README.md                      ← ⭐ phải có mục "5 QUY TẮC CHO PHÒNG NHÂN SỰ"
├── notebooks/decision_tree_hr.ipynb
├── src/train.py
├── models/tree.joblib
├── reports/{cay_quyet_dinh.png, overfit_theo_depth.png, feature_importance.png}
└── requirements.txt
```

> ⚖️ **Đạo đức:** model này KHÔNG được dùng để sa thải hay đánh giá cá nhân.
> Nó dùng để **sửa chính sách ở cấp nhóm**. Kiểm tra xem cây có dùng `Gender`,
> `MaritalStatus` làm nút chia không — nếu có, phải cân nhắc bỏ để tránh phân biệt đối xử.

---

## 9. MỞ RỘNG

```
   1. So sánh độ ổn định: train 10 cây với 10 random_state → cây có giống nhau không?
      (đây chính là lý do sinh ra Random Forest — xem TT-03)
   2. Thử ccp_alpha (cost-complexity pruning) thay vì max_depth
   3. Chuyển cây thành bộ luật IF-THEN và cài vào Excel cho HR dùng offline
```

**Tham khảo:** [Buổi 3 — Feature Engineering & Tree](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-03-Feature-Eng-Tree/Tai-Lieu)
