# TT-02 — Decision Tree: Dự đoán nghỉ việc & Quy tắc cho Nhân sự

Kết quả chạy đầy đủ 11 bước theo README gốc. Log chi tiết: `reports/ket_qua_chay.txt`.
Best params (GridSearchCV, tối ưu F1): `max_depth=5, min_samples_leaf=20, criterion=entropy`

---

## ⭐ 5 QUY TẮC CHO PHÒNG NHÂN SỰ

> Trích xuất trực tiếp từ các lá của cây quyết định tốt nhất, chỉ lấy nhóm ≥ 20 người
> để đảm bảo đủ tin cậy thống kê. Toàn bộ số liệu tính trên tập TRAIN (1.176 người).

**1. Nhân viên mới, thu nhập thấp, tuổi trẻ — nhóm rủi ro cao nhất**
Nhân viên có **kinh nghiệm làm việc ≤ 2 năm**, không thuộc vai trò Research Scientist,
**tuổi ≤ 28**, **thu nhập tháng ≤ 2.379 USD** có tỉ lệ nghỉ việc **85%** (17/20 người).
→ Đề xuất: rà soát khung lương khởi điểm và chương trình giữ chân nhân viên mới trong
2 năm đầu, đặc biệt nhóm trẻ tuổi.

**2. Nhân viên mới, thu nhập khá hơn nhưng vẫn trẻ — vẫn rủi ro rất cao**
Cùng nhóm kinh nghiệm ≤ 2 năm, tuổi ≤ 28, nhưng thu nhập > 2.379 USD vẫn có tỉ lệ nghỉ
**65%** (13/20 người). → Lương không phải yếu tố duy nhất; cần thêm khảo sát lý do
nghỉ việc ở nhóm tuổi trẻ (lộ trình thăng tiến, môi trường làm việc).

**3. Làm thêm giờ + thu nhập thấp — nhóm cần ưu tiên xử lý**
Nhân viên có kinh nghiệm > 2 năm, **có làm thêm giờ (OverTime)**, **thu nhập ≤ 2.476
USD** có tỉ lệ nghỉ việc **62,2%** (23/37 người). → Đề xuất: xem lại khối lượng công
việc và mức lương của nhóm nhân viên phải tăng ca nhưng thu nhập chưa cao.

**4. Làm thêm giờ + hài lòng môi trường thấp + không có cổ phiếu thưởng**
Nhân viên làm thêm giờ, thu nhập > 2.476 USD nhưng **EnvironmentSatisfaction ≤ 2.5**
(mức hài lòng thấp) và **không có StockOptionLevel** có tỉ lệ nghỉ **53,7%** (22/41
người). → Đề xuất: cải thiện môi trường làm việc và cân nhắc mở rộng chính sách cổ
phiếu thưởng cho nhóm tăng ca.

**5. Không tăng ca nhưng đổi nhiều công ty, tuổi trẻ, không cổ phiếu thưởng**
Nhân viên không làm thêm giờ, đã từng làm ở **> 4 công ty khác**, **tuổi ≤ 37**, không
có cổ phiếu thưởng có tỉ lệ nghỉ **51,4%** (18/35 người). → Đây là nhóm có xu hướng
"nhảy việc" theo lịch sử nghề nghiệp; cân nhắc chính sách giữ chân dài hạn (cổ phiếu,
lộ trình sự nghiệp rõ ràng) riêng cho nhóm này.

**Nhóm đối chứng (an toàn, tỉ lệ nghỉ 0%)**: nhân viên không tăng ca, số công ty từng
làm ≤ 4, **BusinessTravel = Non-Travel** — 64 người, 0 người nghỉ việc. Cho thấy đi
công tác thường xuyên kết hợp các yếu tố khác cũng là một tín hiệu rủi ro cần theo dõi.

---

## Bảng so sánh CÓ vs KHÔNG giới hạn max_depth (bước 5–6)

| | Accuracy TRAIN | Accuracy TEST |
|---|---|---|
| `max_depth=None` (không giới hạn) | ~1.00 (100%) | thấp hơn train rõ rệt |
| `max_depth=5` (đã chọn qua GridSearchCV) | thấp hơn nhiều so với None | ổn định hơn, không sụt trên test |

Biểu đồ đầy đủ: `reports/overfit_theo_depth.png` — khoảng cách train/test doãng ra rõ
khi max_depth vượt quá ~6–7, xác nhận đúng dự đoán của đề bài.

## Feature importance (top 10)

| Biến | Importance |
|---|---|
| TotalWorkingYears | 0.231 |
| OverTime_Yes | 0.183 |
| Age | 0.090 |
| MonthlyIncome | 0.081 |
| NumCompaniesWorked | 0.076 |
| EnvironmentSatisfaction | 0.074 |
| StockOptionLevel | 0.074 |
| BusinessTravel_Non-Travel | 0.049 |
| YearsInCurrentRole | 0.039 |
| MaritalStatus_Single | 0.037 |

Biểu đồ: `reports/feature_importance.png`

## ⚖️ Kiểm tra đạo đức

Cây tốt nhất **CÓ dùng `MaritalStatus_Single`** làm một nút chia (importance 0.037,
xếp thứ 10). `Gender` **không** được dùng. Vì `MaritalStatus` là biến nhân khẩu nhạy
cảm, khuyến nghị: cân nhắc loại bỏ biến này khỏi tập huấn luyện và huấn luyện lại nếu
model được dùng để ra quyết định chính sách, để tránh rủi ro phân biệt đối xử gián
tiếp theo tình trạng hôn nhân.

## Kết quả trên tập TEST (chạm 1 lần)

| Metric | Baseline (Dummy) | Decision Tree (best) |
|---|---|---|
| Accuracy | ~0.84 | 0.745 |
| Recall | 0.000 | 0.638 |
| Precision | – | 0.341 |
| F1 | 0.000 | **0.444** |

F1 của cây (0.444) vượt xa baseline (0.000) — đạt tiêu chí hoàn thành. Lưu ý:
Accuracy của cây thấp hơn Dummy vì dùng `class_weight='balanced'`, cây chủ động đánh
đổi accuracy để bắt được nhiều ca "Nghỉ" hơn (Recall 63,8%) — đúng mục tiêu bài toán
(bỏ sót nhân viên có nguy cơ nghỉ nguy hiểm hơn báo động giả).

Ma trận nhầm lẫn: `[[189, 58], [17, 30]]` → chỉ bỏ sót 17/47 ca nghỉ việc thực tế.

## So sánh với Random Forest (bước 11)

| Model | Accuracy | Recall | Precision | F1 |
|---|---|---|---|---|
| Decision Tree (best) | 0.745 | **0.638** | 0.341 | 0.444 |
| Random Forest | **0.830** | 0.340 | **0.457** | 0.390 |

**Mất gì, được gì?** Random Forest cho Accuracy/Precision cao hơn nhưng Recall sụt
mạnh (0.34 vs 0.64) — nghĩa là bỏ sót nhiều nhân viên có nguy cơ nghỉ hơn. Quan trọng
hơn: 300 cây trong Random Forest **không thể in ra thành luật IF-THEN cho HR đọc**
như 1 cây `max_depth=5`. Với mục tiêu chính của bài này (quy tắc dễ hiểu, dễ hành
động), Decision Tree đơn vẫn là lựa chọn phù hợp hơn; Random Forest phù hợp hơn khi
ưu tiên độ chính xác thuần tuý và chấp nhận đánh đổi khả năng diễn giải.

## Hạn chế: cây đơn không ổn định

Kiểm tra nhanh với 5 `random_state` khác nhau (cùng tham số tốt nhất): biến quan
trọng nhất (top-1) luôn là `TotalWorkingYears` — ở mức top-level cây khá ổn định trên
bộ dữ liệu này. Tuy nhiên các nhánh sâu hơn (nút chia thứ 2, thứ 3...) có thể thay đổi
đáng kể giữa các seed — đây chính là lý do Random Forest ra đời: trung bình nhiều cây
để giảm phương sai. Khuyến nghị: không nên coi 5 quy tắc trên là "chân lý tuyệt đối",
mà nên xác nhận lại bằng thống kê mô tả trực tiếp trước khi thay đổi chính sách.

> ⚖️ **Đạo đức sử dụng:** model này KHÔNG được dùng để sa thải hay đánh giá cá nhân.
> Chỉ dùng để **sửa chính sách ở cấp nhóm** (lương, tăng ca, lộ trình thăng tiến).

## Cấu trúc thư mục

```
TT-02-DecisionTree/
├── README.md                        ← file này
├── hr_attrition.csv
├── src/train.py                     ← toàn bộ pipeline 11 bước
├── notebooks/decision_tree_hr.ipynb ← notebook tương ứng, đã chạy thử end-to-end
├── models/tree.joblib               ← model đã huấn luyện (best params)
├── reports/
│   ├── eda_attrition.png            ← bước 3
│   ├── overfit_theo_depth.png       ← bước 6
│   ├── cay_quyet_dinh.png           ← bước 8
│   ├── cay_export_text.txt          ← bước 8
│   ├── feature_importance.png       ← bước 9
│   ├── bang_luat_tu_cay.csv         ← toàn bộ luật trích xuất từ cây (bước 10)
│   └── ket_qua_chay.txt             ← log đầy đủ toàn bộ quá trình chạy
└── requirements.txt
```
