# TT-17 — Random Forest Regressor: Dự đoán giá vé máy bay

## Các bước đã thực hiện

1. Nạp dữ liệu `Clean_Dataset.csv`, bỏ cột `Unnamed: 0` và `flight` (mã chuyến bay có hàng nghìn giá trị duy nhất, one-hot sẽ tạo hàng nghìn cột và cây học thuộc mã chuyến thay vì học quy luật giá).
2. EDA: giá theo `days_left` (xem `reports/gia_theo_days_left.png`) và boxplot giá theo `class`/`airline`.
3. Xây pipeline `OneHotEncoder` cho biến phân loại, giữ nguyên `duration`/`days_left` (cây không cần chuẩn hoá).
4. Baseline: DummyRegressor, Linear Regression, 1 cây quyết định đơn.
5. Random Forest (n_estimators=300, max_features=1.0, min_samples_leaf=2) — OOB score: 0.9862
6. Khảo sát RMSE theo số cây (xem `reports/rmse_theo_so_cay.png`) để tìm điểm bão hoà.
7. Permutation importance (không dùng `feature_importances_` mặc định vì thiên vị biến nhiều mức).

## Partial Dependence Plot (PDP) cho days_left
Xem `reports/pdp_days_left.png`.
Mua sớm 7 ngày thay vì 1 ngày trước chuyến bay tiết kiệm trung bình khoảng 6,003 Rupee.

## Khoảng dự báo 10–90%
Xem `reports/khoang_du_bao.png`.
Tỉ lệ giá thật rơi trong khoảng dự báo 10–90%: 86.23%.

## Thí nghiệm ngoại suy
Dự đoán tại days_left=100 (ngoài dải train) gần như bằng giá tại days_left lớn nhất trong tập train
(19,324 vs 19,324 Rupee) → xác nhận Random Forest KHÔNG ngoại suy được, dự đoán bị kẹp trần.

## So sánh với XGBoost
Random Forest RMSE=2,699, R2=0.9859
XGBoost RMSE=3,460, R2=0.9768

## Hạn chế
- Không ngoại suy được ngoài dải `days_left` đã train.
- Dữ liệu Ấn Độ 2022, không áp dụng trực tiếp cho thị trường Việt Nam.
- `class` chi phối giá mạnh nhất — nên cân nhắc tách 2 model riêng cho Economy/Business.
