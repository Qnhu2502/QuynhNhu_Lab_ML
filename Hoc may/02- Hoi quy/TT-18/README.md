# TT-18 — Gradient Boosting Regressor: Thẩm định giá nhà tự động (AVM)

## Các bước đã thực hiện

1. Thống kê giá trị thiếu (19 cột có NaN), phân loại 2 nhóm: 15 cột "không có tiện ích" (PoolQC, Alley, FireplaceQu...) và 4 cột thiếu dữ liệu thật (LotFrontage, MasVnrArea, GarageYrBlt, Electrical).
2. Điền 'None' cho nhóm không có tiện ích, điền median/mode cho nhóm thiếu thật — không xoá cột, không one-hot nhầm nhóm.
3. Mã hoá thứ tự (OrdinalEncoder thủ công) cho 15 cột chất lượng/thứ tự (ExterQual, KitchenQual, BsmtExposure...) theo đúng thang Ex>Gd>TA>Fa>Po.
4. One-hot các biến danh mục thuần còn lại.
5. Kiểm tra độ lệch SalePrice (skew gốc 1.88) → áp dụng log1p để huấn luyện.
6. Baseline: DummyRegressor, Linear Regression, RidgeCV trên thang log.
7. Gradient Boosting với early stopping (n_iter_no_change=50), dừng ở 378 cây.
8. Vẽ train/validation loss theo số cây để xác định điểm overfit (xem `reports/loss_theo_so_cay.png`).
9. Feature engineering (TotalSF, TuoiNha, DaSuaChua): RMSE(log) giảm từ 0.1386 xuống 0.1385.
10. Hồi quy phân vị 10/50/90 cho khoảng giá (xem `reports/khoang_gia.png`).
11. Median APE: 6.20% (mục tiêu README < 12%) — xem `reports/ape_distribution.png`.
12. Cơ chế human-in-the-loop: 56.16% hồ sơ tự động định giá (khoảng dự báo ≤ 25%), 43.84% chuyển thẩm định viên.
13. So sánh thời gian train: GradientBoosting 5.90s vs HistGradientBoosting 3.98s.

## Hạn chế
- Dữ liệu chỉ 1.460 căn tại Ames, Iowa (2006–2010), không đại diện thị trường khác.
- Hồi quy phân vị huấn luyện độc lập từng phân vị, đôi khi khoảng không đơn điệu hoàn hảo.
- Median APE là chỉ số tổng thể, không đảm bảo đồng đều ở mọi phân khúc giá.
