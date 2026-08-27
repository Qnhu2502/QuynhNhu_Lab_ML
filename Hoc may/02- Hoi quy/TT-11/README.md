# TT-11 — Linear Regression: Định giá nhà (California Housing)

## 1. Kết quả model
- Baseline RMSE: 1.145
- Linear Regression: RMSE=0.679, MAE=0.498, R2=0.648

## 2. Kiểm tra 4 giả định
- Tuyến tính: xem `reports/residual_plot.png`
- Phương sai đều: residual plot có dạng loe ra ở giá cao -> vi phạm nhẹ
- Sai số chuẩn: xem `reports/qq_plot.png`
- Đa cộng tuyến (VIF):
   feature        VIF
 Longitude 814.296388
  Latitude 601.328010
 AveBedrms  83.082357
  AveRooms  53.416616
  AveOccup  18.338949
    MedInc  13.701064
  HouseAge   7.389340
Population   3.007872

## 3. Bẫy dữ liệu đã xử lý
- Nhãn bị cắt ngọn ở 5.0: 992 dòng (4.81%)
- Outlier AveRooms/AveBedrms/AveOccup: clip theo phân vị 99

## 4. Hệ số đã chuẩn hoá (3 yếu tố mạnh nhất)
Longitude   -0.973245
MedInc       0.793000
Latitude    -0.759841

## 5. Hạn chế
- Model không dự đoán chính xác nhà giá trên 500k USD do nhãn bị cắt ngọn.
- Quan hệ toạ độ–giá là phi tuyến, Linear Regression bắt kém hơn Random Forest.
