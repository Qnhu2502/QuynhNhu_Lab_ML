# TT-12 - Ridge Regression: phân bổ ngân sách quảng cáo đa kênh

Đề bài gốc: `DE_BAI.md`.

## Các bước thực hiện

1. **Nạp và làm sạch dữ liệu.** Giữ 4546 dòng sau khi loại các dòng thiếu; one-hot biến Influencer.
2. **Tương quan và VIF** (có cột hằng số).
3. **Chia train/test 80/20**, dựng Dummy baseline và Linear Regression.
4. **RidgeCV** chọn alpha bằng KFold 5 lần (có xáo trộn) trên train: alpha = 0.0518.
5. **Bootstrap 100 lần** (mẫu con 80%) so độ ổn định hệ số Linear và Ridge.
6. **Thí nghiệm bổ sung** trên dữ liệu tự sinh có đa cộng tuyến nặng.
7. **Coefficient path** và **RMSE theo alpha**.
8. **So sánh** Dummy, Linear, Ridge, Lasso, ElasticNet trên tập test.
9. **ROI theo đơn vị chi tiêu** có khoảng tin cậy 95%, kiểm tra bão hoà (TV^2).
10. **Đề xuất phân bổ ngân sách** và lưu model.

## Kết quả và nhận xét

### Đa cộng tuyến (VIF)
| feature | VIF |
|---|---|
| Radio | 4.67 |
| TV | 4.09 |
| Social Media | 1.58 |
| Influencer_Mega | 1.52 |
| Influencer_Micro | 1.52 |
| Influencer_Nano | 1.52 |

Nhận xét: VIF lớn nhất là 4.67, chưa đạt ngưỡng 10: đa cộng tuyến chỉ ở mức vừa trên dữ liệu thật.

### Độ chính xác trên tập test
|  | RMSE | MAE | R2 |
|---|---|---|---|
| Dummy (trung bình) | 91.7197 | 79.544 | -0.0002 |
| Linear | 2.8849 | 2.3129 | 0.999 |
| Ridge (CV) | 2.8849 | 2.3129 | 0.999 |
| Lasso (CV) | 2.8831 | 2.3129 | 0.999 |
| ElasticNet (CV) | 2.8828 | 2.3115 | 0.999 |

Nhận xét: mô hình giảm RMSE từ 91.7 (baseline) xuống 2.88.
Ridge, Linear, Lasso và ElasticNet gần như bằng nhau vì tín hiệu chủ yếu đến từ TV.

### Độ ổn định hệ số (dữ liệu thật)
Nhận xét: Ridge và Linear gần như bằng nhau (tỉ lệ độ lệch chuẩn Ridge/Linear từ 1.00 đến 1.00); lý do: VIF thấp, n lớn, alpha CV nhỏ.

### Dữ liệu tự sinh (VIF = 26.4), độ lệch chuẩn hệ số
|  | Linear | Ridge alpha CV (0.49) | Ridge alpha=10 |
|---|---|---|---|
| TV | 7.81 | 7.22 | 3.97 |
| Facebook | 6.16 | 5.75 | 3.89 |
| Google | 2.7 | 2.7 | 2.68 |

Nhận xét: alpha lớn làm hệ số ổn định hơn nhưng bị chệch; không nên chọn alpha chỉ để hệ số "đẹp".
Ridge không đưa hệ số về đúng 0 vì đạo hàm phạt L2 là `2λw` tiến về 0 cùng `w`; Lasso mới cho hệ số bằng 0.

### ROI theo kênh (Sales trên 1 đơn vị ngân sách)
|  | ROI (Sales / 1 đv ngân sách) | CI95 thấp | CI95 cao | CI chứa 0? |
|---|---|---|---|---|
| TV | 3.5632 | 3.5589 | 3.5665 | False |
| Radio | -0.0021 | -0.011 | 0.0092 | True |
| Social Media | -0.0146 | -0.0426 | 0.0104 | True |

Nhận xét: kênh có tác động tách khỏi 0: TV; khoảng tin cậy chứa 0: Radio, Social Media.
Kiểm tra bão hoà TV^2: p = 0.405 (chưa có bằng chứng bão hoà).

### Đề xuất phân bổ ngân sách 2 tỷ/tháng
|  | Tỷ lệ hiện tại | Tỷ lệ đề xuất | Ngân sách đề xuất (triệu/tháng) | Ngân sách hiện tại tương đương (triệu/tháng) |
|---|---|---|---|---|
| TV | 71.6 | 86.6 | 1731.0 | 1431.0 |
| Radio | 24.0 | 8.4 | 169.0 | 481.0 |
| Social Media | 4.4 | 5.0 | 100.0 | 88.0 |

Nhận xét: không dồn 100% vào một kênh vì mô hình tuyến tính không có bão hoà, dồn toàn bộ là ngoại suy ngoài khoảng dữ liệu,
và cần giữ nhóm đối chứng cho A/B test. Mô hình dự báo Sales thay đổi +21.0%
(CI95 +20.9% đến +21.1%, chỉ phản ánh sai số lấy mẫu) so với doanh số trung bình.

## Hạn chế

- Hệ số hồi quy không phải quan hệ nhân quả; cần geo-test hoặc A/B test.
- Dữ liệu dummy, Sales gần như tuyến tính hoàn toàn theo TV; không nên suy rộng.
- Không có cột thời gian nên chưa mô hình hoá được hiệu ứng trễ (adstock).
- Giả định ngân sách và Sales cùng đơn vị (triệu/kỳ) cần được xác nhận.
