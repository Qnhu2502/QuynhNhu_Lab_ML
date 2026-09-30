# TT-26 - CNN: Sàng lọc viêm phổi trên ảnh X-quang ngực

## ⚠️ CẢNH BÁO Y TẾ
Đây là công cụ **SẮP THỨ TỰ ƯU TIÊN ĐỌC PHIM**, **KHÔNG PHẢI** công cụ chẩn đoán.
Dữ liệu từ Quảng Châu (Trung Quốc), trên bệnh nhi 1-5 tuổi - **KHÔNG tổng quát hoá** cho người lớn hay dân số khác.
Mọi ca đều phải có bác sĩ đọc lại.

## Hướng dẫn chạy
1. Cài thư viện: `pip install -r requirements.txt`
2. Tải dataset Kaggle `paultimothymooney/chest-xray-pneumonia`, giải nén thành `chest_xray/train|val|test/NORMAL|PNEUMONIA/`
   đặt cạnh notebook (hoặc đặt biến môi trường `CHEST_XRAY_DIR` trỏ tới thư mục `chest_xray`).
3. Restart Kernel rồi Run All (nên chạy trên máy có GPU).
4. Kết quả được lưu vào `models/` và `reports/`.

## Các bước đã thực hiện và kết quả
1. **Đếm ảnh:** {'train': {'NORMAL': 1341, 'PNEUMONIA': 3875}, 'val': {'NORMAL': 8, 'PNEUMONIA': 8}, 'test': {'NORMAL': 234, 'PNEUMONIA': 390}}
   Xác nhận 3 vấn đề: val gốc chỉ 16 ảnh; train mất cân bằng (PNEUMONIA 74.3%); phân phối test khác train (PNEUMONIA 62.5%).
2. **Validation:** bỏ val gốc, tự tách 15% từ train bằng `stratify` (train 4433 / val 783 / test 624). Dùng `class_weight` = {0: 1.944, 1: 0.673}.
3-4. **Pipeline `tf.data`:** resize 224x224, augmentation chỉ cho train: xoay ±10°, dịch ±10%, zoom ±10%, đổi độ sáng/tương phản nhẹ.
   **Không lật ngang** (tim nằm bên trái - lật ngang là sai giải phẫu) và không lật dọc/biến dạng mạnh.
   Ảnh giữ thang 0-255; EfficientNetB0 tự chuẩn hoá bên trong, CNN tự xây có `Rescaling(1/255)` trong model.
5-7. **So sánh 3 cách trên VALIDATION** (mỗi model dùng ngưỡng riêng để đạt recall val >= 0.97; test chưa được dùng để so sánh):

| Mô hình | nguong | recall | specificity | precision | auc | bo_sot_FN | bao_dong_gia_FP |
|---|---|---|---|---|---|---|---|
| CNN từ đầu | 0.1400 | 0.9708 | 0.6766 | 0.8968 | 0.9668 | 17 | 65 |
| Transfer Learning (đóng băng) | 0.1000 | 0.9742 | 0.8756 | 0.9578 | 0.9861 | 15 | 25 |
| Fine-tuning | 0.1300 | 0.9725 | 0.9453 | 0.9809 | 0.9928 | 16 | 11 |

   Huấn luyện dùng `EarlyStopping` + `ModelCheckpoint` theo `val_auc`; lịch sử lưu ở `reports/history_*.json`.
   Model được chọn: **Fine-tuning** (specificity val cao nhất tại recall val >= 0.97).
8. **Learning curves:** `reports/learning_curves.png` (transfer learning 2 giai đoạn), `reports/learning_curves_cnn_tu_dau.png`.
9. **Ngưỡng** chọn trên validation để đạt Recall >= 0.97: **0.13**.
10. **Test (đánh giá đúng 1 lần, model đã chọn):**

| | Giá trị |
|---|---|
| Ca viêm phổi bị **BỎ SÓT** | **3 / 390** |
| Báo động giả (NORMAL bị gắn cờ) | 101 / 234 |
| Recall | 0.9923 (ĐẠT yêu cầu >= 0.96) |
| Specificity | 0.5684 |
| Precision | 0.7930 |
| Accuracy | 0.8333 |
| AUC | 0.9685 |

   Accuracy thấp hơn recall: ngưỡng được chọn để ưu tiên recall (đánh đổi bằng báo động giả) và phân phối test khác train.
   Ma trận nhầm lẫn: `reports/confusion_matrix.png`.
11. **Grad-CAM** (`reports/gradcam_examples.png`): trên 96 ảnh test ngẫu nhiên, nhiệt Grad-CAM ở vùng rìa ảnh trung bình = 0.41
    (nếu nhiệt rải đều thì ≈ 0.41); 8.3% ảnh có >60% nhiệt ở rìa.
    Nhận xét: (chưa điền - xem ảnh rồi điền biến NHAN_XET_GRADCAM trong notebook)
12. **10 ca dự đoán sai** (`reports/ca_du_doan_sai.png`): tổng 3 ca bỏ sót + 101 báo động giả trên test.
    Nhận xét: (chưa điền - xem ảnh rồi điền biến NHAN_XET_CA_SAI trong notebook)

## Hạn chế
- Recall cao không thay thế được bác sĩ - công cụ chỉ sắp ưu tiên đọc phim.
- Dữ liệu chỉ từ 1 bệnh viện, bệnh nhi 1-5 tuổi, không đại diện dân số khác.
- Tách train/val theo ảnh (stratify), không theo bệnh nhân; một bệnh nhân có thể có nhiều ảnh ở cả train và val nên val có thể lạc quan hơn thực tế.
- Val vừa dùng cho EarlyStopping/chọn model vừa dùng chọn ngưỡng nên số liệu val hơi lạc quan; test chỉ dùng 1 lần cho model cuối.
- Grad-CAM chỉ mang tính giải thích định tính, không đảm bảo model luôn đúng.
