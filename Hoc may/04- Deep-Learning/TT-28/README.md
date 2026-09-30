# TT-28 Transformer: tự động phân loại chủ đề tin tức cho toà soạn

Dữ liệu AG News (120.000 train, 7.600 test, 4 chuyên mục cân bằng). So sánh 4 cách: TF-IDF + LinearSVC, BiLSTM, Transformer tự xây, fine-tune DistilBERT.

## 1. Bài toán nghiệp vụ

Toà soạn nhận 5.000 tin/ngày, một biên tập viên xử lý khoảng 400 tin/ngày nên cần 12 người. Mục tiêu: mô hình tự phân loại các tin chắc chắn, chỉ chuyển biên tập viên những tin có độ tin cậy thấp. Câu hỏi cuối cùng của dự án: tăng vài % accuracy so với baseline có đáng với chi phí GPU không.

## 2. Cấu trúc và cách chạy

```
TT-28-Transformer/
├── README.md
├── requirements.txt
├── notebooks/  01_baseline_tfidf, 02_transformer_tu_xay, 03_finetune_bert, 04_tong_hop_so_sanh
├── src/        config, data, evaluate, transformer_block, train, report
├── models/     trọng số (không commit, sinh lại bằng notebook)
└── reports/    so_sanh_4_cach.png, attention_heatmap.png, confusion_4x4.png,
                khao_sat_head.png, ket_qua.md, results/ (json + npz từng mô hình)
```

```
pip install -r requirements.txt
# chạy lần lượt 01 -> 02 -> 03 -> 04 (02 và 03 nên chạy trên GPU)
```

Notebook 04 tự sinh 3 hình bắt buộc và `reports/ket_qua.md`. Dán các bảng trong file này vào mục 4 bên dưới.

## 3. Các quyết định phương pháp

| Vấn đề | Cách xử lý |
|---|---|
| Chia dữ liệu | Tách 10% val từ train (stratify, `random_state=42`), một lần trong `src/data.py`, mọi mô hình dùng chung. Test chỉ dùng đánh giá cuối. |
| Chống leakage | TF-IDF fit trong pipeline chỉ trên train. Từ điển Keras chỉ `adapt` trên train. C của SVC, early stopping, checkpoint tốt nhất đều chọn theo val. Có kiểm tra văn bản trùng giữa các tập. |
| Metric | Accuracy là chính vì lớp cân bằng hoàn toàn, thêm macro-F1 để không bỏ sót lớp yếu, thêm ma trận nhầm lẫn. |
| So sánh công bằng | Cùng split, cùng seed, cùng cách đo thời gian dự đoán 1 tin (trung bình 100 tin, đã khởi động). |
| Padding | Có mask: attention không chú ý vào padding, average pooling chỉ trên token thật. |
| Tái lập | `set_seed`, đường dẫn tính từ vị trí file, `requirements.txt`, lưu json kết quả từng mô hình. |

## 4. Các bước thực hiện và nhận xét

Các số liệu và nhận xét bên dưới điền sau khi chạy xong notebook. Mức tham chiếu của đề (không phải kết quả của bài này): TF-IDF+SVC khoảng 91%, Transformer tự xây 88 đến 90%, DistilBERT 94 đến 95%.

### Bước 1. Nạp dữ liệu, kiểm tra cân bằng lớp (notebook 01)
- Kết quả: TODO (số mẫu mỗi lớp, số văn bản trùng giữa các tập).
- Nhận xét: TODO. Lớp cân cân bằng nên accuracy đủ tin cậy. Nếu có văn bản trùng train/test thì ghi rõ ảnh hưởng.

### Bước 2. Baseline TF-IDF + LinearSVC (notebook 01)
- Kết quả: TODO (C được chọn, val acc, test acc, macro-F1, thời gian fit).
- Nhận xét: TODO. Đây là mốc để đánh giá mọi cách còn lại.

### Bước 3 và 4. Transformer tự xây (notebook 02)
- Kiến trúc: Embedding, positional encoding sin/cos, 2 khối Transformer (d_model 128, 4 head, d_ff 256), GlobalAveragePooling có mask, Dense softmax.
- Kết quả: TODO (test acc, macro-F1, số tham số).
- Nhận xét: TODO. So với TF-IDF và BiLSTM. Dự kiến nhánh này khó vượt TF-IDF với 120k mẫu, vì kiến trúc hiện đại cần nhiều dữ liệu hơn.

### Bước 5. Thí nghiệm bỏ positional encoding (notebook 02)
- Kết quả: TODO (acc có pos và không pos, chênh bao nhiêu điểm).
- Nhận xét: TODO. Không có vị trí thì mô hình coi câu là túi từ. Với phân loại chủ đề, từ khoá quan trọng hơn thứ tự nên mức tụt có thể nhỏ. Cần nêu đúng số đo được thay vì kỳ vọng.

### Bước 6. Khảo sát số head 1 / 2 / 4 / 8 (notebook 02, hình `khao_sat_head.png`)
- Kết quả: TODO (bảng acc và thời gian).
- Nhận xét: TODO. `key_dim = d_model / số head` nên số tham số không đổi, chênh lệch chỉ do cách chia không gian chú ý.

### Bước 7. Fine-tune DistilBERT (notebook 03)
- Cấu hình: lr 3e-5, 3 epoch, `max_length` 128, chọn checkpoint theo val accuracy.
- Kết quả: TODO.
- Nhận xét: TODO.

### Bước 8. Bảng so sánh 4 cách (notebook 04, hình `so_sanh_4_cach.png`)
TODO: dán bảng từ `reports/ket_qua.md`.
- Nhận xét: TODO. Thời gian train phụ thuộc phần cứng, ghi rõ GPU đã dùng.

### Bước 9. Ma trận nhầm lẫn (hình `confusion_4x4.png`)
- Cặp lớp hay nhầm: TODO (dự kiến Business và Sci/Tech).
- Nhận xét: TODO. Giải thích bằng ví dụ tin cụ thể, ví dụ tin về công ty công nghệ vừa thuộc kinh doanh vừa thuộc công nghệ.

### Bước 10. Heatmap attention (hình `attention_heatmap.png`)
- Nhận xét: TODO. Từ nào chú ý mạnh vào từ nào, có từ khoá chuyên mục không. Lưu ý: heatmap là khối đầu tiên, trung bình các head, nên chỉ minh hoạ chứ không chứng minh mô hình "hiểu".

### Bước 11. Human-in-the-loop, ngưỡng 95%
TODO: dán bảng human-in-the-loop từ `reports/ket_qua.md`.
- Nhận xét: TODO. Nêu % tin tự động, số biên tập viên còn cần (so với 12 người ban đầu). Lưu ý xác suất softmax của mạng sâu thường quá tự tin, nên nên kiểm tra thêm bằng độ chính xác thực tế của phần tự động (cột `Acc phan tu dong`).

## 5. Kết luận chi phí và lợi ích

TODO: chênh lệch accuracy giữa DistilBERT và TF-IDF là bao nhiêu điểm, quy ra số tin bị phân loại sai mỗi ngày (5.000 tin), đối chiếu với thời gian huấn luyện, chi phí GPU và độ trễ dự đoán. Kết luận nên dùng cách nào cho toà soạn và vì sao.

## 6. Cạm bẫy đã tránh

Quên positional encoding (có thí nghiệm đối chứng), lr fine-tune quá lớn (dùng 3e-5), thiếu baseline TF-IDF, `max_length` quá lớn (dùng 64 và 128), fine-tune quá nhiều epoch (3 epoch + chọn checkpoint theo val).
