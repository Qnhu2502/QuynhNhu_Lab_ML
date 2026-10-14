# TT-34: Embeddings + Cosine Similarity, khớp CV với JD theo ngữ nghĩa

Hệ thống này chỉ SẮP THỨ TỰ ĐỌC CV để nhà tuyển dụng đọc các CV phù hợp nhất trước.
Nó KHÔNG loại ứng viên tự động. Mọi CV vẫn phải có người xem.

## Cấu trúc và cách chạy

```
src/chunking.py    chia CV theo mục, cắt đoạn theo giới hạn token
src/matcher.py     TF-IDF, embedding nguyên CV, embedding + chunking
src/evaluate.py    NDCG@10, Precision@10, cặp đồng nghĩa, kiểm tra thiên lệch
notebooks/embeddings_cv_matching.ipynb   chạy từng bước, sinh toàn bộ file trong reports/
data/cv_an_danh/   <nganh>__<id>.txt  (vd IT__01.txt), đã ẩn danh hoàn toàn
data/jd/           <id>.txt
data/nhan_vang.csv sinh mẫu ở lần chạy đầu, tự chấm cột nhan (0-3)
```

```bash
pip install -r requirements.txt
jupyter notebook notebooks/embeddings_cv_matching.ipynb
```

Lần chạy đầu notebook dừng ở Bước 2 để bạn chấm nhãn vàng; chấm xong chạy lại từ đầu.

## Các bước thực hiện và nhận xét

Phần "Nhận xét" điền sau khi chạy, dựa trên số liệu trong `reports/`. Dòng "Gợi ý" cho biết cần nhìn vào đâu.

### Bước 1. Chuẩn bị 30 CV + 5 JD ẩn danh
Đã xoá tên, ảnh, số điện thoại, email, địa chỉ, ngày sinh, giới tính. Tên file chứa ngành để tô màu biểu đồ.
- Nhận xét: (số CV theo từng ngành, nguồn dữ liệu, cách ẩn danh)

### Bước 2. Nhãn vàng 150 cặp (30 x 5)
Chấm tay 0-3 từng cặp, độc lập với kết quả của model (chưa xem điểm trước khi chấm).
- Nhận xét: (phân bố nhãn 0/1/2/3; có cặp nào khó chấm, tiêu chí chấm)

### Bước 3. Kiểm tra độ dài CV theo token
File: `reports/do_dai_cv.csv`. Đối chiếu số token với `max_seq_length` của model (MiniLM đa ngôn ngữ chỉ 128 token).
- Gợi ý: tỉ lệ CV bị cắt, tỉ lệ nội dung bị bỏ trung bình.
- Nhận xét: (xác nhận vấn đề cắt cụt âm thầm)

### Bước 4. Baseline TF-IDF + cosine
`score_tfidf`, unigram + bigram, `min_df=2`.

### Bước 5. Embedding nguyên CV (không chia nhỏ)
`score_full`: chỉ phần đầu CV được model nhìn thấy.

### Bước 6. Embedding + chunking theo mục
`score_chunk`: chia CV theo mục (kinh nghiệm, kỹ năng, dự án, học vấn), mục dài cắt tiếp thành cửa sổ vừa giới hạn token.
Điểm mục = MAX điểm các đoạn; điểm CV = trung bình có trọng số (0,5 / 0,3 / 0,15 / 0,05, phần còn lại 0,1),
chia cho tổng trọng số các mục có mặt để CV thiếu mục không bị phạt oan.

### Bước 7. So sánh 3 cách bằng NDCG@10 và Precision@10
Files: `reports/so_sanh_3_cach.csv`, `reports/so_sanh_3_cach.png`. Mức tham chiếu: embedding + chunking ~0,75-0,85; TF-IDF ~0,55-0,65.

| Cách | NDCG@10 | P@10 |
|---|---|---|
| TF-IDF | | |
| Embedding (nguyên CV) | | |
| Embedding + chunking | | |

- Nhận xét: (embedding + chunking có thắng TF-IDF rõ rệt không; chunking cải thiện bao nhiêu so với nguyên CV; JD nào model làm kém và vì sao)

### Bước 8. So sánh 3 model embedding
File: `reports/so_sanh_model.csv` (NDCG@10, P@10, chiều vector, dung lượng, thời gian encode).
Lưu ý `keepitreal/vietnamese-sbert` nên tách từ bằng `pyvi` trước khi encode.
- Nhận xét: (model nào cân bằng tốt nhất giữa điểm, tốc độ, dung lượng)

### Bước 9. Mười cặp đồng nghĩa TF-IDF bỏ sót
File: `reports/cap_dong_nghia.md`. Cặp được giữ khi cosine embedding >= 0,6 và cosine TF-IDF <= 0,1.
- Nhận xét: (nhóm cặp nào embedding bắt tốt: viết tắt, Anh-Việt, tên framework; có cặp nào embedding bắt sai)

### Bước 10. Trực quan hoá PCA / t-SNE
File: `reports/tsne_cv.png`, tô màu theo ngành.
- Nhận xét: (các ngành có tách cụm không; CV nào nằm lẫn cụm khác và vì sao). Lưu ý t-SNE trên 30 điểm chỉ mang tính minh hoạ.

### Bước 11. Kiểm tra thiên lệch giới tính
Files: `reports/bias_check.csv` (theo từng CV), `reports/bias_check_tong_hop.csv` (tóm tắt).
Cùng một CV, chỉ đổi dòng "Họ tên" từ tên nam sang tên nữ; so điểm, thứ hạng và kiểm định Wilcoxon.
- Gợi ý: so `chenh_lech_tuyet_doi_tb` với độ lệch chuẩn của điểm (cột `so_voi_do_lech_chuan_cua_diem`), xem `doi_hang_tb` và `p_wilcoxon`.
- Kết luận: (có / không có thiên lệch đáng kể; nếu có, nêu hướng thiên lệch và cách giảm, ví dụ bỏ tên khỏi văn bản trước khi encode)

### Bước 12. Tốc độ
Trong `reports/so_sanh_model.csv`: cột `encode_1000_CV_giay(uoc_tinh)` và `tra_cuu_1_JD_ms` (1 JD so với kho vector dựng sẵn).
- Nhận xét: (encode một lần, tra cứu nhanh; với kho lớn nên dùng ChromaDB)

## Kết luận chung
- (Tóm tắt phương pháp tốt nhất và con số NDCG@10)
- Đây là hệ thống XẾP HẠNG hỗ trợ, không phải hệ thống LOẠI tự động. Mọi CV vẫn phải có người xem.
- Đã kiểm tra thiên lệch giới tính, kết quả: (điền)
- Hạn chế: 30 CV là mẫu nhỏ, nhãn vàng do một người chấm, kết quả có thể dao động.

## Mở rộng
1. Lưu vector vào ChromaDB để tra cứu trên 100.000 CV (nối sang TT-35).
2. Dùng Claude API sinh giải thích vì sao CV phù hợp (embedding xếp hạng, LLM diễn giải).
3. Tinh chỉnh model embedding trên dữ liệu tuyển dụng của công ty bằng contrastive learning.
