# TT-29 — TF-IDF: Phân loại ticket hỗ trợ khách hàng

## 1. Bảng chứng minh rò rỉ (Bước 1)

| Phiên bản dữ liệu | Accuracy |
|---|---|
| CÓ headers/footers/quotes (rò rỉ) | 0.853 |
| ĐÃ remove (dùng chính thức) | 0.692 |

Chênh lệch: +0.161. Toàn bộ các bước sau chỉ dùng bản ĐÃ remove.

## 2. Kết quả mô hình chính (Bước 4)

TF-IDF (ngram (1,2), min_df=3, max_df=0.8, sublinear_tf=True) + LinearSVC
(calibrated): **accuracy = 0.689**, **macro-F1 =
0.676** (mức tham chiếu README: ~0.69-0.73).

## 3. Khảo sát tham số vectorizer (Bước 5, top 5)

| ngram_range | min_df | sublinear_tf | so_dac_trung | accuracy_cv |
| --- | --- | --- | --- | --- |
| (1, 1) | 1 | True | 50000 | 0.7541983345017637 |
| (1, 1) | 1 | False | 50000 | 0.7513699177149498 |
| (1, 1) | 3 | True | 27052 | 0.7493367880999898 |
| (1, 1) | 3 | False | 27052 | 0.7471272222320646 |
| (1, 1) | 5 | True | 18100 | 0.7440333191509425 |

Tham số tốt nhất theo CV: `{'ngram_range': (1, 1), 'min_df': 1, 'sublinear_tf': True}`

## 4. So sánh 3 bộ phân loại (Bước 6)

| Mô hình | Accuracy | Macro-F1 |
| --- | --- | --- |
| Naive Bayes | 0.6237387148167818 | 0.5978361147478956 |
| Logistic Regression | 0.6836165693043016 | 0.6679872215846842 |
| LinearSVC | 0.6959638874137015 | 0.6847747897431566 |

## 5. Top từ đặc trưng mỗi lớp (Bước 7)

- **alt.atheism**: atheism, motto, atheists, bobby, religion, islam, islamic, nanci, deletion, cruel
- **comp.graphics**: graphics, 3d, image, images, 68070, animation, pov, tiff, cview, 3do
- **comp.os.ms-windows.misc**: windows, cica, win3, risc, mfc, ini, file, w4wg, smartdrv, nt
- **comp.sys.ibm.pc.hardware**: vlb, ide, bios, gateway, cmos, f550iw, os, t560i, fastmicro, controller
- **comp.sys.mac.hardware**: mac, apple, powerbook, quadra, se, lc, centris, adb, duo, nubus
- **comp.windows.x**: motif, server, widget, x11r5, xterm, window, mit, widgets, x11, clients
- **misc.forsale**: sale, shipping, sell, offer, 2190, interested, condition, buyer, forsale, asking
- **rec.autos**: car, cars, dealer, oil, ford, vw, toyota, gt, autos, sho
- **rec.motorcycles**: bike, dod, bikes, ride, motorcycle, helmet, riding, harley, bmw, motorcycles
- **rec.sport.baseball**: baseball, stadium, phillies, braves, cubs, alomar, ball, mets, royals, sox
- **rec.sport.hockey**: hockey, nhl, puck, playoff, mask, leafs, playoffs, devils, coach, ice
- **sci.crypt**: clipper, encryption, nsa, key, security, pgp, privacy, keys, vesselin, crypto
- **sci.electronics**: circuit, electronics, 8051, dial, cci, motorola, power, dsp, dayton, scope
- **sci.med**: msg, doctor, medical, disease, treatment, photography, needles, diet, health, patients
- **sci.space**: space, orbit, launch, spacecraft, nasa, solar, shuttle, moon, lunar, mars
- **soc.religion.christian**: church, god, christianity, easter, scripture, marriage, resurrection, catholic, christians, abstinence
- **talk.politics.guns**: gun, guns, firearms, weapons, firearm, nra, weapon, jmd, fire, bd
- **talk.politics.mideast**: israel, israeli, arab, turkish, arabs, loser, jews, turkey, armenians, retarded
- **talk.politics.misc**: libertarians, tax, trial, libertarian, jobs, homosexuals, drugs, taxes, clinton, blacks
- **talk.religion.misc**: koresh, kent, rosicrucian, ekr, hudson, christian, amorc, 666, paradise, objective

## 6. Cơ chế ngưỡng tin cậy (Bước 9)

| Chỉ số | Giá trị |
| --- | --- |
| % ticket tự động xử lý | 49.8% |
| Accuracy của phần tự động | 0.916 |
| % ticket cần người xử lý | 50.2% |
| Accuracy nếu xử lý 100% tự động (không ngưỡng) | 0.689 |

## 7. Thời gian xử lý (Bước 10)

- Train toàn bộ: 31.6 giây
- Dự đoán 1 ticket: 27.41 ms
  (CHƯA ĐẠT yêu cầu < 5 ms)

## 8. Hạn chế

TF-IDF chỉ đếm từ, không hiểu nghĩa ("tuyệt vời" ≠ "hay" với TF-IDF) — xem
Bước 8 (ma trận nhầm lẫn) để biết các cặp chủ đề gần nghĩa hay bị nhầm.

## Cấu trúc thư mục

- `notebooks/` — notebook đầy đủ 11 bước
- `src/preprocess.py` — tải & làm sạch dữ liệu
- `src/train.py` — script train đầy đủ (xuất từ notebook)
- `models/tfidf_pipeline.joblib` — pipeline đã train
- `reports/` — leakage_comparison.png, top_tu_moi_lop.png, confusion_matrix.png, nguong_tin_cay.png
