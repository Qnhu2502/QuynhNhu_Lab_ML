# TT-23 — K-MEANS CLUSTERING
## Gom nhóm sản phẩm để sắp xếp lại kệ hàng siêu thị

| | |
|---|---|
| 🎓 **Khoá** | HỌC MÁY · [Buổi 5](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-05-Unsupervised-PCA) |
| 🧠 **Nhóm** | **Học KHÔNG giám sát** — Gom cụm |
| 🔧 **Thuật toán** | K-Means |
| 🏭 **Lĩnh vực** | Bán lẻ · Quản lý danh mục hàng hoá |
| ⏱ **Thời lượng** | 5–7 giờ |
| 📈 **Độ khó** | ⭐⭐ |

---

## 1. THUẬT TOÁN NÀY LÀ GÌ

```
   ① Chọn ngẫu nhiên K tâm cụm
   ② Gán mỗi điểm về tâm GẦN NHẤT
   ③ Dời tâm về TRUNG BÌNH các điểm vừa gán
   ④ Lặp lại ②③ tới khi tâm không đổi

        ●●● ✚          ▲▲▲ ✚         ✚ = tâm cụm
       ●●●●            ▲▲▲▲
                                     Mục tiêu: tối thiểu hoá tổng
        ■■■ ✚                        khoảng cách bình phương tới tâm (inertia)
       ■■■■
```

⚠️ **Ba giả định ngầm của K-Means** (phải nêu trong báo cáo):
```
   ① Cụm có dạng HÌNH CẦU và kích thước tương đương
   ② Mọi đặc trưng quan trọng NGANG NHAU (→ bắt buộc chuẩn hoá)
   ③ Số cụm K phải biết TRƯỚC
   → Nếu cụm thật có dạng dài/cong → K-Means chia sai. Khi đó dùng DBSCAN.
```

---

## 2. BÀI TOÁN THỰC TẾ

```
   Siêu thị có 4.000 mã hàng. Muốn sắp xếp lại kệ theo HÀNH VI MUA,
   không phải theo danh mục nhà cung cấp.

   Ví dụ phát hiện có giá trị: nhóm hàng "mua đột biến cuối tuần, giá trị cao,
   ít lặp lại" → nên đặt ở khu vực dễ thấy vào thứ 6–7.

   → Sản phẩm bàn giao: file CSV gán mỗi mã hàng vào 1 nhóm + mô tả nhóm
     để bộ phận trưng bày thực hiện.
```

---

## 3. BỘ DỮ LIỆU

| | |
|---|---|
| **Tên** | Online Retail II (UCI) |
| **Link** | https://archive.ics.uci.edu/dataset/502/online+retail+ii |
| **Kích thước** | ~1.067.000 dòng giao dịch |
| **Đơn vị phân tích** | **SẢN PHẨM** (`StockCode`), không phải khách hàng |

> ⚠️ Khác với dự án [Phân khúc khách hàng RFM](../../../00-Du-an-Tong-hop/Bai-03-Phan-khuc-Khach-hang-RFM/)
> vốn gom cụm **khách hàng**. Bài này gom cụm **sản phẩm** từ cùng dữ liệu — góc nhìn khác.

### Đặc trưng cấp sản phẩm cần tự tạo

```python
sp = df.groupby('StockCode').agg(
    tong_so_luong   = ('Quantity',  'sum'),
    tong_doanh_thu  = ('ThanhTien', 'sum'),
    gia_trung_binh  = ('Price',     'mean'),
    so_don_hang     = ('Invoice',   'nunique'),
    so_khach_mua    = ('Customer ID','nunique'),
    do_lech_sl      = ('Quantity',  'std'),
).reset_index()

sp['sl_moi_don']     = sp['tong_so_luong'] / sp['so_don_hang']
sp['ty_le_mua_lai']  = sp['so_don_hang'] / sp['so_khach_mua']
```

### ⚠️ Làm sạch bắt buộc (giống dự án RFM)

```
   ☐ Loại hoá đơn huỷ (Invoice bắt đầu bằng 'C')
   ☐ Loại Quantity <= 0 và Price <= 0
   ☐ Loại StockCode không phải sản phẩm: POST, M, BANK CHARGES, DOT, ADJUST
   ☐ Loại sản phẩm bán quá ít (< 5 đơn) → không đủ dữ liệu để gom cụm
```

---

## 4. HƯỚNG ĐI ĐÚNG

### 4.1. Xử lý phân phối lệch — bước quyết định

```
   Doanh thu và số lượng LỆCH PHẢI CỰC NẶNG (vài mã bán khổng lồ).
   K-Means dùng khoảng cách Euclid → vài "sản phẩm cá voi" kéo lệch toàn bộ tâm cụm.

   → BẮT BUỘC: np.log1p() trước, RỒI MỚI StandardScaler.
```

```python
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.cluster import KMeans

pipe = Pipeline([
    ('log',   FunctionTransformer(np.log1p)),
    ('scale', StandardScaler()),
    ('km',    KMeans(n_clusters=5, n_init=10, random_state=42)),
])
```

### 4.2. Chọn K bằng 3 căn cứ

```
   ① ELBOW      — inertia theo K, tìm "khuỷu tay" (thường mơ hồ)
   ② SILHOUETTE — > 0,5 tốt · 0,25–0,5 chấp nhận · < 0,25 cụm chồng lấn
   ③ KINH DOANH — K = 4–6 (siêu thị không quản lý nổi 15 nhóm trưng bày)
```

---

## 5. CÁC BƯỚC THỰC HIỆN

```
   ☐ 1. Làm sạch giao dịch (4 bước ở mục 3), ghi số dòng loại mỗi bước
   ☐ 2. Tổng hợp lên cấp sản phẩm, tạo 8 đặc trưng
   ☐ 3. Vẽ histogram TRƯỚC và SAU log1p → chứng minh cần biến đổi
   ☐ 4. Chạy K từ 2..12, ghi inertia + silhouette
   ☐ 5. Vẽ Elbow + Silhouette cạnh nhau, CHỌN K có lý do (cả kỹ thuật lẫn kinh doanh)
   ☐ 6. ⚠️ Kiểm tra `n_init`: chạy K-Means với n_init=1 và seed khác nhau
        → kết quả có ĐỔI không? → giải thích vì sao cần n_init=10
   ☐ 7. PCA 2 chiều để vẽ scatter tô màu theo cụm
   ☐ 8. Bảng mô tả cụm: số mã hàng · % doanh thu · giá TB · tần suất mua
   ☐ 9. ⭐ ĐẶT TÊN từng cụm bằng tiếng Việt + đề xuất vị trí trưng bày
   ☐ 10. So sánh với DBSCAN → DBSCAN tìm ra bao nhiêu nhiễu (sản phẩm cá biệt)?
   ☐ 11. Xuất file bàn giao: StockCode | Description | cụm | tên cụm
```

---

## 6. TIÊU CHÍ HOÀN THÀNH

```
   ☐ Có histogram trước/sau log1p
   ☐ Có Elbow + Silhouette, K chọn có căn cứ kép (kỹ thuật + kinh doanh)
   ☐ Silhouette của K đã chọn > 0,25
   ☐ Có thí nghiệm n_init chứng minh K-Means phụ thuộc khởi tạo
   ☐ Có scatter PCA tô màu cụm
   ☐ ⭐ Mỗi cụm có TÊN tiếng Việt + mô tả + đề xuất trưng bày cụ thể
   ☐ Có file CSV bàn giao
   ☐ Nêu được 3 giả định của K-Means và bài này có thoả mãn không
```

---

## 7. CẠM BẪY

| Cạm bẫy | Hậu quả |
|---------|---------|
| Không log-transform | Vài mã "cá voi" kéo lệch toàn bộ tâm cụm |
| Không chuẩn hoá | Doanh thu (đơn vị nghìn) áp đảo số đơn (đơn vị chục) |
| `n_init=1` | Kết quả đổi mỗi lần chạy |
| Chọn K = 10+ | Bộ phận trưng bày không triển khai nổi |
| Dừng ở nhãn "Cluster 0,1,2" | Kết quả vô dụng với người dùng |
| Bỏ qua sản phẩm bán ít | Hoặc ngược lại, giữ lại làm nhiễu cụm |

---

## 8. SẢN PHẨM NỘP & MỞ RỘNG

```
TT-23-KMeans-<HoTen>/
├── README.md                     ← có bảng mô tả cụm + đề xuất trưng bày
├── notebooks/kmeans_san_pham.ipynb
├── src/{features.py, cluster.py}
├── outputs/san_pham_theo_cum.csv ← ⭐ bàn giao cho bộ phận trưng bày
├── reports/{truoc_sau_log.png, elbow_silhouette.png, pca_scatter.png, mo_ta_cum.png}
└── requirements.txt
```

**Mở rộng:**
1. **MiniBatchKMeans** cho dữ liệu lớn → nhanh hơn bao nhiêu, kém chính xác bao nhiêu?
2. So sánh với **Hierarchical Clustering** — vẽ dendrogram, có thấy cấu trúc phân tầng không?
3. Kết hợp với **Market Basket Analysis** (thuật toán Apriori): sản phẩm cùng cụm
   có hay được mua chung không? → kiểm chứng chéo kết quả gom cụm

**Tham khảo:** [Buổi 5 — Unsupervised & PCA](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-05-Unsupervised-PCA/Tai-Lieu)
