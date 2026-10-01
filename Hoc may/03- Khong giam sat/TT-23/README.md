# TT-23 - K-Means Clustering: Gom nhom san pham

## Huong dan chay
1. Cai thu vien: `pip install scikit-learn pandas numpy matplotlib`
2. Dat file `online_retail.csv` CUNG thu muc voi notebook `notebooks/kmeans_san_pham.ipynb`
3. QUAN TRONG: mo notebook truc tiep (khong mo ca thu muc cha chua nhieu bai TT khac) roi
   Restart Kernel + Run All, de outputs/ va reports/ duoc tao dung ngay canh notebook
4. Ket qua duoc luu vao outputs/san_pham_theo_cum.csv va reports/*.png khi chay den cell cuoi

## Cac buoc da thuc hien va nhan xet

1. Lam sach: ban dau 541909 dong, sau 4 buoc loai bo con 527938 dong.
2. Tong hop len 3470 ma san pham (sau khi loai san pham ban duoi 5 don), 8 dac trung.
3. Histogram xac nhan doanh thu/so luong lech phai cuc nang, bat buoc log1p truoc StandardScaler.
4-5. Bang silhouette day du cho K=2..12 (xem reports/elbow_silhouette.png). Trong khoang
     kinh doanh chap nhan duoc K=4..6, chon K=5 vi co silhouette cao nhat
     (0.253) trong nhom nay.
6. n_init=1 qua 5 seed cho inertia dao dong 683.9,
   trong khi n_init=10 on dinh hon han.
7. PCA 2 chieu giai thich 78.7% phuong sai.
8-9. Bang mo ta va TEN CUM duoc tinh tu XEP HANG TUONG DOI (khong hardcode):
 cum                                                       ten_cum                                                             de_xuat_trung_bay
   0              Gia cao, tan suat mua cao, chiem 21.0% doanh thu                       Vi tri trung tam, de thay nhat - dong gop doanh thu lon
   1              Gia cao, tan suat mua thap, chiem 1.1% doanh thu Khu trung bay rieng, anh sang tot - hang gia tri cao can tao cam giac cao cap
   2       Gia thap, tan suat mua trung binh, chiem 6.1% doanh thu                                             Ke thong thuong, theo doi dinh ky
   3 Gia trung binh, tan suat mua trung binh, chiem 0.6% doanh thu                                             Ke thong thuong, theo doi dinh ky
   4       Gia trung binh, tan suat mua cao, chiem 71.1% doanh thu                       Vi tri trung tam, de thay nhat - dong gop doanh thu lon
10. DBSCAN tim ra 4 cum va 344 san pham nhieu (9.9%).
11. Da xuat file ban giao outputs/san_pham_theo_cum.csv.

## Ba gia dinh K-Means va muc do thoa man
1. Cum hinh cau, kich thuoc tuong duong - so_ma_hang moi cum tu 276
   den 946, chi thoa man MOT PHAN.
2. Dac trung quan trong ngang nhau - da chuan hoa bang StandardScaler nen thoa man ve ky thuat.
3. So cum K biet truoc - da chon K=5 co can cu ky thuat va kinh doanh ro rang.
