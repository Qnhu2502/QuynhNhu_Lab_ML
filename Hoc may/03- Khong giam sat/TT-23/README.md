# TT-23 - K-Means Clustering: Gom nhom san pham

## Huong dan chay
1. Cai thu vien: `pip install scikit-learn pandas numpy matplotlib`
2. Dat file `online_retail.csv` cung thu muc voi notebook `notebooks/kmeans_san_pham.ipynb`
3. Restart Kernel roi Run All de chay dung thu tu tu dau den cuoi
4. Ket qua duoc luu vao outputs/san_pham_theo_cum.csv va reports/*.png khi chay den cell cuoi

## Cac buoc da thuc hien va nhan xet

1. Lam sach: ban dau 541909 dong, sau 4 buoc loai bo con 527938 dong.
2. Tong hop len 3470 ma san pham (sau khi loai san pham ban duoi 5 don), 8 dac trung.
3. Histogram xac nhan doanh thu/so luong lech phai cuc nang, bat buoc log1p truoc StandardScaler.
4-5. Chon K=5 dua tren silhouette=0.253
     va can cu kinh doanh (sieu thi quan ly noi toi da 5-6 nhom trung bay).
6. n_init=1 qua 5 seed cho inertia dao dong 683.9,
   trong khi n_init=10 on dinh hon han - xac nhan K-Means phu thuoc khoi tao ngau nhien.
7. PCA 2 chieu giai thich 78.7% phuong sai.
8-9. Bang mo ta va dat ten cum (xem reports/mo_ta_cum.png).
10. DBSCAN tim ra 4 cum va 344 san pham nhieu (9.9%).
11. Da xuat file ban giao outputs/san_pham_theo_cum.csv.

## Ba gia dinh K-Means va muc do thoa man
1. Cum hinh cau, kich thuoc tuong duong - can kiem tra bang mo ta cum co cum nao qua lech khong.
2. Dac trung quan trong ngang nhau - da chuan hoa bang StandardScaler nen thoa man ve ky thuat.
3. So cum K biet truoc - da chon K=5 co can cu ky thuat va kinh doanh ro rang.
