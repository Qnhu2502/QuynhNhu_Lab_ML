# TT-24 - PCA: Nen 561 tin hieu cam bien

## Huong dan chay
1. Cai thu vien: `pip install scikit-learn pandas numpy matplotlib joblib`
2. Dat file `har_train.csv` va `har_test.csv` cung thu muc voi notebook `notebooks/pca_har_sensors.ipynb`
3. Restart Kernel roi Run All de chay dung thu tu tu dau den cuoi
4. Ket qua duoc luu vao models/ va reports/ khi chay den cell cuoi

## Cac buoc da thuc hien va nhan xet

1-2. Giu nguyen chia theo nguoi: 21 nguoi train, 9 nguoi test, khong trung nguoi nao. Chuan hoa fit chi tren train.
3-4. Scree plot va phuong sai tich luy (xem reports/scree_plot.png, reports/variance_tich_luy.png).
5. Bang nguong phuong sai:
 Nguong_phuong_sai  So_chieu_can  Ty_le_giam_chieu
              0.80            26          0.953654
              0.90            63          0.887701
              0.95           102          0.818182
              0.99           179          0.680927
6. Diem ngot: K=561, accuracy=0.9623 so voi day du 561 chieu accuracy=0.9623 (xem reports/danh_doi_chieu_accuracy.png).
7. Scatter PC1-PC2 (xem reports/pc1_pc2_scatter.png) - nhom hoat dong tinh va dong tach ro ngay voi 2 chieu.
8. PC1 dai dien chu yeu cho muc do chuyen dong/nang luong co the, dua tren top 10 dac trung dong gop.
9. Dung luong giam roi tu 31.47 MB xuong con vai MB tuong ung voi K da chon.
10. Sai so tai tao giam dan khi K tang, nhu ky vong.
11. So sanh PCA (accuracy=0.9620) voi SelectKBest (accuracy=0.9623) cung K=561.

## Han che
- PCA lam mat tinh giai thich, cac thanh phan chinh khong con y nghia vat ly truc tiep.
- PCA chi bat quan he tuyen tinh, co the bo sot cau truc phi tuyen trong du lieu chuyen dong.
