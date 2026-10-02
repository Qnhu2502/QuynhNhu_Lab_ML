# TT-16 - Decision Tree Regressor: Dinh gia cuoc chuyen xe

## Huong dan chay
1. Cai thu vien: `pip install -r requirements.txt`
2. Tai 1 thang du lieu tu https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page,
   dat file `yellow_tripdata_2024-01.parquet` CUNG thu muc voi notebook nay (duong dan
   tuong doi, chay duoc tren moi may)
3. Restart Kernel roi Run All
4. Ket qua duoc luu vao outputs/, reports/, models/ khi chay den cell cuoi

## Cac buoc da thuc hien va nhan xet

1. Lay mau 200.000 dong tu file goc.
2. Lam sach, loai bo chi tiet tung dieu kien:
   - fare_amount <= 0: 2,679 dong
   - trip_distance <= 0 hoac > 100 miles: 3,981 dong
   - passenger_count == 0: 11,665 dong
   Con lai 183,633 dong sau lam sach.
3. Loai 3 cot ro ri (tip_amount, tolls_amount, total_amount) khoi dac trung, co
   assert tu dong kiem tra khong lot vao FEATURES.
4. Tao dac trung thoi gian: gio_don, thu_trong_tuan, gio_cao_diem.
5. Baseline: DummyRegressor MAE=11.15. Cong thuc tuyen tinh uoc luong tu
   chinh du lieu NYC (fare = 6.3 + 3.7 x trip_distance) MAE=2.65.
6. Cay khong gioi han do sau: 117,701 la, MAE train=0.009
   vs MAE test=2.725 - chenh lech rat lon, xac nhan OVERFIT ro ret.
7. Chon max_depth bang 5-Fold CV tren TRAIN (khong nhin test): CV goi y depth=14
   (MAE CV=2.325), nhung van dung max_depth=5 cho model chinh thuc vi
   uu tien kha nang tra bang tay cho tong dai (yeu cau nghiep vu).
8-9. Cay max_depth=5: 26 la (xem reports/ham_bac_thang.png,
   reports/cay_quyet_dinh.png).
10. Da xuat 26 luat ra outputs/bang_tra_cuoc.csv.
11. Ti le chuyen dat sai so +-15%: 59.6%
    -> CHUA DAT yeu cau nghiep vu (can >=80%).
12. So sanh model (xem reports/so_sanh_mo_hinh.png):
                                   model       MAE      MAPE      RMSE
         Baseline - DummyRegressor(mean) 11.152338 89.842537 17.355618
Baseline - Cong thuc tuyen tinh thu cong  2.651025 24.070403  5.368395
             Decision Tree (max_depth=5)  2.440616 22.352413  5.626314
                 Random Forest Regressor  2.018496 18.470262  4.637883
                       Linear Regression  2.636228 23.684935  5.361881

## Vi sao cay khong ngoai suy duoc
Xem muc rieng trong notebook - cay chi tra ve trung binh cua la, khong co cong thuc
dai so de uoc luong ngoai pham vi du lieu da thay. Minh hoa bang chuyen 200 miles.

## Han che
- Baseline cong thuc tuyen tinh 1 bien khong du de nam bat cuoc taxi NYC (con phu
  thuoc gio cao diem, khu vuc don/tra) - day la ly do can model phuc tap hon.
- Cay max_depth=5 danh doi do chinh xac de lay kha nang tra bang tay - MAE co the
  giam them neu dung do sau CV goi y, nhung se mat tinh de tra cuu.
- Du lieu chi 1 thang (01/2024), chua tinh yeu to mua vu/le tet ca nam.
