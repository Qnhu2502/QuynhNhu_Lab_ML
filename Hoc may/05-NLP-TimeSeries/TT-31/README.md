# TT-31 - Lag + Rolling Features: Du bao phu tai dien ngay mai

## Huong dan chay
1. Cai thu vien: `pip install -r requirements.txt`
2. Dat file `household_power_consumption.txt` cung thu muc voi notebook
3. Restart Kernel roi Run All
4. Ket qua duoc luu vao models/ va reports/ khi chay den cell cuoi

## Cac buoc da thuc hien va nhan xet

1-2. Gop du lieu theo phut ve theo NGAY (trung binh), reindex phat hien
   0 ngay thieu hoan toan, noi suy chi lo hong ngan (<=3 ngay).
3. EDA xac nhan phu tai cao hon vao mua dong va ngay thuong cao hon cuoi tuan
   (xem reports/eda_mua_vu.png).
4-5. Tao lag (1,2,3,7,14,30) va rolling (7,14,30) LUON co shift(1) truoc, xoa
   61 dong dau thieu dac trung (khong dien bang mean).
6. Chia theo thoi gian: train 2006-2008, val 2009, test 2010.
7. Baseline: Naive (hom qua) MAE=0.2220, Seasonal naive (tuan truoc) MAE=0.2591.
8. So sanh 3 thuat toan:
        Model      MAE  Thoi_gian_train(s)
        Ridge 0.192636            0.018616
Random Forest 0.192167            0.700147
      XGBoost 0.206461            2.273124
9. Thi nghiem ro ri (xem reports/thi_nghiem_ro_ri.png): MAE dung=0.2065 vs
   MAE ro ri (khong shift)=0.1764 - giam gia tao 14.5%,
   xac nhan day la loi ro ri nghiem trong nhat trong chuoi thoi gian.
10. Feature importance: dac trung quan trong nhat thuong la lag_1 va lag_7.
11. Thi nghiem ngoai suy (xem reports/ngoai_suy.png): tuyet doi MAE=0.2065
    vs chenh lech MAE=0.2060.
12. Du bao vs thuc te 60 ngay cuoi (xem reports/du_bao_60_ngay.png).
13. TimeSeriesSplit 5 fold: MAE trung binh=0.2296 +/- 0.0342.
14. Quy doi sai so: 11,530 dong/ngay (gia dien 2,500 d/kWh),
    uoc tinh 4,208,464 dong/nam cho 1 ho gia dinh.

## Han che
- Chua co bien thoi tiet (nhiet do anh huong manh phu tai dieu hoa).
- Du lieu 1 ho gia dinh don le, nhieu hon phu tai he thong tong hop nhieu ho.
- Du bao chenh lech giai quyet van de ngoai suy trend nhung can kiem tra dinh ky
  khi cau truc tieu thu thay doi (lap thiet bi moi...).
