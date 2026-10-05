# TT-30 - ARIMA/SARIMA: Du bao luong khach hang theo thang

## Huong dan chay
1. Cai thu vien: `pip install -r requirements.txt`
2. Can mang o lan chay dau de statsmodels tai du lieu AirPassengers
3. Restart Kernel roi Run All
4. Ket qua duoc luu vao models/ va reports/ khi chay den cell cuoi

## Cac buoc da thuc hien va nhan xet

1. Chuoi goc co trend tang va mua vu chu ky 12 thang, bien do phinh dan theo trend.
2. seasonal_decompose xac nhan mo hinh multiplicative hop hon additive (phan du phang hon).
3. Lay log(y) de bien thanh cong tinh, bien do mua vu da on dinh qua cac nam.
4. Bang ADF cho 4 phien ban chuoi:
                  phien_ban  adf_stat  p_value   ket_luan
                  Chuoi goc  0.815369 0.991880 KHONG DUNG
                     log(y) -1.717017 0.422367 KHONG DUNG
           log(y) + diff(1) -2.717131 0.071121 KHONG DUNG
log(y) + diff(1) + diff(12) -4.443325 0.000249       DUNG
   De xuat d=1, D=1.
5. ACF/PACF cua chuoi da dung (xem reports/acf_pacf.png) goi y p=1, q=1,
   P=1, Q=1.
6. Chia train 120 thang, test 24 thang.
7. Baseline: Naive MAPE=23.58%, Seasonal naive MAPE=15.52%.
8. SARIMA thu cong order=(1,1,1) seasonal_order=(1,1,1,12),
   AIC=-332.62.
9. Do tu dong bang grid search theo AIC: bac tot nhat order=(1, 1, 0),
   seasonal_order=(0, 1, 0), AIC=-359.98.
   Dung model tu dong cho cac buoc sau vi AIC thap hon.
10. Ljung-Box tai lag 12, 24 (xem reports/residual_diagnostics.png):
      lb_stat  lb_pvalue
12  29.522520   0.003293
24  29.769199   0.192519
    Phan du la nhieu trang: KHONG.
11. Du bao 24 thang + khoang tin cay 95% (xem reports/du_bao_khoang.png).
12. MAPE tren test: SARIMA=14.73%, so voi Naive=23.58%,
    Seasonal naive=15.52%.
    SARIMA THANG seasonal naive.
13. Ti le phu khoang tin cay 95%: 100.0% (ly tuong gan 95%).

## Han che
ARIMA/SARIMA chi dung lich su cua chinh chuoi, khong dua vao bien ngoai sinh.
Can bien ngoai (khuyen mai, le tet, thoi tiet) thi dung SARIMAX hoac ML + lag features.
