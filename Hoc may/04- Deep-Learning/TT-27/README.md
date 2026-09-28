# TT-27 - RNN/LSTM: Du bao luu luong giao thong theo gio

## Huong dan chay
1. Cai thu vien: `pip install tensorflow xgboost scikit-learn pandas numpy matplotlib joblib`
2. Dat file `Metro_Interstate_Traffic_Volume.csv` cung thu muc voi notebook `notebooks/lstm_traffic.ipynb`
3. Restart Kernel roi Run All de chay dung thu tu tu dau den cuoi
4. Ket qua duoc luu vao models/ va reports/ khi chay den cell cuoi

## Cac buoc da thuc hien va nhan xet

1-2. Lam sach: khu trung 48204 -> 40575 dong (bo dong trung date_time),
   danh dau 10 dong temp=0 va 1 dong rain_1h phi ly, reindex phat hien
   11976 gio thieu. Noi suy chi cho lo hong <= 3 gio, con lai 43750 gio chia
   thanh 139 doan lien tuc de cua so truot khong bac qua lo hong.
3-4. EDA xac nhan 2 dinh gio cao diem va chu ky tuan (reports/eda_theo_gio.png).
5. Chia theo thoi gian 70/15/15. Hai scaler (dac trung va luu luong) fit chi tren train.
   Chuoi dau vao GOM luu luong qua khu (scaled), target scaled khi train va dua ve xe/gio khi danh gia.
6. Tat ca baseline va model duoc danh gia tren CUNG 6464 moc thoi gian test.
   Baseline tinh tren luoi gio day du nen khong lech khi co lo hong.
   - Du bao bang trung binh train: MAE=1730.7
   - Naive (lag 24h): MAE=559.5
   - Seasonal naive (lag 168h): MAE=337.9
7. XGBoost + lag features: MAE=145.1, train 2.6s.
8. So sanh kien truc (reports/rnn_lstm_gru.png, reports/learning_curves.png):
Kien_truc        MAE  Thoi_gian_train(s)  So_tham_so  So_epoch
SimpleRNN 176.032867           95.327210        8641        26
     LSTM 165.227509          291.341619       34465        25
      GRU 163.194778          227.681482       26145        30
9. Khao sat do dai cua so, MAE (xe/gio): 6h: 161.8, 12h: 166.6, 24h: 165.9, 48h: 172.5
10. Du bao vs thuc te 1 tuan cuoi (reports/du_bao_vs_thuc_te.png).
11. Du bao nhieu buoc, MAE LSTM: 1h: 167.5, 3h: 254.7, 6h: 240.5;
    seasonal naive cung tap: 1h: 338.4, 3h: 338.5, 6h: 338.7
12. Gio sai nhieu nhat: 16h, thu sai nhieu nhat: 5 (0=Thu 2).

## Ket luan
LSTM (MAE 165.2) THANG seasonal naive (MAE 337.9), nen deep learning co gia tri so voi baseline theo mua vu. XGBoost + lag features (MAE 145.1, train 2.6s) bang hoac tot hon LSTM va nhanh hon nhieu lan, nen day la lua chon thuc te hon.

## Han che
- Khong dung Bidirectional LSTM vi du bao tuong lai khong the nhin du lieu sau thoi diem hien tai.
- Du lieu chi tu 1 tram do (I-94 westbound), khong dai dien moi tuyen duong.
- Lo hong dai (vai thang) bi loai bo, co the mat thong tin xu huong dai han.
- Thoi tiet dung la thoi tiet thuc te o cua so qua khu, chua dung thoi tiet du bao.
