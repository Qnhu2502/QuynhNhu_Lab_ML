# TT-27 - RNN/LSTM: Du bao luu luong giao thong theo gio

## Huong dan chay
1. Cai thu vien: `pip install tensorflow xgboost scikit-learn pandas numpy matplotlib`
2. Dat file `Metro_Interstate_Traffic_Volume.csv` cung thu muc voi notebook `notebooks/lstm_traffic.ipynb`
3. Restart Kernel roi Run All de chay dung thu tu tu dau den cuoi
4. Ket qua duoc luu vao models/ va reports/ khi chay den cell cuoi

## Cac buoc da thuc hien va nhan xet

1-2. Da xu ly du 4 van de: khu trung -4347 dong, danh dau 10 dong
   temp=0 va 1 dong rain_1h phi ly, reindex phat hien 11976 gio thieu
   hoan toan, con lai 43750 gio sau xu ly voi 139 doan lien tuc.
3-4. EDA xac nhan 2 dinh gio cao diem va chu ky tuan ro ret (xem reports/eda_theo_gio.png).
5. Chia theo thoi gian 70/15/15, scaler fit chi tren train.
6. Baseline naive (lag 24h): MAE=568.0. Seasonal naive (lag 168h): MAE=370.6.
7. XGBoost + lag features: MAE=147.6.
8. So sanh kien truc (xem reports/rnn_lstm_gru.png):
Kien_truc         MAE  Thoi_gian_train(s)  So_tham_so
SimpleRNN 3024.344443           73.443355        8513
     LSTM 3037.730937          138.115064       33953
      GRU 3024.540459          180.659398       25761
9. Khao sat do dai cua so (xem reports/do_dai_cua_so.png): {6: 3079.774762106105, 12: 3083.515657882085, 24: 3096.3121215335163, 48: 3109.5241714184626}
10. Du bao vs thuc te 1 tuan cuoi (xem reports/du_bao_vs_thuc_te.png).
11. Du bao nhieu buoc: {1: 3103.5927538160454, 3: 3097.7297538186335, 6: 3098.1857910772833}
12. Gio sai nhieu nhat: 16,
    thu sai nhieu nhat: 4.

## Ket luan
LSTM KHONG vuot troi ro ret so voi seasonal naive/XGBoost - day la ket luan trung thuc: voi du lieu chu ky manh nhu giao thong, mo hinh don gian hon co the du dung, khong bat buoc dung deep learning.

## Han che
- Khong dung Bidirectional LSTM vi du bao tuong lai khong the nhin du lieu sau thoi diem hien tai.
- Du lieu chi tu 1 tram do (I-94 westbound), khong dai dien moi tuyen duong.
- Lo hong thoi gian dai (vai thang) van bi loai bo hoan toan, co the mat thong tin xu huong dai han.
