# TT-19 - XGBoost Regressor: Du bao nhu cau thue xe dap theo gio

## Huong dan chay
1. Cai thu vien: `pip install xgboost scikit-learn pandas numpy matplotlib shap --break-system-packages`
2. Dat file `hour.csv` cung thu muc voi notebook `notebooks/xgboost_bike_demand.ipynb`
3. Mo notebook, chon Restart Kernel roi Run All de chay dung thu tu tu dau den cuoi
4. Ket qua (model, hinh, requirements) se duoc luu vao `models/` va `reports/` khi chay den cell cuoi cung

## Chung minh ro ri
Giu casual + registered lam dac trung cho R2 = 1.0000,
he so hoi quy = [1. 1.], xac nhan cnt = casual + registered dung tuyet doi.
Da bo ca hai cot nay va cot instant truoc khi train model that, tranh ro ri du lieu.

## Nhan xet ket qua
- Baseline naive (cung gio tuan truoc): RMSE = 156.79. Day la muc san XGBoost phai vuot qua.
- XGBoost + early stopping: dung 1853 cay, RMSE tren tap test = 118.70
  (THANG baseline naive).
- So sanh log1p: khong log1p RMSE=118.70, co log1p RMSE=124.60.
  Log1p khong cai thien dang ke trong truong hop nay.
- Tham so toi uu tu RandomizedSearchCV (TimeSeriesSplit): {'subsample': 1.0, 'min_child_weight': 5, 'max_depth': 6, 'learning_rate': 0.05, 'colsample_bytree': 0.8}
- Model sai nhieu nhat o gio 8 va dieu kien thoi tiet 1
  (xem chi tiet o reports/phan_tich_loi.png).

## Han che
- Du lieu chi thu thap tai 1 thanh pho (Washington D.C.) trong 2 nam 2011-2012, khong dai dien khu vuc/thoi tiet khac.
- Baseline naive "cung gio tuan truoc" rat manh do nhu cau xe dap co tinh chu ky ro ret - luon phai so sanh voi baseline nay truoc khi ket luan model phuc tap co gia tri thuc su.
- Model chua tinh den cac su kien dac biet (le hoi, thi dau the thao) co the lam nhu cau tang dot bien ngoai quy luat thong thuong.
