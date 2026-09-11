# TT-20 - SVR: Du doan cuong do chiu nen be tong

## Huong dan chay
1. Cai thu vien: `pip install scikit-learn pandas numpy matplotlib seaborn xgboost xlrd joblib`
2. Dat file `Concrete_Data.xls` cung thu muc voi notebook `notebooks/svr_concrete.ipynb`
3. Restart Kernel roi Run All de chay dung thu tu tu dau den cuoi
4. Ket qua (model, hinh, requirements) duoc luu vao `models/` va `reports/` khi chay den cell cuoi

## Dac trung tu kien thuc mien
Da tao 3 dac trung: water_cement_ratio (ty le nuoc/xi mang), tong_chat_ket_dinh
(cement + slag + fly_ash), log_age (Age lech phai manh nen dung log1p).
RMSE KHONG co dac trung mien: 6.041 MPa
RMSE CO dac trung mien: 5.351 MPa
=> Cai thien 11.4%, xac nhan kien thuc mien giup ich ro ret.

## Nhan xet ket qua

- SVR KHONG scale: RMSE=8.361, R2=0.729 — rat kem vi epsilon=0.1
  qua nho so voi thang do y (2-83 MPa), hau het diem nam ngoai ong.
- SVR CO scale ca X va y: RMSE=5.332, R2=0.890 — cai thien ro ret,
  xac nhan SVR bat buoc phai TransformedTargetRegressor de scale nhan.
- Kernel tot nhat: rbf (RMSE=5.332).
- Tham so toi uu tu GridSearchCV: {'regressor__svr__C': 1000, 'regressor__svr__epsilon': 0.1, 'regressor__svr__gamma': 0.01}
- So support vectors: 508/824 mau train (61.7%).
- SVR (toi uu) RMSE tren test: 5.351, R2=0.889.
- Thoi gian train tang tu 0.85s (1x) len 128.06s (10x du lieu)
  — tang nhanh hon tuyen tinh, xac nhan SVR khong phu hop du lieu lon (>50.000 dong).

## Han che
- SVR khong giai thich duoc tung du doan cu the (khong co he so ro rang nhu Linear Regression).
- SVR khong mo rong duoc cho du lieu lon do do phuc tap O(n^2)-O(n^3).
- Model du bao gia tri trung binh, chua uu tien tranh du bao cao hon thuc te — nganh xay dung
  can xem xet du bao phan vi thap (vi du GradientBoostingRegressor loss='quantile', alpha=0.1)
  de an toan ket cau, vi du bao cuong do cao hon thuc te co the gay nguy hiem khi thi cong.
