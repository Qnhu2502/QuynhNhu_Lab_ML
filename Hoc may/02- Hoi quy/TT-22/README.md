# TT-22 - MLP Regressor: Du doan muc tieu hao nhien lieu

## Huong dan chay
1. Cai thu vien: `pip install scikit-learn pandas numpy matplotlib joblib`
2. Dat file `auto-mpg.csv` cung thu muc voi notebook `notebooks/mlp_regressor_mpg.ipynb`
3. Restart Kernel roi Run All de chay dung thu tu tu dau den cuoi
4. Ket qua (model, hinh, requirements) duoc luu vao models/ va reports/ khi chay den cell cuoi

## Cac buoc da thuc hien va nhan xet

1. Xu ly 6 gia tri thieu o horsepower bang median.
2. One-hot origin, bo cot dinh danh name.
3. EDA xac nhan weight va mpg co quan he nghich, hoi cong; mpg tang dan theo nam san xuat.
4. Baseline: Linear RMSE=2.888 R2=0.845, Random Forest RMSE=2.171 R2=0.912.
5. MLP khong scale: RMSE=2.976 - rat kem do thang do dac trung chenh lech lon.
6. So sanh 3 truong hop scale (xem reports/scale_comparison.png):
   khong scale=2.976, chi scale X=2.389, scale ca X va y=2.143.
7. So sanh 4 kien truc (xem reports/kien_truc_overfit.png):
     Kien_truc  So_tham_so  RMSE_train  RMSE_test  Overfit
         (16,)         161    2.792489   2.402292 khong ro
         (64,)         641    2.329774   2.117654 khong ro
      (64, 32)        2689    2.388540   2.142923 khong ro
(256, 128, 64)       43521    1.577430   2.369165 khong ro
   Kien truc lon nhat (256,128,64) co so tham so vuot xa 398 mau train, RMSE train rat thap
   nhung RMSE test khong cai thien tuong ung - dau hieu overfit ro ret.
8. Loss curve xem reports/loss_curve.png - loss giam dan va on dinh nho early_stopping.
9. Khao sat alpha (xem reports/alpha_sweep.png) - alpha qua nho de overfit, qua lon de underfit.
10. So sanh activation relu/tanh/logistic - relu thuong on dinh va nhanh hoi tu nhat.
11. Bang so sanh cuoi voi 3 thuat toan khac:
            Model     RMSE       R2  Thoi_gian_train(s)               Giai_thich_duoc
Linear Regression 2.887673 0.844910            0.002424            Co (he so ro rang)
    Random Forest 2.171027 0.912336            0.699843 Mot phan (feature importance)
              SVR 2.560018 0.878108            0.072043               Khong (hop den)
              MLP 2.142923 0.914591            0.363744               Khong (hop den)

## Ket luan trung thuc
Voi chi 398 dong du lieu bang, MLP KHONG the hien loi the ro ret so voi Random Forest.
Mang no-ron can nhieu du lieu de phat huy suc manh xap xi ham lien tuc; voi tap nho nhu
Auto MPG, cac model cay/ensemble thuong dat do chinh xac ngang hoac tot hon, de tune hon,
va nhanh hon dang ke. MLP chi thuc su co loi the khi du lieu du lon hoac quan he giua
dac trung va nhan that su rat tron/phuc tap.

## Han che
- Bo du lieu nho (398 dong), khong dai dien cho cac dong xe hien dai (du lieu tu nam 1970-1982).
- MLP nhay cam voi kien truc va sieu tham so, can tune can than hon cac model cay.
- Uoc tinh tien xang/nam dua tren gia xang co dinh, thuc te bien dong theo thoi diem.
