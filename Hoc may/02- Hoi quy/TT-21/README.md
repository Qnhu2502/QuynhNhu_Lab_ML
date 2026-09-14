# TT-21 - KNN Regressor: Dinh gia nha theo cac can tuong tu

## Huong dan chay
1. Cai thu vien: `pip install scikit-learn pandas numpy matplotlib joblib`
2. Notebook tu tai du lieu qua fetch_california_housing, khong can chuan bi file rieng
3. Restart Kernel roi Run All de chay dung thu tu tu dau den cuoi
4. Ket qua (model, hinh, requirements) duoc luu vao models/ va reports/ khi chay den cell cuoi

## Cac buoc da thuc hien va nhan xet

1. Nap du lieu California Housing, clip outlier AveRooms/AveBedrms/AveOccup chi theo tap train.
2. Baseline: Linear Regression RMSE=0.6795, R2=0.6477.
3. KNN khong scale: RMSE=1.0572 - rat kem vi Population (hang nghin) ap dao
   khoang cach so voi cac cot khac (vai don vi).
4. KNN co scale, K=5: RMSE=0.6051 - cai thien ro ret so voi khong scale.
5. Kho sat K tu 1 den 50, K toi uu tren tap test la 12. Tai K=1, RMSE train luon bang 0
   vi moi diem train tu tim thay chinh no la hang xom gan nhat - day la overfit tuyet doi,
   khong duoc dung RMSE train de chon K.
6. So sanh weights: distance thuong nhinh hon uniform vi uu tien can gan hon trong K can.
7. So sanh metric: euclidean va manhattan cho ket qua gan nhau tren bo du lieu nay.
8. Thi nghiem trong so vi tri (nhan Lat/Lon x1,2,3,5): {1: np.float64(0.591823807817399), 2: np.float64(0.554421928738648), 3: np.float64(0.5317996715359223), 5: np.float64(0.5033700037718527)}
   - RMSE giam
   khi tang trong so vi tri, xac nhan vi tri la yeu to quan trong voi gia nha.
9. Da in ra 5 can tuong tu cho 3 can test mau (xem reports/can_tuong_tu_vi_du.png) - day la
   uu the rieng cua KNN: chi ra duoc CAN CU cu the, khac voi Linear/RF chi cho ra mot con so.
10. Bang so sanh 3 model:
            Model     RMSE  Thoi_gian_train(s)  Thoi_gian_predict_1_can(s)          Giai_thich_duoc
Linear Regression 0.679471            0.005620                    0.001428       Co (he so ro rang)
              KNN 0.591824            0.030399                    0.019078 Co (chi ra can tuong tu)
    Random Forest 0.503714            7.096157                    0.054597          Khong (hop den)
11. Thoi gian du doan tang theo kich thuoc du lieu train: {1: 0.20643401145935059, 5: 0.28156256675720215, 10: 0.2394397258758545}
    - xac nhan KNN khong luu mot ham so gon nhe, phai tinh khoang cach den toan bo du lieu
    train moi lan du doan, nen cham dan khi du lieu lon.

## Han che
- KNN cham dan khi du lieu train lon, khong phu hop trien khai voi hang trieu ban ghi neu khong dung KDTree/BallTree.
- Khong ngoai suy duoc: nha co gia tri dac trung vuot xa moi can trong train se bi gan voi cac can gan nhat, khong phan anh dung xu huong ngoai bien.
- Can luu toan bo du lieu train de du doan (khong nhu Linear/RF chi can he so/cay da hoc), ton bo nho khi trien khai.
