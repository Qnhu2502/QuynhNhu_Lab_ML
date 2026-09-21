# TT-25 - MLP voi Keras: Cham diem khach hang tiem nang

## Huong dan chay
1. Cai thu vien: `pip install tensorflow lightgbm scikit-learn pandas numpy matplotlib`
2. Dat file `health_insurance.csv` cung thu muc voi notebook `notebooks/mlp_keras_insurance.ipynb`
3. Restart Kernel roi Run All de chay dung thu tu tu dau den cuoi (co GPU se nhanh hon dang ke)
4. Ket qua duoc luu vao models/ va reports/ khi chay den cell cuoi

## Cac buoc da thuc hien va nhan xet

1. EDA xac nhan Previously_Insured va Vehicle_Damage la 2 yeu to phan biet manh nhat ty le quan tam.
2. Tien xu ly: log1p Annual_Premium, one-hot bien it muc, giu Region_Code/Policy_Sales_Channel
   dang ma so (53 va 155 muc) de dung Embedding.
3. Chia train/val/test co stratify, ty le Response giu nguyen ~12.3% o ca 3 tap.
4. Baseline LightGBM: PR-AUC=0.3709, thoi gian train=5.1s.
5. MLP co ban (khong Dropout/BatchNorm): PR-AUC=0.3471.
6. MLP + Dropout/BatchNorm: PR-AUC=0.3483.
7. Learning curve (reports/learning_curves.png) dung de chan doan overfit/underfit qua EarlyStopping.
8. So sanh 3 kien truc (reports/kien_truc_comparison.png):
     Kien_truc  So_tham_so   PR_AUC
         (64,)        1025 0.343215
     (128, 64)       10497 0.348549
(256, 128, 64)       45825 0.351664
9. class_weight: khong dung PR-AUC=0.3463, co dung PR-AUC=0.3470.
10. Embedding vs one-hot (reports/embedding_vs_onehot.png): one-hot PR-AUC=0.3483
    (10497 tham so), embedding PR-AUC=0.3683
    (13953 tham so).
11. Precision@3000 = 0.4130, nguong xac suat tuong ung = 0.3596.
12. Ket luan trung thuc:
                Model   PR_AUC  Thoi_gian_train(s)                                        Cong_suc_tinh_chinh
             LightGBM 0.370894            5.116787                      Thap - it sieu tham so can tinh chinh
MLP Keras (embedding) 0.368286                 NaN Cao - nhieu kien truc/sieu tham so, can GPU de train nhanh

## Han che
- Voi du lieu dang bang, MLP thuong khong vuot troi ro ret so voi LightGBM, dung ky vong ly thuyet cua README.
- Embedding chi phat huy loi the khi so muc cua bien phan loai du lon va co du du lieu de hoc bieu dien tot.
- Chua thu nghiem TabNet/FT-Transformer hay xep chong model - la huong mo rong tiep theo.
