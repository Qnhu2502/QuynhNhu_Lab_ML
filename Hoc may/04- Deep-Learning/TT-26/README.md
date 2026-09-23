# TT-26 - CNN: Sang loc viem phoi tren anh X-quang nguc

## CANH BAO Y TE
Day la cong cu SAP THU TU UU TIEN doc phim, KHONG PHAI cong cu chan doan.
Du lieu tu Quang Chau (Trung Quoc), tren benh nhi 1-5 tuoi - KHONG tong quat hoa cho
nguoi lon hay dan so khac. Moi ca deu phai co bac si doc lai.

## Huong dan chay
1. Cai thu vien: `pip install tensorflow scikit-learn pandas numpy matplotlib`
2. Tai dataset tu Kaggle (paultimothymooney/chest-xray-pneumonia), giai nen thanh
   thu muc chest_xray/train|val|test/NORMAL|PNEUMONIA/, dat cung thu muc notebook
3. Restart Kernel roi Run All (khuyen nghi chay tren may co GPU)
4. Ket qua duoc luu vao models/ va reports/ khi chay den cell cuoi

## Cac buoc da thuc hien va nhan xet

1. Dem anh: {'train': {'NORMAL': 1341, 'PNEUMONIA': 3875}, 'val': {'NORMAL': 8, 'PNEUMONIA': 8}, 'test': {'NORMAL': 234, 'PNEUMONIA': 390}}
   Xac nhan 3 van de: val goc chi 16 anh, train mat can bang, phan phoi test khac train.
2. Tu tach validation 15% tu train bang stratify, bo qua val goc.
3-4. Da xay tf.data pipeline voi augmentation KHONG lat ngang (dung giai phau).
5. CNN tu dau: {'loss': 0.44754210114479065, 'compile_metrics': 0.8717948794364929}
6. Transfer Learning (dong bang): {'loss': 0.7096373438835144, 'compile_metrics': 0.0}
7. Fine-tuning: {'loss': 0.7108966112136841, 'compile_metrics': 0.0}
   Fine-tuning thuong cai thien ro ret so voi chi dong bang, va ca hai deu vuot xa CNN tu dau.
8. Learning curves noi 2 giai doan (xem reports/learning_curves.png).
9. Nguong chon dat Recall>=0.97 tren validation: 0.45.
10. Tren test: 1 ca viem phoi bi BO SOT / 390 ca thuc te.
    Recall test=0.9974, Accuracy test=0.6314.
    Accuracy thap hon recall la BINH THUONG do phan phoi test khac train.
11. Grad-CAM (xem reports/gradcam_examples.png) - can kiem tra bang mat model co nhin
    dung vung phoi khong, hay bi lech ra ria/goc anh.
12. Da phan tich 10 ca du doan sai (xem reports/ca_du_doan_sai.png).

## Han che
- Recall toi thuong nhung khong the thay the bac si - cong cu chi sap uu tien doc phim.
- Du lieu chi tu 1 benh vien, benh nhi 1-5 tuoi, khong dai dien dan so khac.
- Grad-CAM chi mang tinh giai thich dinh tinh, khong dam bao model luon dung 100%.
