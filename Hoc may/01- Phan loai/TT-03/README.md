# TT-03 — Random Forest: Ai sẽ mở sổ tiết kiệm?

## Cách chạy
1. Tải `bank-additional-full.csv` từ https://archive.ics.uci.edu/dataset/222/bank+marketing
   và đặt vào thư mục gốc dự án (cùng cấp với `notebooks/`).
2. `pip install -r requirements.txt`
3. Chạy `notebooks/01_leakage_demo.ipynb` trước — chứng minh vì sao phải bỏ cột `duration`.
4. Chạy `notebooks/02_random_forest.ipynb` — pipeline chính, sinh ra:
   - `models/rf_pipeline.joblib`
   - `outputs/danh_sach_goi_top5000.csv`
   - `reports/*.png`

## Cấu trúc
```
notebooks/01_leakage_demo.ipynb   ← so sánh AUC có/không có duration
notebooks/02_random_forest.ipynb  ← pipeline đầy đủ (bước 3–11 trong đề bài)
requirements.txt
outputs/  models/  reports/       ← sinh ra khi chạy notebook
```
