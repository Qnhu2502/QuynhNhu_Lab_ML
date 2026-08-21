# TT-03 — Random Forest: Ai sẽ mở sổ tiết kiệm?

## Cách chạy
1. Tải **`bank-full.csv`** (bộ **"bank"** gốc, KHÔNG phải `bank-additional-full.csv`) từ
   https://archive.ics.uci.edu/dataset/222/bank+marketing — trong file zip tải về là `bank.zip`,
   giải nén và đặt `bank-full.csv` vào `data/bank/bank-full.csv` (cùng cấp với `README.md`
   này, không có thư mục `notebooks/`).
   > Bộ `bank-additional-full.csv` dùng quy ước `pdays == 999` cho "chưa từng liên hệ", còn
   > `bank-full.csv` dùng `pdays == -1`. Nhầm bộ dữ liệu sẽ làm hỏng đặc trưng
   > `was_contacted_before` ở bước 3 của `random_forest.ipynb` — xem chú thích trong notebook.
2. `pip install -r requirements.txt`
3. Chạy `leakage_demo.ipynb` trước — chứng minh bằng số vì sao phải bỏ cột `duration`
   (AUC 0.929 nếu giữ lại vs 0.792 nếu bỏ — chênh lệch đó chính là rò rỉ dữ liệu).
4. Chạy `random_forest.ipynb` — pipeline chính, sinh ra:
   - `models/rf_pipeline.joblib`
   - `outputs/danh_sach_goi_top5000.csv`
   - `reports/*.png` và `reports/metrics.json`

## Cấu trúc
```
leakage_demo.ipynb     ← so sánh AUC có/không có duration
random_forest.ipynb    ← pipeline đầy đủ: tiền xử lý → baseline → RF (OOB + CV/GridSearch,
                          không tune bằng tập test) → feature importance → Precision@5000/Lift
                          → xuất danh sách gọi
requirements.txt
data/bank/bank-full.csv    ← tự tải về, KHÔNG commit (xem .gitignore)
outputs/  models/  reports/ ← sinh ra khi chạy notebook, KHÔNG commit
```

Cả hai notebook nằm ngay ở gốc thư mục `TT-03/` — mọi đường dẫn output trong notebook
(`reports/`, `outputs/`, `models/`) đều tương đối so với gốc này, không dùng `../`.

## Lưu ý khi commit
- Không commit `data/*.csv` (bộ dữ liệu gốc, ~10 MB) — mỗi người tự tải theo bước 1.
- Không commit nội dung sinh ra bởi notebook (`outputs/`, `models/`, `reports/`) — đã có trong
  `.gitignore`, tự sinh lại được bằng cách chạy notebook.
- Không commit rác hệ điều hành/IDE (`__MACOSX/`, `.DS_Store`, `.Rhistory`) — đã có trong
  `.gitignore`. Nếu đã lỡ commit, dọn bằng:
  ```
  git rm -r --cached __MACOSX .DS_Store .Rhistory outputs models reports data
  git commit -m "chore: gỡ dữ liệu/rác không nên commit"
  ```
