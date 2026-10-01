"""Sinh README.md từ kết quả THẬT của notebook, để số liệu trong README không bao giờ lệch với code."""
from __future__ import annotations

import pandas as pd

from .pca_pipeline import N_FEATURES


def md_bang(df: pd.DataFrame, dinh_dang: dict[str, str] | None = None, giu_chi_so: bool = False) -> str:
    """DataFrame → bảng markdown (không cần thư viện tabulate). dinh_dang: {tên cột: format, vd '{:.4f}'}."""
    dinh_dang = dinh_dang or {}
    if giu_chi_so:
        df = df.rename_axis("dac_trung").reset_index()
    df = df.astype(object)          # giữ nguyên kiểu int/float từng cột (iterrows sẽ ép về float nếu cột lẫn kiểu)
    cot = list(df.columns)
    dong = ["| " + " | ".join(map(str, cot)) + " |", "|" + "---|" * len(cot)]
    for _, r in df.iterrows():
        o = []
        for c in cot:
            v = r[c]
            if c in dinh_dang:
                o.append(dinh_dang[c].format(v))
            elif isinstance(v, float):
                o.append(f"{v:.4f}")
            else:
                o.append(str(v))
        dong.append("| " + " | ".join(o) + " |")
    return "\n".join(dong)


# Định dạng cột cho từng bảng (đặt ngoài f-string để tránh nhầm với dấu {{ }})
DD_BANG_NGUONG = {'Nguong_phuong_sai': '{:.0%}', 'Ty_le_giam_chieu': '{:.1%}'}
DD_DF_CV = {'K': '{}', 'cv_accuracy': '{:.4f}', 'cv_accuracy_std': '{:.4f}', 'cv_f1_macro': '{:.4f}', 'thoi_gian_fit_s': '{:.2f}', 'ty_le_giam_chieu': '{:.1%}'}
DD_BANG_TEST = {'K': '{}', 'Accuracy': '{:.4f}', 'F1_macro': '{:.4f}', 'Thoi_gian_fit_s': '{:.2f}'}
DD_SO_SANH_KBEST = {'K': '{}', 'PCA_cv_accuracy': '{:.4f}', 'SelectKBest_cv_accuracy': '{:.4f}'}
DD_PC1_TOP = {'he_so': '{:.4f}'}
DD_BANG_THIET_BI = {'K': '{}', 'du_lieu_train_MB': '{:.2f}', 'kb_hang_so_float32': '{:.1f}', 'kb_ma_tran_chieu_float32': '{:.1f}', 'so_phep_nhan_cong': '{}', 'so_float_gui_di': '{}'}
DD_BANG_SAI_SO = {'K': '{}', 'MSE_train': '{:.4f}', 'MSE_test': '{:.4f}'}


def viet_readme(kq: dict) -> str:
    k = kq["k_ngot"]
    bt = kq["bang_test"].set_index("Mo_hinh")
    ten_pca = next(t for t in bt.index if t.startswith("PCA"))
    ten_full = next(t for t in bt.index if t.startswith("Không giảm chiều"))
    ten_kbest = next(t for t in bt.index if t.startswith("SelectKBest"))
    acc_pca, acc_full, acc_kbest = bt.loc[ten_pca, "Accuracy"], bt.loc[ten_full, "Accuracy"], bt.loc[ten_kbest, "Accuracy"]
    chenh = acc_full - acc_pca
    trong_dung_sai = "nằm trong" if chenh <= kq["dung_sai"] else "VƯỢT"
    ti_le_thoi_gian = bt.loc[ten_full, "Thoi_gian_fit_s"] / bt.loc[ten_pca, "Thoi_gian_fit_s"]
    tb = kq["bang_thiet_bi"].set_index("K")
    mb_goc, mb_k = tb.loc[N_FEATURES, "du_lieu_train_MB"], tb.loc[k, "du_lieu_train_MB"]
    ty_le_hang_so = tb.loc[k, "kb_hang_so_float32"] / tb.loc[N_FEATURES, "kb_hang_so_float32"]
    ty_le_phep_tinh = tb.loc[k, "so_phep_nhan_cong"] / tb.loc[N_FEATURES, "so_phep_nhan_cong"]
    std_cv = float(kq["df_cv"].loc[kq["df_cv"]["K"] == k, "cv_accuracy_std"].iloc[0])
    sk = kq["so_sanh_kbest"]
    k_pca_hon = ", ".join(str(int(x)) for x in sk.loc[sk["PCA_cv_accuracy"] > sk["SelectKBest_cv_accuracy"], "K"]) or "không K nào"
    k_kbest_hon = ", ".join(str(int(x)) for x in sk.loc[sk["PCA_cv_accuracy"] < sk["SelectKBest_cv_accuracy"], "K"]) or "không K nào"
    cap = kq["cap_nham"]
    cap_txt = "; ".join(f"{r.That} → {r.Du_doan} ({r.So_ca} ca)" for r in cap.itertuples())
    pc1_lop = kq["pc1_theo_lop"]
    ten_pc1 = list(kq["pc1_top"].index)
    n_nang_luong = sum(any(t in ten for t in ("sma", "Jerk", "Mag", "mad", "std", "energy")) for ten in ten_pc1)
    dong_tinh = kq["pc1_tach_dong_tinh"]
    nhan_xet_pc1 = (
        f"{n_nang_luong}/{len(ten_pc1)} đặc trưng top thuộc nhóm sma/Jerk/Mag/mad/std/energy (biên độ và năng lượng chuyển động). "
        + ("Các lớp chuyển động (WALKING*) nằm hẳn về một phía của PC1, các lớp tĩnh (SITTING, STANDING, LAYING) về phía kia, "
           "nên PC1 xấp xỉ **\"mức vận động\"** (dấu của PC là tuỳ ý). " if dong_tinh else
           "Trung bình PC1 của nhóm chuyển động và nhóm tĩnh KHÔNG tách hẳn nhau trong lần chạy này, nên diễn giải \"mức vận động\" chỉ mang tính gợi ý. ")
        + "PC1 là tổ hợp tuyến tính của hàng trăm đặc trưng tương quan mạnh nên không có ý nghĩa vật lý trực tiếp.")
    canh_bao_k = ("" if k < N_FEATURES else
                  "\n> ⚠️ CV cho thấy không K nào nhỏ hơn 561 thoả dung sai; không có điểm ngọt giảm chiều trong lần chạy này.\n")

    return f"""# TT-24 — PCA: nén {N_FEATURES} tín hiệu cảm biến cho thiết bị đeo

Bài toán: vòng đeo tay thu {N_FEATURES} chỉ số (gia tốc kế + con quay) để nhận biết 6 hoạt động; chip yếu (64 KB RAM, pin 7 ngày)
nên cần nén xuống vài chục–vài trăm chiều mà vẫn nhận dạng đúng. Dữ liệu: UCI HAR Smartphones
([link](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)).

## Kết quả chính

- **Điểm ngọt: K = {k}** (giảm {1 - k / N_FEATURES:.1%} số chiều), chọn bằng **cross-validation chia theo người trên tập train**, không dùng test.
  Quy tắc: K nhỏ nhất có accuracy CV ≥ accuracy CV của {N_FEATURES} chiều ({kq['baseline_cv']:.4f}) − {kq['dung_sai']:.0%}; PCA(K={k}) đạt {kq['acc_cv_ngot']:.4f}.
- **Test (9 người chưa từng thấy, đánh giá một lần):** PCA K={k} accuracy **{acc_pca:.4f}** so với **{acc_full:.4f}** khi dùng đủ {N_FEATURES} chiều
  (chênh {chenh:+.4f}, {trong_dung_sai} dung sai {kq['dung_sai']:.0%} đặt ra trên CV). Độ lệch chuẩn accuracy giữa các fold CV tại K={k} là {std_cv:.4f}, cùng cỡ hoặc lớn hơn mức chênh này, nên không nên đọc quá chi tiết sự khác biệt giữa các K lân cận.
- Dữ liệu train giảm từ **{mb_goc:.2f} MB** xuống **{mb_k:.2f} MB** (float64); thời gian fit nhanh gấp **{ti_le_thoi_gian:.1f} lần**.
- **Lưu ý triển khai (ước lượng phân tích, mục "Dung lượng"):** chuỗi scaler + PCA(K={k}) + LinearSVC không gộp cần ~{tb.loc[k, "kb_hang_so_float32"]:.1f} KB hằng số
  và {int(tb.loc[k, "so_phep_nhan_cong"]):,} phép nhân-cộng mỗi cửa sổ, gấp {ty_le_hang_so:.1f}× và {ty_le_phep_tinh:.1f}× so với LinearSVC chạy thẳng trên {N_FEATURES} đặc trưng
  ({tb.loc[N_FEATURES, "kb_hang_so_float32"]:.1f} KB; gộp tuyến tính còn {kq['fused_kb']:.1f} KB). Lợi ích của PCA ở bài này nằm ở dữ liệu lưu/truyền và thời gian huấn luyện, không ở suy luận.
{canh_bao_k}
## Cấu trúc & cách chạy

```
TT-24-PCA/
├── README.md                  ← sinh tự động từ notebook (số liệu luôn khớp code)
├── notebooks/pca_har_sensors.ipynb
├── src/{{pca_pipeline.py, bieu_do.py, bao_cao.py}}
├── tests/test_diem_ngot.py
├── models/pca_pipeline.joblib ← Pipeline(scale → PCA(K) → LinearSVC) đã fit trên train
├── reports/                   ← biểu đồ PNG + metrics.json + các bảng CSV
├── data/                      ← har_train.csv, har_test.csv (không commit)
└── requirements.txt
```

1. `pip install -r requirements.txt`
2. Đặt `har_train.csv`, `har_test.csv` vào `data/` (hoặc đặt biến môi trường `HAR_DATA_DIR`).
3. Mở `notebooks/pca_har_sensors.ipynb`, **Restart Kernel → Run All** (đường dẫn không phụ thuộc thư mục chạy; mọi `random_state=42`).
4. Kiểm thử: `pip install pytest && pytest`.

## Phương pháp (chống rò rỉ và cách chọn K)

- **Chia theo người, giữ nguyên bộ gốc:** {kq['n_nguoi_train']} người train ({kq['n_train']} mẫu) / {kq['n_nguoi_test']} người test ({kq['n_test']} mẫu), code kiểm tra không người nào trùng.
- **StandardScaler và PCA nằm trong `Pipeline`**, chỉ fit trên train (trong CV thì fit lại ở từng fold).
- **Chọn K bằng `GroupKFold(5)` theo người trên train**, lưới K dày ({len(kq['df_cv'])} giá trị từ 2 đến {N_FEATURES}).
  Không chọn K theo accuracy đo trên test (sẽ làm rò rỉ tập test vào quyết định), và lưới K phải đủ dày: với lưới thưa (…, 200, 561) so với accuracy đo trên test,
  tiêu chí "chênh < 1%" chỉ còn K=561, tức không giảm chiều nào. `tests/test_diem_ngot.py` có test cho đúng tình huống này.
- **Metric:** accuracy (lớp khá cân bằng) và macro-F1; có ma trận nhầm lẫn. Bộ phân loại: LinearSVC (C=1 mặc định, chưa tinh chỉnh).

## Scree plot, phương sai tích luỹ

![scree](reports/scree_plot.png) ![tich luy](reports/variance_tich_luy.png)

{md_bang(kq['bang_nguong'], DD_BANG_NGUONG)}

## Đánh đổi số chiều và độ chính xác (CV theo người, train)

![danh doi](reports/danh_doi_chieu_accuracy.png)

{md_bang(kq['df_cv'], DD_DF_CV)}

`thoi_gian_fit_s` là thời gian fit toàn pipeline (scale + PCA + LinearSVC) trung bình mỗi fold, trên máy chạy notebook.

## Đánh giá cuối trên test (một lần, sau khi đã chốt K)

{md_bang(kq['bang_test'], DD_BANG_TEST)}

Cặp bị nhầm nhiều nhất của PCA K={k}: {cap_txt}. Ma trận nhầm lẫn: `reports/confusion_pca.png`.

**So sánh SelectKBest cùng K (CV theo người):**

{md_bang(kq['so_sanh_kbest'], DD_SO_SANH_KBEST)}

Theo CV, PCA cao hơn SelectKBest ở K = {k_pca_hon}; SelectKBest cao hơn ở K = {k_kbest_hon}.

t-SNE chỉ để trực quan hoá (`reports/tsne_vs_pca.png`), không dùng làm tiền xử lý vì không có `transform` ổn định cho dữ liệu mới.

## PC1–PC2 và ý nghĩa PC1

![scatter](reports/pc1_pc2_scatter.png)

PC1 giữ {kq['pc1_ty_le']:.1%} phương sai. 10 đặc trưng gốc có |hệ số| lớn nhất (hệ số có dấu):

{md_bang(kq['pc1_top'], DD_PC1_TOP, giu_chi_so=True)}

Giá trị PC1 trung bình theo lớp (train): {", ".join(f"{t} {v:+.1f}" for t, v in pc1_lop.items())}.
{nhan_xet_pc1}

## Dung lượng và triển khai trên thiết bị

{md_bang(kq['bang_thiet_bi'], DD_BANG_THIET_BI)}

(`du_lieu_train_MB`: ma trận train float64, số liệu đo. Các cột còn lại là **ước lượng phân tích** cho suy luận 1 cửa sổ ở float32, không đo trên phần cứng.)

Nhận xét khi đưa vào thiết bị 64 KB RAM:
- Giảm chiều giúp rõ rệt ở **lưu trữ/truyền dữ liệu** ({N_FEATURES} → {k} số mỗi cửa sổ) và **thời gian huấn luyện**.
- Với bộ phân loại **tuyến tính**, scaler + PCA + LinearSVC gộp được thành một ma trận 6×{N_FEATURES} ({kq['fused_kb']:.1f} KB float32;
  sai lệch so với pipeline gốc {kq['fusion_sai_lech']:.1e}), nên PCA **không giảm số phép tính suy luận** so với chạy thẳng {N_FEATURES} đặc trưng.
  PCA có lợi cho suy luận khi bộ phân loại phía sau tốn kém hơn (SVM kernel, mạng nơ-ron) hoặc khi phép chiếu thực hiện ở phía thu.
- Ma trận chiếu K×{N_FEATURES} nên đặt ở flash (hằng số chỉ đọc); RAM chỉ cần bộ đệm cửa sổ và vector K chiều.
- Nếu mục tiêu là bớt việc tính đặc trưng trên chip, SelectKBest (chỉ tính K đặc trưng được giữ) mới thực sự tránh phải tính các đặc trưng còn lại; PCA vẫn cần cả {N_FEATURES}.

## Sai số tái tạo theo K

![tai tao](reports/sai_so_tai_tao.png)

{md_bang(kq['bang_sai_so'], DD_BANG_SAI_SO)}

## Hạn chế

- PCA tạo đặc trưng mới là tổ hợp tuyến tính của mọi đặc trưng cũ nên mất tính giải thích vật lý; PCA chỉ bắt quan hệ **tuyến tính**.
- CV chỉ có {kq['n_nguoi_train']} người (5 fold, mỗi fold khoảng 4 người) nên điểm ngọt có nhiễu; test chỉ có {kq['n_nguoi_test']} người.
- C của LinearSVC giữ mặc định, chưa tinh chỉnh cùng K; chưa thử Kernel PCA / IncrementalPCA.
- {N_FEATURES} đặc trưng là đặc trưng thủ công đã trích sẵn từ cửa sổ tín hiệu; chi phí tính chúng trên thiết bị không nằm trong các con số trên.
- Số liệu RAM/phép tính là ước lượng phân tích, chưa đo trên chip thật.
"""
