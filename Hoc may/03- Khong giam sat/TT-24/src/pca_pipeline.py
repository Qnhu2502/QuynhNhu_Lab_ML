"""Logic chính của TT-24: nén 561 đặc trưng cảm biến HAR bằng PCA.

Nguyên tắc chống rò rỉ (leakage) được áp dụng xuyên suốt:
  * giữ nguyên cách chia train/test THEO NGƯỜI của bộ gốc (21/9 người, không trùng);
  * StandardScaler, PCA, SelectKBest luôn nằm TRONG Pipeline, chỉ được fit trên dữ liệu train;
  * chọn số chiều K bằng cross-validation chia theo người (GroupKFold) TRÊN TRAIN;
    tập test chỉ dùng để báo cáo kết quả cuối, không dùng để chọn K.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

N_FEATURES = 561
# Lưới K dày (không thưa) để "điểm ngọt" được xác định chính xác, không bị ép về K lớn nhất.
K_GRID = (2, 5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100, 125, 150, 175, 200, 250, 300, 400, N_FEATURES)
NGUONG_PHUONG_SAI = (0.80, 0.90, 0.95, 0.99)
DUNG_SAI_DIEM_NGOT = 0.01      # chấp nhận accuracy giảm tối đa 1 điểm % so với dùng đủ 561 chiều
N_FOLDS = 5
SO_LOP = 6                      # 6 hoạt động


# --------------------------------------------------------------------------- dữ liệu
@dataclass
class DuLieuHAR:
    X_train: pd.DataFrame
    y_train: pd.Series
    nguoi_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    nguoi_test: pd.Series

    @property
    def ten_dac_trung(self) -> list[str]:
        return list(self.X_train.columns)

    @property
    def ten_lop(self) -> list[str]:
        return sorted(self.y_train.unique())


def tim_file_du_lieu(ten_file: str) -> Path:
    """Dò file dữ liệu, không dùng đường dẫn tuyệt đối.
    Thứ tự: biến môi trường HAR_DATA_DIR → <gốc dự án>/data → gốc dự án → thư mục đang chạy (và cha của nó)."""
    thu_muc = []
    if os.environ.get("HAR_DATA_DIR"):
        thu_muc.append(Path(os.environ["HAR_DATA_DIR"]))
    thu_muc += [ROOT / "data", ROOT, Path.cwd(), Path.cwd().parent]
    for d in thu_muc:
        if (d / ten_file).is_file():
            return d / ten_file
    raise FileNotFoundError(
        f"Không tìm thấy {ten_file}. Đặt har_train.csv và har_test.csv vào thư mục data/ "
        "hoặc đặt biến môi trường HAR_DATA_DIR.")


def kiem_tra_chia_theo_nguoi(nguoi_train: pd.Series, nguoi_test: pd.Series) -> None:
    trung = set(nguoi_train) & set(nguoi_test)
    if trung:
        raise ValueError(f"Rò rỉ danh tính: người {sorted(trung)} có mặt ở cả train và test.")


def nap_du_lieu() -> DuLieuHAR:
    df_train = pd.read_csv(tim_file_du_lieu("har_train.csv"))
    df_test = pd.read_csv(tim_file_du_lieu("har_test.csv"))
    kiem_tra_chia_theo_nguoi(df_train["subject"], df_test["subject"])
    meta = ["subject", "Activity"]
    return DuLieuHAR(
        X_train=df_train.drop(columns=meta), y_train=df_train["Activity"], nguoi_train=df_train["subject"],
        X_test=df_test.drop(columns=meta), y_test=df_test["Activity"], nguoi_test=df_test["subject"])


# --------------------------------------------------------------------------- mô hình
def tao_pipeline(k: int | None = None, kieu: str = "pca") -> Pipeline:
    """Pipeline(scale → giảm chiều → LinearSVC). kieu: 'pca' | 'kbest'.
    k=None hoặc k>=561 nghĩa là KHÔNG giảm chiều (baseline 561 chiều)."""
    if k is None or k >= N_FEATURES:
        giam_chieu = "passthrough"
    elif kieu == "pca":
        giam_chieu = PCA(n_components=k, random_state=SEED)
    elif kieu == "kbest":
        giam_chieu = SelectKBest(f_classif, k=k)
    else:
        raise ValueError(f"kieu không hợp lệ: {kieu}")
    return Pipeline([
        ("scale", StandardScaler()),
        ("giam_chieu", giam_chieu),
        ("clf", LinearSVC(dual=False, max_iter=5000, random_state=SEED)),   # dual=False: n_mau > n_chieu
    ])


def danh_gia_cv(data: DuLieuHAR, k_list=K_GRID, kieu: str = "pca", n_folds: int = N_FOLDS) -> pd.DataFrame:
    """Cross-validation chia THEO NGƯỜI trên tập train. Scaler/PCA được fit lại trong từng fold.
    n_jobs=1 để thời gian fit đo được không bị nhiễu bởi chạy song song."""
    cv = GroupKFold(n_splits=n_folds)
    dong = []
    for k in k_list:
        r = cross_validate(tao_pipeline(k, kieu), data.X_train, data.y_train, groups=data.nguoi_train,
                           cv=cv, scoring=("accuracy", "f1_macro"), n_jobs=1)
        dong.append({"K": k, "cv_accuracy": r["test_accuracy"].mean(), "cv_accuracy_std": r["test_accuracy"].std(),
                     "cv_f1_macro": r["test_f1_macro"].mean(), "thoi_gian_fit_s": r["fit_time"].mean()})
    df = pd.DataFrame(dong)
    df["ty_le_giam_chieu"] = 1 - df["K"] / N_FEATURES
    return df


def chon_diem_ngot(df_cv: pd.DataFrame, dung_sai: float = DUNG_SAI_DIEM_NGOT) -> int:
    """Điểm ngọt = K NHỎ NHẤT có cv_accuracy >= cv_accuracy(dùng đủ 561 chiều) - dung_sai.
    Baseline là dòng K lớn nhất; lưới K phải đủ dày thì K nhỏ nhất thỏa điều kiện mới có nghĩa."""
    baseline = df_cv.loc[df_cv["K"].idxmax(), "cv_accuracy"]
    dat = df_cv[df_cv["cv_accuracy"] >= baseline - dung_sai]
    return int(dat["K"].min())


# --------------------------------------------------------------------------- PCA đầy đủ & phân tích
def fit_scaler_pca(X_train: pd.DataFrame) -> tuple[StandardScaler, PCA]:
    scaler = StandardScaler().fit(X_train)
    pca_full = PCA(random_state=SEED).fit(scaler.transform(X_train))
    return scaler, pca_full


def so_chieu_cho_nguong(pca_full: PCA, nguong: float) -> int:
    return int(np.argmax(np.cumsum(pca_full.explained_variance_ratio_) >= nguong)) + 1


def bang_nguong_phuong_sai(pca_full: PCA, nguong_list=NGUONG_PHUONG_SAI) -> pd.DataFrame:
    dong = []
    for n in nguong_list:
        k = so_chieu_cho_nguong(pca_full, n)
        dong.append({"Nguong_phuong_sai": n, "So_chieu_can": k, "Ty_le_giam_chieu": 1 - k / N_FEATURES})
    return pd.DataFrame(dong)


def sai_so_tai_tao(pca_full: PCA, X_scaled: np.ndarray, k_list) -> list[float]:
    """MSE tái tạo theo K. Fit PCA đầy đủ một lần rồi cắt K thành phần đầu (tương đương fit PCA(K))."""
    z = pca_full.transform(X_scaled)
    ket_qua = []
    for k in k_list:
        x_tai_tao = z[:, :k] @ pca_full.components_[:k] + pca_full.mean_
        ket_qua.append(float(np.mean((X_scaled - x_tai_tao) ** 2)))
    return ket_qua


def phan_tich_pc(pca_full: PCA, ten_dac_trung, chi_so: int = 0, top: int = 10) -> pd.DataFrame:
    """Hệ số (có dấu) của các đặc trưng gốc đóng góp nhiều nhất vào một thành phần chính."""
    he_so = pd.Series(pca_full.components_[chi_so], index=ten_dac_trung, name="he_so")
    return he_so.reindex(he_so.abs().sort_values(ascending=False).index).head(top).to_frame()


# --------------------------------------------------------------------------- đánh giá cuối trên test
def danh_gia_test(pipe: Pipeline, data: DuLieuHAR) -> dict:
    y_du_doan = pipe.predict(data.X_test)
    return {"accuracy": float(accuracy_score(data.y_test, y_du_doan)),
            "f1_macro": float(f1_score(data.y_test, y_du_doan, average="macro")),
            "confusion": confusion_matrix(data.y_test, y_du_doan, labels=data.ten_lop)}


def cap_nham_nhieu_nhat(cm: np.ndarray, ten_lop, top: int = 3) -> pd.DataFrame:
    """Các cặp (thật → đoán) bị nhầm nhiều nhất, bỏ đường chéo."""
    dong = [{"That": ten_lop[i], "Du_doan": ten_lop[j], "So_ca": int(cm[i, j])}
            for i in range(len(ten_lop)) for j in range(len(ten_lop)) if i != j and cm[i, j] > 0]
    return pd.DataFrame(dong).sort_values("So_ca", ascending=False).head(top).reset_index(drop=True)


# --------------------------------------------------------------------------- dung lượng & triển khai thiết bị
def dung_luong_du_lieu_mb(n_mau: int, so_chieu: int, so_byte: int = 8) -> float:
    return n_mau * so_chieu * so_byte / 1024 ** 2


def uoc_luong_thiet_bi(k: int, so_lop: int = SO_LOP, n: int = N_FEATURES) -> dict:
    """Ước lượng (phân tích, KHÔNG phải đo trên phần cứng) chi phí suy luận 1 cửa sổ trên thiết bị, float32.
    Chuỗi: chuẩn hoá (mean, scale) → chiếu PCA (mean, ma trận K×n) → LinearSVC (so_lop×K + so_lop)."""
    if k >= n:
        so_tham_so, mac = 2 * n + so_lop * n + so_lop, n + so_lop * n
    else:
        so_tham_so = 2 * n + n + k * n + so_lop * k + so_lop
        mac = n + k * n + so_lop * k
    return {"K": k, "kb_hang_so_float32": so_tham_so * 4 / 1024,
            "kb_ma_tran_chieu_float32": (k * n * 4 / 1024) if k < n else 0.0,
            "so_phep_nhan_cong": mac, "so_float_gui_di": k}


def gop_pipeline_tuyen_tinh(pipe: Pipeline) -> tuple[np.ndarray, np.ndarray]:
    """Gộp scaler + PCA + LinearSVC thành MỘT phép biến đổi tuyến tính trên 561 đặc trưng gốc:
    decision = X @ W + b. (Chỉ đúng khi bước giảm chiều là PCA hoặc passthrough.)"""
    scaler, giam_chieu, clf = pipe.named_steps["scale"], pipe.named_steps["giam_chieu"], pipe.named_steps["clf"]
    if isinstance(giam_chieu, str):      # "passthrough"
        a, mean_pc = clf.coef_.T, np.zeros(scaler.scale_.shape[0])
    else:
        a, mean_pc = giam_chieu.components_.T @ clf.coef_.T, giam_chieu.mean_
    w = a / scaler.scale_[:, None]
    b = clf.intercept_ - (scaler.mean_ / scaler.scale_ + mean_pc) @ a
    return w, b


# --------------------------------------------------------------------------- lưu / nạp
def luu_pipeline(pipe: Pipeline, duong_dan: Path | None = None) -> Path:
    duong_dan = duong_dan or MODELS_DIR / "pca_pipeline.joblib"
    duong_dan.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, duong_dan)
    return duong_dan


def nap_pipeline(duong_dan: Path | None = None) -> Pipeline:
    return joblib.load(duong_dan or MODELS_DIR / "pca_pipeline.joblib")


def _json_an_toan(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pd.DataFrame):
        return o.to_dict(orient="records")
    raise TypeError(f"Không serialize được {type(o)}")


def luu_json(obj: dict, duong_dan: Path) -> None:
    duong_dan.parent.mkdir(parents=True, exist_ok=True)
    duong_dan.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_json_an_toan), encoding="utf-8")
