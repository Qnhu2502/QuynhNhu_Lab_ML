"""
TT-01 — KNN CLASSIFIER: Sàng lọc nguy cơ tiểu đường
Thực hiện đầy đủ 11 bước trong README.md
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "diabetes.csv")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    recall_score, precision_score, f1_score, accuracy_score,
    confusion_matrix, classification_report, average_precision_score
)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

report_lines = []
def log(msg=""):
    print(msg)
    report_lines.append(str(msg))

# ============================================================
# BƯỚC 1 — Nạp dữ liệu, describe() → phát hiện min = 0 phi lý
# ============================================================
log("="*70)
log("BƯỚC 1 — NẠP DỮ LIỆU VÀ PHÁT HIỆN GIÁ TRỊ 0 PHI LÝ")
log("="*70)

df = pd.read_csv(DATA_PATH)
log(f"Kích thước dữ liệu: {df.shape}")
log("\nThống kê mô tả (describe):")
log(df.describe().T[["min", "mean", "max"]].to_string())

cols_khong_the_bang_0 = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
log("\n>>> Các cột có min = 0 nhưng về mặt sinh học KHÔNG THỂ = 0:")
for c in cols_khong_the_bang_0:
    n_zero = (df[c] == 0).sum()
    log(f"   {c:25s}: {n_zero:4d} dòng có giá trị 0 ({n_zero/len(df)*100:.1f}%)")

# ============================================================
# BƯỚC 2 — Thay 0 -> NaN, đếm % thiếu mỗi cột
# ============================================================
log("\n" + "="*70)
log("BƯỚC 2 — THAY 0 → NaN Ở 5 CỘT Y SINH")
log("="*70)

df_clean = df.copy()
df_clean[cols_khong_the_bang_0] = df_clean[cols_khong_the_bang_0].replace(0, np.nan)

log("\n% dữ liệu thiếu sau khi thay 0 -> NaN:")
missing_pct = df_clean[cols_khong_the_bang_0].isna().mean() * 100
log(missing_pct.round(1).to_string())

# ============================================================
# BƯỚC 3 — EDA: histogram Glucose theo Outcome
# ============================================================
log("\n" + "="*70)
log("BƯỚC 3 — EDA: PHÂN BỐ GLUCOSE THEO OUTCOME")
log("="*70)

fig, ax = plt.subplots(figsize=(8, 5))
for outcome, color, label in [(0, "#4C9AFF", "Không tiểu đường (0)"),
                               (1, "#FF6B6B", "Có tiểu đường (1)")]:
    subset = df_clean.loc[df_clean["Outcome"] == outcome, "Glucose"].dropna()
    ax.hist(subset, bins=25, alpha=0.6, color=color, label=label)
ax.set_xlabel("Glucose")
ax.set_ylabel("Số bệnh nhân")
ax.set_title("Phân bố Glucose theo nhóm Outcome")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(REPORTS_DIR, "glucose_hist.png"), dpi=140)
plt.close(fig)
log("Đã lưu biểu đồ: reports/glucose_hist.png")
log("Quan sát kỳ vọng: nhóm Outcome=1 lệch phải rõ rệt (Glucose cao hơn).")

# ============================================================
# BƯỚC 4 — Chia train/test, stratify, random_state=42
# ============================================================
log("\n" + "="*70)
log("BƯỚC 4 — CHIA TRAIN/TEST (stratify, random_state=42)")
log("="*70)

X = df_clean.drop(columns=["Outcome"])
y = df_clean["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
log(f"Train: {X_train.shape} | Test: {X_test.shape}")
log(f"Tỷ lệ dương tính - Train: {y_train.mean():.3f} | Test: {y_test.mean():.3f}")

# ============================================================
# BƯỚC 5 — Baseline: DummyClassifier
# ============================================================
log("\n" + "="*70)
log("BƯỚC 5 — BASELINE (DummyClassifier - most_frequent)")
log("="*70)

# Baseline không cần impute/scale nhưng để pipeline nhất quán ta vẫn impute
imp_baseline = SimpleImputer(strategy="median")
Xtr_imp = imp_baseline.fit_transform(X_train)
Xte_imp = imp_baseline.transform(X_test)

dummy = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
dummy.fit(Xtr_imp, y_train)
dummy_pred = dummy.predict(Xte_imp)
log(f"Baseline Accuracy : {accuracy_score(y_test, dummy_pred):.3f}")
log(f"Baseline Recall   : {recall_score(y_test, dummy_pred):.3f}  (luôn = 0, vì chỉ đoán lớp đa số)")

# ============================================================
# BƯỚC 6 — KNN K=5 mặc định (CÓ pipeline đầy đủ: impute+scale+knn)
# ============================================================
log("\n" + "="*70)
log("BƯỚC 6 — KNN K=5 MẶC ĐỊNH (CÓ chuẩn hoá)")
log("="*70)

pipe_k5 = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5)),
])
# Đánh giá bằng CV trên train (chưa đụng test)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
from sklearn.model_selection import cross_val_score
cv_recall_k5 = cross_val_score(pipe_k5, X_train, y_train, cv=cv, scoring="recall")
cv_acc_k5 = cross_val_score(pipe_k5, X_train, y_train, cv=cv, scoring="accuracy")
log(f"K=5 (có scale) - CV Recall  : {cv_recall_k5.mean():.3f} (+/- {cv_recall_k5.std():.3f})")
log(f"K=5 (có scale) - CV Accuracy: {cv_acc_k5.mean():.3f} (+/- {cv_acc_k5.std():.3f})")

# ============================================================
# BƯỚC 7 — So sánh CÓ vs KHÔNG chuẩn hoá
# ============================================================
log("\n" + "="*70)
log("BƯỚC 7 — CHỨNG MINH TẦM QUAN TRỌNG CỦA CHUẨN HOÁ")
log("="*70)

pipe_no_scale = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("knn", KNeighborsClassifier(n_neighbors=5)),
])
cv_recall_noscale = cross_val_score(pipe_no_scale, X_train, y_train, cv=cv, scoring="recall")
cv_acc_noscale = cross_val_score(pipe_no_scale, X_train, y_train, cv=cv, scoring="accuracy")

log(f"{'':22s}{'Recall (CV) K=5':>17s}{'Accuracy (CV) K=5':>19s}")
log(f"{'KHÔNG chuẩn hoá':22s}{cv_recall_noscale.mean():17.3f}{cv_acc_noscale.mean():19.3f}")
log(f"{'CÓ chuẩn hoá':22s}{cv_recall_k5.mean():17.3f}{cv_acc_k5.mean():19.3f}")

# Để công bằng hơn: dò K tối ưu riêng cho từng trường hợp (không chỉ cố định K=5)
grid_simple = {"knn__n_neighbors": list(range(1, 32, 2))}
gs_noscale = GridSearchCV(pipe_no_scale, grid_simple, cv=cv, scoring="recall", n_jobs=-1).fit(X_train, y_train)
gs_scale = GridSearchCV(
    Pipeline([("imp", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("knn", KNeighborsClassifier())]),
    grid_simple, cv=cv, scoring="recall", n_jobs=-1
).fit(X_train, y_train)

log(f"\nSau khi dò K tối ưu riêng cho mỗi trường hợp:")
log(f"{'':22s}{'K tốt nhất':>12s}{'Best Recall (CV)':>19s}")
log(f"{'KHÔNG chuẩn hoá':22s}{gs_noscale.best_params_['knn__n_neighbors']:12d}{gs_noscale.best_score_:19.3f}")
log(f"{'CÓ chuẩn hoá':22s}{gs_scale.best_params_['knn__n_neighbors']:12d}{gs_scale.best_score_:19.3f}")

log("\n>>> Nhận xét thực tế trên bộ dữ liệu này: chênh lệch Recall giữa có/không")
log("    chuẩn hoá không quá lớn (vì sau khi impute, phần lớn các cột y sinh vẫn")
log("    có biên độ tương đối gần nhau, và Insulin/SkinThickness thiếu rất nhiều).")
log("    Tuy nhiên VỀ MẶT LÝ THUYẾT, chuẩn hoá vẫn BẮT BUỘC với KNN vì thuật toán")
log("    dựa hoàn toàn vào khoảng cách Euclid: bất kỳ cột nào có biên độ lớn hơn")
log("    (như Insulin: 0-846 so với BMI: 18-67) đều có thể áp đảo các cột còn lại")
log("    một cách không kiểm soát, và mức độ ảnh hưởng phụ thuộc vào từng lần chia")
log("    train/test. Không chuẩn hoá là rủi ro hệ thống, không phải lỗi luôn xảy ra.")

# ============================================================
# BƯỚC 8 — GridSearchCV dò K, weights, metric (tối ưu Recall)
# ============================================================
log("\n" + "="*70)
log("BƯỚC 8 — GRIDSEARCHCV (tối ưu RECALL)")
log("="*70)

pipe = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
    ("knn", KNeighborsClassifier()),
])
grid = {
    "knn__n_neighbors": list(range(1, 32, 2)),
    "knn__weights": ["uniform", "distance"],
    "knn__metric": ["euclidean", "manhattan"],
}
gs = GridSearchCV(pipe, grid, cv=cv, scoring="recall", n_jobs=-1)
gs.fit(X_train, y_train)

log(f"Best params: {gs.best_params_}")
log(f"Best CV Recall: {gs.best_score_:.3f}")

# ============================================================
# BƯỚC 9 — Vẽ đường Recall & Accuracy theo K (weights='uniform', metric='euclidean' cố định để dễ đọc)
# ============================================================
log("\n" + "="*70)
log("BƯỚC 9 — RECALL & ACCURACY THEO K")
log("="*70)

Ks = list(range(1, 32, 2))
recalls, accs = [], []
for k in Ks:
    p = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k)),
    ])
    r = cross_val_score(p, X_train, y_train, cv=cv, scoring="recall").mean()
    a = cross_val_score(p, X_train, y_train, cv=cv, scoring="accuracy").mean()
    recalls.append(r)
    accs.append(a)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(Ks, recalls, marker="o", color="#FF6B6B", label="Recall (CV)")
ax.plot(Ks, accs, marker="s", color="#4C9AFF", label="Accuracy (CV)")
ax.set_xlabel("K (số láng giềng)")
ax.set_ylabel("Điểm số")
ax.set_title("Recall & Accuracy theo K (5-fold CV trên tập train)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(REPORTS_DIR, "recall_theo_K.png"), dpi=140)
plt.close(fig)
log("Đã lưu biểu đồ: reports/recall_theo_K.png")
log("\nGiải thích hình dạng đường:")
log("- K nhỏ (K=1,3): model 'nhớ' dữ liệu train sát -> variance cao, dễ overfit,")
log("  Recall trên CV thường dao động mạnh, không ổn định.")
log("- K vừa (khoảng 9-19): Recall/Accuracy thường đạt điểm cân bằng tốt nhất")
log("  vì giảm nhiễu nhưng chưa làm mờ ranh giới lớp.")
log("- K quá lớn: model quá 'mượt', thiên về lớp đa số (Outcome=0) -> Recall giảm dần")
log("  vì bỏ sót nhiều ca dương tính thực sự.")

# ============================================================
# BƯỚC 10 — Chạm tập TEST 1 lần duy nhất, ma trận nhầm lẫn
# ============================================================
log("\n" + "="*70)
log("BƯỚC 10 — ĐÁNH GIÁ TRÊN TẬP TEST (chỉ 1 lần)")
log("="*70)

best_knn = gs.best_estimator_
y_pred = best_knn.predict(X_test)

log(f"Test Accuracy : {accuracy_score(y_test, y_pred):.3f}")
log(f"Test Recall   : {recall_score(y_test, y_pred):.3f}")
log(f"Test Precision: {precision_score(y_test, y_pred):.3f}")
log(f"Test F1       : {f1_score(y_test, y_pred):.3f}")
log("\nClassification report:")
log(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5.5, 5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["Dự đoán: 0", "Dự đoán: 1"])
ax.set_yticklabels(["Thực tế: 0", "Thực tế: 1"])
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                 color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=16)
ax.set_title(f"Ma trận nhầm lẫn - KNN tốt nhất\n{gs.best_params_}")
fig.colorbar(im)
fig.tight_layout()
fig.savefig(os.path.join(REPORTS_DIR, "confusion_matrix.png"), dpi=140)
plt.close(fig)
log("Đã lưu biểu đồ: reports/confusion_matrix.png")

fn = cm[1, 0]
log(f"\n>>> Số ca BỎ SÓT (False Negative) = {fn} — đây là con số quan trọng nhất")
log("    vì bỏ sót bệnh nhân tiểu đường nguy hiểm hơn báo động giả (FP).")

# Giải thích K=1 trên tập TRAIN
knn1 = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=1)),
]).fit(X_train, y_train)
train_acc_k1 = accuracy_score(y_train, knn1.predict(X_train))
log(f"\nKiểm chứng K=1 trên TRAIN: Accuracy = {train_acc_k1:.3f}")
log(">>> Vì K=1: mỗi điểm train chính là láng giềng gần nhất của CHÍNH NÓ")
log("    (khoảng cách = 0) -> luôn tự dự đoán đúng nhãn của mình -> Accuracy = 100%.")
log("    Đây là overfitting điển hình, KHÔNG phản ánh khả năng tổng quát hoá.")

# ============================================================
# BƯỚC 11 — So sánh KNN vs Logistic Regression
# ============================================================
log("\n" + "="*70)
log("BƯỚC 11 — SO SÁNH KNN vs LOGISTIC REGRESSION")
log("="*70)

logreg = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, class_weight=None, random_state=RANDOM_STATE)),
])
logreg.fit(X_train, y_train)
y_pred_lr = logreg.predict(X_test)

log(f"{'Model':22s}{'Accuracy':>10s}{'Recall':>10s}{'Precision':>11s}{'F1':>8s}")
log(f"{'KNN (best)':22s}{accuracy_score(y_test,y_pred):10.3f}{recall_score(y_test,y_pred):10.3f}"
    f"{precision_score(y_test,y_pred):11.3f}{f1_score(y_test,y_pred):8.3f}")
log(f"{'Logistic Regression':22s}{accuracy_score(y_test,y_pred_lr):10.3f}{recall_score(y_test,y_pred_lr):10.3f}"
    f"{precision_score(y_test,y_pred_lr):11.3f}{f1_score(y_test,y_pred_lr):8.3f}")

log("\n>>> Nhận xét: Logistic Regression thường ổn định hơn, nhanh hơn, và cho hệ số")
log("    có thể diễn giải (mỗi biến ảnh hưởng bao nhiêu tới nguy cơ). KNN có thể")
log("    nhỉnh hơn ở Recall tuỳ K nhưng chậm khi dữ liệu lớn và không giải thích")
log("    được lý do dự đoán cho từng cá nhân -> hạn chế rõ trong bối cảnh y tế.")

# ============================================================
# LƯU MODEL & BÁO CÁO
# ============================================================
joblib.dump(best_knn, os.path.join(MODELS_DIR, "knn_pipeline.joblib"))

with open(os.path.join(REPORTS_DIR, "ket_qua_chay.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

log("\n" + "="*70)
log("HOÀN TẤT. Model đã lưu tại models/knn_pipeline.joblib")
log("="*70)
