"""
TT-02 — DECISION TREE: Dự đoán nhân viên nghỉ việc & rút ra quy tắc cho HR
Thực hiện đầy đủ 11 bước trong README.md
"""
import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "WA_Fn-UseC_-HR-Employee-Attrition.csv")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    confusion_matrix, classification_report
)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

report_lines = []
def log(msg=""):
    print(msg)
    report_lines.append(str(msg))

# ============================================================
# ☐ BƯỚC 1 — df.nunique() → phát hiện & bỏ 4 cột vô dụng
# ============================================================
log("="*70)
log("BƯỚC 1 — DF.NUNIQUE() → PHÁT HIỆN & BỎ CỘT VÔ DỤNG")
log("="*70)

df = pd.read_csv(DATA_PATH)
log(f"Kích thước dữ liệu gốc: {df.shape}")

nunique = df.nunique()
log("\nCác cột có SỐ GIÁ TRỊ DUY NHẤT <= 1 (0 thông tin):")
cot_vo_dung_hangso = nunique[nunique <= 1].index.tolist()
log(f"  {cot_vo_dung_hangso}")

log("\nCác cột có SỐ GIÁ TRỊ DUY NHẤT = số dòng (mã định danh):")
cot_dinh_danh = nunique[nunique == len(df)].index.tolist()
log(f"  {cot_dinh_danh}")

cot_bo = list(set(cot_vo_dung_hangso + cot_dinh_danh))
log(f"\n>>> Quyết định BỎ 4 cột: {cot_bo}")

df = df.drop(columns=cot_bo)
log(f"Kích thước sau khi bỏ cột vô dụng: {df.shape}")

# Mã hoá nhãn Attrition: Yes -> 1, No -> 0
df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})
log(f"\nTỷ lệ nghỉ việc (Attrition=1): {df['Attrition'].mean()*100:.1f}%")

# ============================================================
# ☐ BƯỚC 2 — Mã hoá biến phân loại (OneHotEncoder trong Pipeline)
# ============================================================
log("\n" + "="*70)
log("BƯỚC 2 — MÃ HOÁ BIẾN PHÂN LOẠI (OneHotEncoder trong Pipeline)")
log("="*70)

X = df.drop(columns=["Attrition"])
y = df["Attrition"]

cot_phan_loai = X.select_dtypes(include=["object"]).columns.tolist()
cot_so = X.select_dtypes(exclude=["object"]).columns.tolist()
log(f"Cột phân loại (categorical) — {len(cot_phan_loai)} cột: {cot_phan_loai}")
log(f"Cột số (numeric) — {len(cot_so)} cột: {cot_so}")

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cot_phan_loai),
        ("num", "passthrough", cot_so),
    ]
)
log("\n>>> Dùng ColumnTransformer + OneHotEncoder, KHÔNG cần StandardScaler")
log("    (Decision Tree không nhạy với thang đo của biến số).")

# ============================================================
# ☐ BƯỚC 3 — EDA: tỉ lệ nghỉ theo OverTime, JobRole, YearsAtCompany
# ============================================================
log("\n" + "="*70)
log("BƯỚC 3 — EDA: TỈ LỆ NGHỈ VIỆC THEO OVERTIME / JOBROLE / YEARSATCOMPANY")
log("="*70)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# OverTime
rate_ot = df.groupby("OverTime")["Attrition"].mean() * 100
axes[0].bar(rate_ot.index, rate_ot.values, color=["#4C9AFF", "#FF6B6B"])
axes[0].set_title("Tỉ lệ nghỉ việc theo OverTime")
axes[0].set_ylabel("% nghỉ việc")
for i, v in enumerate(rate_ot.values):
    axes[0].text(i, v + 1, f"{v:.1f}%", ha="center")

# JobRole
rate_role = df.groupby("JobRole")["Attrition"].mean().sort_values(ascending=False) * 100
axes[1].barh(rate_role.index, rate_role.values, color="#FF9F43")
axes[1].set_title("Tỉ lệ nghỉ việc theo JobRole")
axes[1].set_xlabel("% nghỉ việc")
axes[1].invert_yaxis()

# YearsAtCompany (bin)
bins = [0, 2, 5, 10, 20, 100]
labels = ["0-2", "3-5", "6-10", "11-20", "20+"]
df["_YearsBin"] = pd.cut(df["YearsAtCompany"], bins=bins, labels=labels, right=True, include_lowest=True)
rate_years = df.groupby("_YearsBin", observed=True)["Attrition"].mean() * 100
axes[2].bar(rate_years.index.astype(str), rate_years.values, color="#20BF6B")
axes[2].set_title("Tỉ lệ nghỉ việc theo YearsAtCompany")
axes[2].set_xlabel("Số năm tại công ty")
df.drop(columns=["_YearsBin"], inplace=True)

fig.tight_layout()
fig.savefig(os.path.join(REPORTS_DIR, "eda_attrition.png"), dpi=140)
plt.close(fig)

log(f"\nTỉ lệ nghỉ việc theo OverTime:\n{rate_ot.round(1).to_string()}")
log(f"\nTỉ lệ nghỉ việc theo JobRole (top 3 cao nhất):\n{rate_role.head(3).round(1).to_string()}")
log(f"\nTỉ lệ nghỉ việc theo YearsAtCompany:\n{rate_years.round(1).to_string()}")
log("\nĐã lưu: reports/eda_attrition.png")

# ============================================================
# ☐ BƯỚC 4 — Baseline: DummyClassifier
# ============================================================
log("\n" + "="*70)
log("BƯỚC 4 — CHIA TRAIN/TEST + BASELINE DUMMYCLASSIFIER")
log("="*70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
log(f"Train: {X_train.shape} | Test: {X_test.shape}")

Xtr_enc = preprocessor.fit_transform(X_train)
Xte_enc = preprocessor.transform(X_test)

dummy = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
dummy.fit(Xtr_enc, y_train)
dummy_pred = dummy.predict(Xte_enc)
log(f"Baseline Accuracy: {accuracy_score(y_test, dummy_pred):.3f}")
log(f"Baseline F1      : {f1_score(y_test, dummy_pred):.3f}  (luôn = 0 vì chỉ đoán lớp đa số)")

# ============================================================
# ☐ BƯỚC 5 — Cây KHÔNG giới hạn độ sâu → ghi accuracy train & test
# ============================================================
log("\n" + "="*70)
log("BƯỚC 5 — CÂY KHÔNG GIỚI HẠN ĐỘ SÂU (max_depth=None) → CHỨNG MINH OVERFIT")
log("="*70)

tree_full = Pipeline([
    ("prep", preprocessor),
    ("clf", DecisionTreeClassifier(max_depth=None, class_weight="balanced", random_state=RANDOM_STATE)),
])
tree_full.fit(X_train, y_train)

acc_train_full = accuracy_score(y_train, tree_full.predict(X_train))
acc_test_full = accuracy_score(y_test, tree_full.predict(X_test))
log(f"Accuracy TRAIN (max_depth=None): {acc_train_full:.3f}")
log(f"Accuracy TEST  (max_depth=None): {acc_test_full:.3f}")
log(f"Độ sâu cây thực tế mọc tới: {tree_full.named_steps['clf'].get_depth()}")
log(f"Số lá: {tree_full.named_steps['clf'].get_n_leaves()}")
log("\n>>>  Đúng như dự đoán: train gần như 100%, test thấp hơn nhiều → OVERFIT rõ rệt.")

# ============================================================
# ☐ BƯỚC 6 — Vẽ đường accuracy train/test theo max_depth = 1..20
# ============================================================
log("\n" + "="*70)
log("BƯỚC 6 — ACCURACY TRAIN/TEST THEO max_depth = 1..20")
log("="*70)

depths = list(range(1, 21))
train_accs, test_accs = [], []
for d in depths:
    p = Pipeline([
        ("prep", preprocessor),
        ("clf", DecisionTreeClassifier(max_depth=d, class_weight="balanced", random_state=RANDOM_STATE)),
    ])
    p.fit(X_train, y_train)
    train_accs.append(accuracy_score(y_train, p.predict(X_train)))
    test_accs.append(accuracy_score(y_test, p.predict(X_test)))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(depths, train_accs, marker="o", color="#4C9AFF", label="Train Accuracy")
ax.plot(depths, test_accs, marker="s", color="#FF6B6B", label="Test Accuracy")
gap = np.array(train_accs) - np.array(test_accs)
depth_overfit = depths[int(np.argmax(gap > 0.08))] if any(gap > 0.08) else None
ax.set_xlabel("max_depth")
ax.set_ylabel("Accuracy")
ax.set_title("Overfitting theo max_depth (Decision Tree)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(REPORTS_DIR, "overfit_theo_depth.png"), dpi=140)
plt.close(fig)

log("Đã lưu: reports/overfit_theo_depth.png")
log(f"\nKhoảng cách train-test lớn dần rõ rệt từ khoảng max_depth={depth_overfit}")
log("Giải thích: depth nhỏ → cây quá đơn giản (underfit nhẹ, train~test thấp).")
log("depth tăng → train tăng nhanh tới gần 100%, test bắt đầu chững/giảm nhẹ")
log("→ đó là vùng OVERFIT, cây bắt đầu học thuộc nhiễu của tập train.")

# ============================================================
# ☐ BƯỚC 7 — GridSearchCV: max_depth, min_samples_leaf, criterion (scoring='f1')
# ============================================================
log("\n" + "="*70)
log("BƯỚC 7 — GRIDSEARCHCV: max_depth, min_samples_leaf, criterion (scoring='f1')")
log("="*70)

pipe = Pipeline([
    ("prep", preprocessor),
    ("clf", DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE)),
])
grid = {
    "clf__max_depth": [3, 4, 5, 6, 7, 8],
    "clf__min_samples_leaf": [10, 20, 30, 50],
    "clf__criterion": ["gini", "entropy"],
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
gs = GridSearchCV(pipe, grid, cv=cv, scoring="f1", n_jobs=-1)
gs.fit(X_train, y_train)

log(f"Best params: {gs.best_params_}")
log(f"Best CV F1: {gs.best_score_:.3f}")

best_tree_pipe = gs.best_estimator_
best_tree = best_tree_pipe.named_steps["clf"]

# ============================================================
# ☐ BƯỚC 8 — Vẽ cây tốt nhất, xuất export_text
# ============================================================
log("\n" + "="*70)
log("BƯỚC 8 — VẼ CÂY TỐT NHẤT + EXPORT_TEXT")
log("="*70)

feature_names = preprocessor.get_feature_names_out()
# rút gọn tên cột cho dễ đọc (bỏ tiền tố cat__/num__)
feature_names_clean = [f.replace("cat__", "").replace("num__", "") for f in feature_names]

fig, ax = plt.subplots(figsize=(24, 12))
plot_tree(best_tree, feature_names=feature_names_clean, class_names=["Ở lại", "Nghỉ"],
          filled=True, rounded=True, fontsize=8, ax=ax)
fig.tight_layout()
fig.savefig(os.path.join(REPORTS_DIR, "cay_quyet_dinh.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
log("Đã lưu: reports/cay_quyet_dinh.png")

text_tree = export_text(best_tree, feature_names=list(feature_names_clean), max_depth=10)
log("\nexport_text (dạng văn bản của cây):")
log(text_tree)

with open(os.path.join(REPORTS_DIR, "cay_export_text.txt"), "w", encoding="utf-8") as f:
    f.write(text_tree)

# ============================================================
# ☐ BƯỚC 9 — Feature importance top 10
# ============================================================
log("\n" + "="*70)
log("BƯỚC 9 — FEATURE IMPORTANCE TOP 10")
log("="*70)

importances = pd.Series(best_tree.feature_importances_, index=feature_names_clean)
top10 = importances.sort_values(ascending=False).head(10)
log("\nTop 10 biến quan trọng nhất:")
log(top10.round(4).to_string())

fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(top10.index[::-1], top10.values[::-1], color="#845EF7")
ax.set_xlabel("Feature importance")
ax.set_title("Top 10 biến quan trọng nhất - Decision Tree")
fig.tight_layout()
fig.savefig(os.path.join(REPORTS_DIR, "feature_importance.png"), dpi=140)
plt.close(fig)
log("\nĐã lưu: reports/feature_importance.png")

# Kiểm tra đạo đức: cây có dùng Gender/MaritalStatus không?
nhay_cam = [f for f in feature_names_clean if "Gender" in f or "MaritalStatus" in f]
dung_bien_nhay_cam = importances.loc[importances.index.isin(nhay_cam)]
dung_bien_nhay_cam = dung_bien_nhay_cam[dung_bien_nhay_cam > 0]
log(f"\n⚖️ KIỂM TRA ĐẠO ĐỨC: các biến nhạy cảm (Gender/MaritalStatus) có importance > 0:")
if len(dung_bien_nhay_cam) > 0:
    log(dung_bien_nhay_cam.round(4).to_string())
    log(">>> CẢNH BÁO: cây có dùng biến nhạy cảm để chia nhánh, cần cân nhắc loại bỏ.")
else:
    log(">>> Cây tốt nhất KHÔNG dùng Gender/MaritalStatus làm nút chia. An toàn về đạo đức.")

# ============================================================
# ☐ BƯỚC 10 — Đánh giá trên TEST + trích xuất luật từ các lá
# ============================================================
log("\n" + "="*70)
log("BƯỚC 10 — ĐÁNH GIÁ TRÊN TEST (chạm 1 lần) + TRÍCH XUẤT CÁC LUẬT TỪ LÁ")
log("="*70)

y_pred = best_tree_pipe.predict(X_test)
log(f"Test Accuracy : {accuracy_score(y_test, y_pred):.3f}")
log(f"Test Recall   : {recall_score(y_test, y_pred):.3f}")
log(f"Test Precision: {precision_score(y_test, y_pred):.3f}")
log(f"Test F1       : {f1_score(y_test, y_pred):.3f}")
log("\nClassification report:\n" + classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
log(f"Confusion matrix:\n{cm}")

# Trích xuất luật tự động từ cấu trúc cây (đường đi tới từng lá)
# LƯU Ý: vì dùng class_weight='balanced', tree_.value là số liệu ĐÃ TRỌNG SỐ,
# không phải số người thật -> phải tính số người thật (raw count) bằng cách
# cho toàn bộ X_train đi qua cây (apply) rồi đếm nhãn thật theo từng lá.
def trich_xuat_luat(tree_clf, feature_names, X_raw_encoded, y_raw):
    tree_ = tree_clf.tree_
    duong_di = {}

    def duyet(node, path):
        if tree_.feature[node] != -2:  # không phải lá
            name = feature_names[tree_.feature[node]]
            thresh = tree_.threshold[node]
            duyet(tree_.children_left[node], path + [f"{name} <= {thresh:.2f}"])
            duyet(tree_.children_right[node], path + [f"{name} > {thresh:.2f}"])
        else:
            duong_di[node] = " VÀ ".join(path)
    duyet(0, [])

    leaf_ids = tree_clf.apply(X_raw_encoded)
    df_leaf = pd.DataFrame({"leaf": leaf_ids, "y": np.asarray(y_raw)})

    luat = []
    for leaf_id, nhom in df_leaf.groupby("leaf"):
        n_samples = len(nhom)
        n_nghi = int(nhom["y"].sum())
        luat.append({
            "dieu_kien": duong_di[leaf_id],
            "so_nguoi": n_samples,
            "so_nguoi_nghi": n_nghi,
            "ty_le_nghi_pct": round(n_nghi / n_samples * 100, 1),
        })
    return pd.DataFrame(luat).sort_values("ty_le_nghi_pct", ascending=False)

Xtr_encoded_for_rules = best_tree_pipe.named_steps["prep"].transform(X_train)
bang_luat = trich_xuat_luat(best_tree, feature_names_clean, Xtr_encoded_for_rules, y_train)
bang_luat_dang_chu = bang_luat[bang_luat["so_nguoi"] >= 20].reset_index(drop=True)  # chỉ lấy lá đủ lớn để đáng tin
log("\nBảng luật trích xuất từ các lá (chỉ lấy lá có >= 20 người, sắp theo tỉ lệ nghỉ giảm dần):")
log(bang_luat_dang_chu.to_string())
bang_luat_dang_chu.to_csv(os.path.join(REPORTS_DIR, "bang_luat_tu_cay.csv"), index=False)
log("\nĐã lưu: reports/bang_luat_tu_cay.csv")

# ============================================================
# ☐ BƯỚC 11 — So sánh với Random Forest
# ============================================================
log("\n" + "="*70)
log("BƯỚC 11 — SO SÁNH VỚI RANDOM FOREST → MẤT GÌ, ĐƯỢC GÌ?")
log("="*70)

rf = Pipeline([
    ("prep", preprocessor),
    ("clf", RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=5,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    )),
])
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

log(f"{'Model':22s}{'Accuracy':>10s}{'Recall':>10s}{'Precision':>11s}{'F1':>8s}")
log(f"{'Decision Tree (best)':22s}{accuracy_score(y_test,y_pred):10.3f}{recall_score(y_test,y_pred):10.3f}"
    f"{precision_score(y_test,y_pred):11.3f}{f1_score(y_test,y_pred):8.3f}")
log(f"{'Random Forest':22s}{accuracy_score(y_test,y_pred_rf):10.3f}{recall_score(y_test,y_pred_rf):10.3f}"
    f"{precision_score(y_test,y_pred_rf):11.3f}{f1_score(y_test,y_pred_rf):8.3f}")

# Kiểm tra độ ổn định của 1 cây đơn: train nhiều seed khác nhau, so sánh feature quan trọng nhất
log("\n>>> Kiểm tra độ ỔN ĐỊNH của cây đơn (đổi random_state, cùng tham số tốt nhất):")
top_feature_moi_seed = []
for seed in [0, 1, 2, 3, 4]:
    p = Pipeline([
        ("prep", preprocessor),
        ("clf", DecisionTreeClassifier(
            max_depth=gs.best_params_["clf__max_depth"],
            min_samples_leaf=gs.best_params_["clf__min_samples_leaf"],
            criterion=gs.best_params_["clf__criterion"],
            class_weight="balanced", random_state=seed)),
    ])
    p.fit(X_train, y_train)
    imp = pd.Series(p.named_steps["clf"].feature_importances_, index=feature_names_clean)
    top_feature_moi_seed.append(imp.idxmax())
log(f"Biến quan trọng nhất (top-1) qua 5 random_state khác nhau: {top_feature_moi_seed}")
log(">>> Nếu danh sách trên KHÔNG giống nhau hoàn toàn, đây là bằng chứng cây đơn KHÔNG ỔN ĐỊNH")
log("    — lý do Random Forest ra đời (trung bình nhiều cây để giảm phương sai).")

log("\n>>> Kết luận đánh đổi: Random Forest thường cho Accuracy/F1/Recall cao và ỔN ĐỊNH")
log("    hơn 1 cây đơn (giảm phương sai nhờ trung bình nhiều cây). Đổi lại, MẤT khả")
log("    năng đọc trực tiếp thành luật IF-THEN đơn giản — 300 cây không thể in ra giấy")
log("    cho HR đọc như 1 cây max_depth=4. Với mục tiêu bài này (rút quy tắc dễ hiểu),")
log("    Decision Tree đơn vẫn phù hợp hơn; Random Forest phù hợp hơn khi ưu tiên độ")
log("    chính xác/ổn định và chấp nhận mất khả năng diễn giải trực tiếp.")

# ============================================================
# LƯU MODEL & BÁO CÁO
# ============================================================
joblib.dump(best_tree_pipe, os.path.join(MODELS_DIR, "tree.joblib"))

with open(os.path.join(REPORTS_DIR, "ket_qua_chay.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

with open(os.path.join(REPORTS_DIR, "best_params.json"), "w", encoding="utf-8") as f:
    json.dump(gs.best_params_, f, ensure_ascii=False, indent=2)

log("\n" + "="*70)
log("HOÀN TẤT. Model đã lưu tại models/tree.joblib")
log("="*70)
