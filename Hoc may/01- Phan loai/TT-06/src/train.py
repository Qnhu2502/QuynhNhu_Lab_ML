"""
TT-06 - Naive Bayes - Loc tin nhan rac
Chay: python src/train.py
"""
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, precision_recall_curve,
)

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Nap du lieu (file goc encoding latin-1, khong phai utf-8)
# ---------------------------------------------------------------------------
df = pd.read_csv("spam.csv", encoding="latin-1")
df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "text"})
print(f"Du lieu goc: {df.shape[0]} dong")

# ---------------------------------------------------------------------------
# 2. Loai tin nhan trung lap (bo nay co ~400 dong trung)
# ---------------------------------------------------------------------------
n_dup = df.duplicated(subset=["label", "text"]).sum()
df = df.drop_duplicates(subset=["label", "text"]).reset_index(drop=True)
print(f"Da loai {n_dup} dong trung lap -> con {df.shape[0]} dong")

y = (df["label"] == "spam").astype(int)
print(df["label"].value_counts(normalize=True).round(3))

# ---------------------------------------------------------------------------
# 3. EDA nhanh: do dai tin ham vs spam
# ---------------------------------------------------------------------------
df["n_chars"] = df["text"].str.len()

fig, ax = plt.subplots(figsize=(7, 4))
df[df.label == "ham"]["n_chars"].hist(bins=40, alpha=0.6, label="ham", ax=ax)
df[df.label == "spam"]["n_chars"].hist(bins=40, alpha=0.6, label="spam", ax=ax)
ax.set_xlabel("So ky tu")
ax.set_ylabel("So tin nhan")
ax.set_title("Do dai tin nhan: ham vs spam")
ax.legend()
fig.tight_layout()
fig.savefig("reports/do_dai_tin.png", dpi=120)
plt.close(fig)

print("\nDo dai trung binh (ky tu):")
print(df.groupby("label")["n_chars"].mean().round(1))

# ---------------------------------------------------------------------------
# Chia train/test TRUOC khi vector hoa de tranh ro ri
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# ---------------------------------------------------------------------------
# 4. Baseline: DummyClassifier
# ---------------------------------------------------------------------------
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
print(f"\nBaseline DummyClassifier accuracy: {dummy.score(X_test, y_test):.4f} (vo dung, luon doan ham)")

# ---------------------------------------------------------------------------
# 5-7. Ba to hop: Count+Multinomial / TFIDF+Multinomial / TFIDF+Bernoulli
# ---------------------------------------------------------------------------
combos = {
    "Count + MultinomialNB": make_pipeline(
        CountVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, max_df=0.9),
        MultinomialNB(alpha=0.1),
    ),
    "TF-IDF + MultinomialNB": make_pipeline(
        TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True),
        MultinomialNB(alpha=0.1),
    ),
    "TF-IDF + BernoulliNB": make_pipeline(
        TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True),
        BernoulliNB(alpha=0.1),
    ),
}

rows = []
for name, pipe in combos.items():
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    rows.append({
        "to_hop": name,
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "accuracy": accuracy_score(y_test, pred),
    })

result_table = pd.DataFrame(rows).sort_values("f1", ascending=False)
print("\n=== Bang so sanh 3 to hop ===")
print(result_table.round(4).to_string(index=False))

# ---------------------------------------------------------------------------
# 8. Do alpha va ngram_range bang GridSearch tren to hop tot nhat (TF-IDF + MultinomialNB)
# ---------------------------------------------------------------------------
grid_pipe = make_pipeline(
    TfidfVectorizer(lowercase=True, min_df=2, max_df=0.9, sublinear_tf=True),
    MultinomialNB(),
)
param_grid = {
    "tfidfvectorizer__ngram_range": [(1, 1), (1, 2)],
    "multinomialnb__alpha": [0.01, 0.1, 0.5, 1],
}
grid = GridSearchCV(grid_pipe, param_grid, scoring="f1", cv=5, n_jobs=-1)
grid.fit(X_train, y_train)
print(f"\nGridSearch - tham so tot nhat: {grid.best_params_}, f1 cv: {grid.best_score_:.4f}")

best_model = grid.best_estimator_
best_pred = best_model.predict(X_test)
print("Danh gia tren test:")
print(f"  precision={precision_score(y_test, best_pred):.4f}  "
      f"recall={recall_score(y_test, best_pred):.4f}  "
      f"f1={f1_score(y_test, best_pred):.4f}")

# ---------------------------------------------------------------------------
# 9. Top 20 tu co P(tu|spam) cao nhat
# ---------------------------------------------------------------------------
vectorizer = best_model.named_steps["tfidfvectorizer"]
nb = best_model.named_steps["multinomialnb"]
feature_names = np.array(vectorizer.get_feature_names_out())
log_prob_spam = nb.feature_log_prob_[1]  # lop 1 = spam
top20_idx = np.argsort(log_prob_spam)[::-1][:20]
top20_words = feature_names[top20_idx]
print("\nTop 20 tu dac trung nhat cua spam:")
print(list(top20_words))

fig, ax = plt.subplots(figsize=(7, 6))
ax.barh(top20_words[::-1], log_prob_spam[top20_idx][::-1])
ax.set_xlabel("log P(tu | spam)")
ax.set_title("Top 20 tu dac trung cua spam")
fig.tight_layout()
fig.savefig("reports/top_tu_spam.png", dpi=120)
plt.close(fig)

# ---------------------------------------------------------------------------
# 10. Chon nguong dat precision >= 0.98
# ---------------------------------------------------------------------------
proba = best_model.predict_proba(X_test)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, proba)

target_precision = 0.98
ok = np.where(precisions[:-1] >= target_precision)[0]
if len(ok) > 0:
    chosen_idx = ok[0]  # nguong nho nhat dat duoc precision muc tieu -> recall cao nhat co the
    chosen_threshold = thresholds[chosen_idx]
    print(f"\nNguong dat precision >= {target_precision}: threshold={chosen_threshold:.3f}, "
          f"precision={precisions[chosen_idx]:.4f}, recall={recalls[chosen_idx]:.4f}")
else:
    chosen_threshold = 0.5
    print(f"\nKhong tim duoc nguong dat precision >= {target_precision}, dung mac dinh 0.5")

final_pred = (proba >= chosen_threshold).astype(int)
cm = confusion_matrix(y_test, final_pred)
print("Confusion matrix (nguong da chon):")
print(cm)

fig, ax = plt.subplots(figsize=(4.5, 4))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1], ["ham", "spam"])
ax.set_yticks([0, 1], ["ham", "spam"])
ax.set_xlabel("Du doan")
ax.set_ylabel("Thuc te")
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
ax.set_title(f"Confusion matrix (threshold={chosen_threshold:.2f})")
fig.tight_layout()
fig.savefig("reports/confusion_matrix.png", dpi=120)
plt.close(fig)

# ---------------------------------------------------------------------------
# 11. In 10 tin bi phan loai sai
# ---------------------------------------------------------------------------
X_test_reset = X_test.reset_index(drop=True)
y_test_reset = y_test.reset_index(drop=True)
wrong_mask = final_pred != y_test_reset.values
wrong_idx = np.where(wrong_mask)[0][:10]
print(f"\n10 tin bi phan loai sai (trong {wrong_mask.sum()} loi):")
for i in wrong_idx:
    thuc_te = "spam" if y_test_reset[i] == 1 else "ham"
    du_doan = "spam" if final_pred[i] == 1 else "ham"
    print(f"  [thuc te={thuc_te}, du doan={du_doan}] {X_test_reset[i][:80]}")

# ---------------------------------------------------------------------------
# 12. So sanh voi Logistic Regression + TF-IDF
# ---------------------------------------------------------------------------
logreg_pipe = make_pipeline(
    TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True),
    LogisticRegression(max_iter=1000, class_weight="balanced"),
)
logreg_pipe.fit(X_train, y_train)
logreg_pred = logreg_pipe.predict(X_test)
print("\nSo sanh voi Logistic Regression + TF-IDF:")
print(f"  NB   (best): precision={precision_score(y_test, best_pred):.4f}  "
      f"recall={recall_score(y_test, best_pred):.4f}  f1={f1_score(y_test, best_pred):.4f}")
print(f"  LogReg     : precision={precision_score(y_test, logreg_pred):.4f}  "
      f"recall={recall_score(y_test, logreg_pred):.4f}  f1={f1_score(y_test, logreg_pred):.4f}")

# ---------------------------------------------------------------------------
# Thoi gian train + predict 1 tin (yeu cau < 5ms)
# ---------------------------------------------------------------------------
t0 = time.time()
best_model.fit(X_train, y_train)
train_time = time.time() - t0

sample = [X_test.iloc[0]]
t0 = time.time()
best_model.predict(sample)
predict_time_ms = (time.time() - t0) * 1000

print(f"\nThoi gian train: {train_time:.3f}s")
print(f"Thoi gian du doan 1 tin: {predict_time_ms:.3f}ms")

# ---------------------------------------------------------------------------
# Luu model + bang ket qua
# ---------------------------------------------------------------------------
joblib.dump(best_model, "models/nb_pipeline.joblib")
result_table.to_csv("reports/bang_so_sanh.csv", index=False)
print("\nDa luu models/nb_pipeline.joblib va reports/bang_so_sanh.csv")
