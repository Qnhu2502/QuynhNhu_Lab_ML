"""TT-07 - Gradient Boosting - Adult Census Income
Dự đoán thu nhập >50K/năm để hỗ trợ chấm điểm hồ sơ vay tiêu dùng.
"""
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    HistGradientBoostingClassifier, AdaBoostClassifier,
)
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, accuracy_score
import joblib

DATA_DIR = "data"
REPORT_DIR = "reports"
MODEL_DIR = "models"

COLS = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']


# ---------- 1. Đọc & làm sạch dữ liệu ----------
def load_clean(path, skiprows=0, strip_dot=False):
    df = pd.read_csv(path, header=None, names=COLS, skiprows=skiprows,
                      skipinitialspace=True, na_values='?')
    for c in df.select_dtypes(include='object').columns:
        df[c] = df[c].str.strip()
    df['income'] = df['income'].str.rstrip('.')  # adult.test có dấu '.' cuối nhãn
    df['income'] = (df['income'] == '>50K').astype(int)
    df = df.drop(columns=['fnlwgt', 'education'])  # bẫy 3 & 4
    return df


def main():
    train_df = load_clean(f"{DATA_DIR}/adult.data")
    test_df = load_clean(f"{DATA_DIR}/adult.test", skiprows=1)

    print(f"Train: {train_df.shape}, Test: {test_df.shape}")
    print(f"Tỉ lệ >50K (train): {train_df['income'].mean():.3f}")
    print(f"Missing values (train):\n{train_df.isna().sum()[train_df.isna().sum() > 0]}")

    y_train, y_test = train_df.pop('income'), test_df.pop('income')
    X_train, X_test = train_df, test_df

    cat_cols = X_train.select_dtypes(include='object').columns.tolist()
    num_cols = X_train.select_dtypes(exclude='object').columns.tolist()

    # ---------- 2. EDA nhanh ----------
    eda = X_train.assign(income=y_train)
    print("\n>50K theo education-num (top/bottom):")
    print(eda.groupby('education-num')['income'].mean().round(3))
    print("\n>50K theo marital-status:")
    print(eda.groupby('marital-status')['income'].mean().round(3))
    print(f"\ncapital-gain = 0: {(X_train['capital-gain'] == 0).mean():.1%}")

    # ---------- 3. Pipeline tiền xử lý (Boosting không cần scale) ----------
    pre = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
        ('num', 'passthrough', num_cols),
    ])

    def make_pipe(model):
        return Pipeline([('pre', pre), ('clf', model)])

    results = {}

    # ---------- 4. Baseline ----------
    for name, model in [('Dummy', DummyClassifier(strategy='most_frequent')),
                         ('DecisionTree', DecisionTreeClassifier(max_depth=6, random_state=42))]:
        pipe = make_pipe(model).fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        results[name] = {'pr_auc': average_precision_score(y_test, proba),
                          'roc_auc': roc_auc_score(y_test, proba), 'time': np.nan}
        print(f"{name}: PR-AUC={results[name]['pr_auc']:.3f}")

    # ---------- 5. Gradient Boosting mặc định (theo README) ----------
    gb = GradientBoostingClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=3,
        subsample=0.8, validation_fraction=0.1, n_iter_no_change=20,
        random_state=42,
    )
    gb_pipe = make_pipe(gb)
    t0 = time.time()
    gb_pipe.fit(X_train, y_train)
    gb_time = time.time() - t0
    proba = gb_pipe.predict_proba(X_test)[:, 1]
    results['GradientBoosting'] = {
        'pr_auc': average_precision_score(y_test, proba),
        'roc_auc': roc_auc_score(y_test, proba), 'time': gb_time}
    print(f"\nGradientBoosting: ROC-AUC={results['GradientBoosting']['roc_auc']:.3f}, "
          f"PR-AUC={results['GradientBoosting']['pr_auc']:.3f}, train_time={gb_time:.1f}s")

    # ---------- 6. Đường train/validation loss theo số cây ----------
    Xtr_t = gb_pipe.named_steps['pre'].transform(X_train)
    Xte_t = gb_pipe.named_steps['pre'].transform(X_test)
    fitted_gb = gb_pipe.named_steps['clf']
    train_loss = [log_loss(y_train, p) for p in fitted_gb.staged_predict_proba(Xtr_t)]
    test_loss = [log_loss(y_test, p) for p in fitted_gb.staged_predict_proba(Xte_t)]
    best_iter = int(np.argmin(test_loss))

    plt.figure(figsize=(7, 4))
    plt.plot(train_loss, label='Train loss')
    plt.plot(test_loss, label='Validation loss')
    plt.axvline(best_iter, color='red', ls='--', label=f'Overfit bắt đầu (~cây {best_iter})')
    plt.xlabel('Số cây (boosting stage)'); plt.ylabel('Log loss')
    plt.title('Train vs Validation loss theo số cây'); plt.legend()
    plt.tight_layout(); plt.savefig(f"{REPORT_DIR}/loss_theo_so_cay.png", dpi=120); plt.close()
    print(f"Điểm bắt đầu overfit ~ cây thứ {best_iter} / {len(test_loss)}")

    # ---------- 7. Dò learning_rate x n_estimators ----------
    lrs = [0.01, 0.05, 0.3]
    n_ests = [100, 300, 500]
    grid = np.zeros((len(lrs), len(n_ests)))
    for i, lr in enumerate(lrs):
        for j, ne in enumerate(n_ests):
            m = make_pipe(GradientBoostingClassifier(
                n_estimators=ne, learning_rate=lr, max_depth=3, subsample=0.8, random_state=42))
            m.fit(X_train, y_train)
            p = m.predict_proba(X_test)[:, 1]
            grid[i, j] = roc_auc_score(y_test, p)
    print("\nGrid ROC-AUC (hàng=learning_rate, cột=n_estimators):")
    print(pd.DataFrame(grid, index=lrs, columns=n_ests).round(4))

    plt.figure(figsize=(6, 4))
    im = plt.imshow(grid, cmap='viridis')
    plt.xticks(range(len(n_ests)), n_ests); plt.yticks(range(len(lrs)), lrs)
    plt.xlabel('n_estimators'); plt.ylabel('learning_rate'); plt.title('ROC-AUC: learning_rate x n_estimators')
    for i in range(len(lrs)):
        for j in range(len(n_ests)):
            plt.text(j, i, f"{grid[i,j]:.3f}", ha='center', va='center', color='white')
    plt.colorbar(im); plt.tight_layout()
    plt.savefig(f"{REPORT_DIR}/lr_vs_nestimators.png", dpi=120); plt.close()

    # ---------- 8. So sánh Bagging vs Boosting vs AdaBoost ----------
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1),
        'GradientBoosting': gb,
        'AdaBoost': AdaBoostClassifier(n_estimators=300, random_state=42),
    }
    comp_rows = []
    for name, model in models.items():
        pipe = make_pipe(model)
        t0 = time.time(); pipe.fit(X_train, y_train); dt = time.time() - t0
        p = pipe.predict_proba(X_test)[:, 1]
        comp_rows.append({'model': name, 'pr_auc': average_precision_score(y_test, p),
                           'roc_auc': roc_auc_score(y_test, p),
                           'n_params': sum(x.size for x in [np.array(1)]) if False else '-',
                           'train_time_s': round(dt, 1)})
    comp_df = pd.DataFrame(comp_rows)
    print("\nSo sánh Bagging vs Boosting vs AdaBoost:\n", comp_df)
    comp_df.to_csv(f"{REPORT_DIR}/model_comparison.csv", index=False)

    # ---------- 9. GradientBoosting vs HistGradientBoosting (thời gian) ----------
    pre_dense = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
        ('num', 'passthrough', num_cols),
    ])
    hgb = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05,
                                          early_stopping=True, random_state=42)
    hgb_pipe = Pipeline([('pre', pre_dense), ('clf', hgb)])
    t0 = time.time(); hgb_pipe.fit(X_train, y_train); hgb_time = time.time() - t0
    p_hgb = hgb_pipe.predict_proba(X_test)[:, 1]
    print(f"\nGradientBoosting: {gb_time:.1f}s | HistGradientBoosting: {hgb_time:.1f}s "
          f"(nhanh hơn {gb_time/hgb_time:.1f}x), ROC-AUC HGB={roc_auc_score(y_test, p_hgb):.3f}")

    # ---------- 10. Kiểm tra thiên lệch theo sex & race ----------
    pred = gb_pipe.predict(X_test)
    bias_rows = []
    for group_col in ['sex', 'race']:
        for g in X_test[group_col].unique():
            mask = X_test[group_col] == g
            if mask.sum() < 20:
                continue
            bias_rows.append({
                'group_col': group_col, 'group': g, 'n': int(mask.sum()),
                'positive_rate_true': round(y_test[mask].mean(), 3),
                'positive_rate_pred': round(pred[mask].mean(), 3),
                'accuracy': round(accuracy_score(y_test[mask], pred[mask]), 3),
            })
    bias_df = pd.DataFrame(bias_rows)
    print("\nThiên lệch theo sex/race:\n", bias_df)
    bias_df.to_csv(f"{REPORT_DIR}/bias_by_group.csv", index=False)

    plt.figure(figsize=(7, 4))
    sub = bias_df[bias_df.group_col == 'sex']
    plt.bar(sub['group'], sub['positive_rate_pred'], label='Dự đoán >50K')
    plt.bar(sub['group'], sub['positive_rate_true'], alpha=0.5, label='Thực tế >50K')
    plt.ylabel('Tỉ lệ >50K'); plt.title('Thiên lệch theo giới tính'); plt.legend()
    plt.tight_layout(); plt.savefig(f"{REPORT_DIR}/bias_by_group.png", dpi=120); plt.close()

    # ---------- Lưu model ----------
    joblib.dump(gb_pipe, f"{MODEL_DIR}/gb_pipeline.joblib")
    print(f"\nĐã lưu model tại {MODEL_DIR}/gb_pipeline.joblib")
    print("\n=== TỔNG KẾT ===")
    print(pd.DataFrame(results).T)


if __name__ == "__main__":
    main()
