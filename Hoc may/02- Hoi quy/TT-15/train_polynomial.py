"""
TT-15 — Polynomial Regression: du bao cong suat nha may dien (PE)
Ban script hoa cua notebook polynomial_power_plant.ipynb.

Cach chay:
    python src/train.py --data Folds5x2_pp.xlsx

Ket qua:
    reports/scatter_AT_PE.png
    reports/residual_bac1_truoc.png, residual_truoc_sau.png
    reports/validation_curve.png
    reports/bang_bac_so_cot_rmse.csv
    reports/linear_vs_ridge_bac_cao.csv
    models/poly_pipeline.joblib
"""
import argparse
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler

RANDOM_STATE = 42
FEATURES = ["AT", "V", "AP", "RH"]
TARGET = "PE"
DEGREES = [1, 2, 3, 4, 5]
IMPROVEMENT_THRESHOLD = 0.02  # MW


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Khong tim thay file du lieu tai: {os.path.abspath(path)}\n"
            "-> Truyen dung duong dan bang --data /duong/dan/Folds5x2_pp.xlsx"
        )
    if path.endswith(".ods"):
        return pd.read_excel(path, engine="odf")
    return pd.read_excel(path)


def make_pipeline(degree: int, alpha: float = 1.0, use_ridge: bool = True) -> Pipeline:
    model = Ridge(alpha=alpha, random_state=RANDOM_STATE) if use_ridge else LinearRegression()
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scale", StandardScaler()),  # SAU poly — bat buoc
        ("model", model),
    ])


def plot_scatter(df: pd.DataFrame, out_path: str):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, feat in zip(axes.ravel(), FEATURES):
        ax.scatter(df[feat], df[TARGET], s=5, alpha=0.3)
        ax.set_xlabel(feat)
        ax.set_ylabel(TARGET)
        ax.set_title(f"{feat} vs {TARGET}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_residual(X_part, residuals, title_suffix, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(X_part["AT"], residuals, s=8, alpha=0.3)
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_xlabel("AT")
    axes[0].set_ylabel("Residual")
    axes[0].set_title(f"Residual vs AT — {title_suffix}")

    axes[1].axhline(0, color="red", linestyle="--")
    axes[1].set_title(f"Residual distribution — {title_suffix}")
    axes[1].hist(residuals, bins=50)
    axes[1].set_xlabel("Residual")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def degree_sweep(X_train, y_train, X_test, y_test, out_dir):
    rows = []
    for d in DEGREES:
        pipe = make_pipeline(d, alpha=1.0, use_ridge=True)
        pipe.fit(X_train, y_train)
        pred_tr = pipe.predict(X_train)
        pred_te = pipe.predict(X_test)
        rmse_tr = np.sqrt(mean_squared_error(y_train, pred_tr))
        rmse_te = np.sqrt(mean_squared_error(y_test, pred_te))
        r2_te = r2_score(y_test, pred_te)
        n_cols = pipe["poly"].n_output_features_
        rows.append({"Bac": d, "So_cot_sinh_ra": n_cols,
                     "RMSE_train": round(rmse_tr, 4), "RMSE_test": round(rmse_te, 4),
                     "R2_test": round(r2_te, 4)})
    table = pd.DataFrame(rows).set_index("Bac")
    table.to_csv(os.path.join(out_dir, "bang_bac_so_cot_rmse.csv"))
    return table


def pick_best_degree_elbow(X_train, y_train, out_dir):
    pipe = make_pipeline(degree=1, alpha=1.0, use_ridge=True)
    train_scores, val_scores = validation_curve(
        pipe, X_train, y_train, param_name="poly__degree", param_range=DEGREES,
        cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1,
    )
    train_rmse_mean = -train_scores.mean(axis=1)
    val_rmse_mean = -val_scores.mean(axis=1)
    val_rmse_std = val_scores.std(axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(DEGREES, train_rmse_mean, marker="o", label="RMSE train")
    plt.plot(DEGREES, val_rmse_mean, marker="o", label="RMSE validation (CV=5)")
    plt.fill_between(DEGREES, val_rmse_mean - val_rmse_std, val_rmse_mean + val_rmse_std, alpha=0.15)
    plt.xlabel("Bac da thuc (degree)")
    plt.ylabel("RMSE")
    plt.title("Duong cong xac thuc theo bac da thuc")
    plt.legend()
    plt.xticks(DEGREES)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "validation_curve.png"), dpi=150)
    plt.close()

    best_degree_argmin = DEGREES[int(np.argmin(val_rmse_mean))]
    best_degree = DEGREES[0]
    for i in range(1, len(DEGREES)):
        improvement = val_rmse_mean[i - 1] - val_rmse_mean[i]
        if improvement < IMPROVEMENT_THRESHOLD:
            best_degree = DEGREES[i - 1]
            break
    else:
        best_degree = DEGREES[-1]

    print("RMSE validation theo bac:", dict(zip(DEGREES, val_rmse_mean.round(4))))
    print(f"Bac argmin thuan: {best_degree_argmin} | Bac chon theo elbow: {best_degree}")
    return best_degree


def linear_vs_ridge_high_degree(X_train, y_train, X_test, y_test, out_dir):
    rows = []
    for d in [4, 5]:
        for use_ridge, name in [(False, "Linear"), (True, "Ridge")]:
            pipe = make_pipeline(d, alpha=1.0, use_ridge=use_ridge)
            pipe.fit(X_train, y_train)
            pred_te = pipe.predict(X_test)
            rmse_te = np.sqrt(mean_squared_error(y_test, pred_te))
            r2_te = r2_score(y_test, pred_te)
            max_abs_coef = np.max(np.abs(pipe["model"].coef_))
            rows.append({"Bac": d, "Model": name, "RMSE_test": round(rmse_te, 4),
                         "R2_test": round(r2_te, 4),
                         "He_so_lon_nhat": round(max_abs_coef, 2)})
    table = pd.DataFrame(rows).set_index(["Bac", "Model"])
    table.to_csv(os.path.join(out_dir, "linear_vs_ridge_bac_cao.csv"))
    return table


def main():
    parser = argparse.ArgumentParser(description="Train Polynomial+Ridge cho CCPP (PE)")
    parser.add_argument("--data", default="Folds5x2_pp.xlsx")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--models-dir", default="models")
    args = parser.parse_args()

    os.makedirs(args.reports_dir, exist_ok=True)
    os.makedirs(args.models_dir, exist_ok=True)

    df = load_data(args.data)
    print("Kich thuoc du lieu:", df.shape)
    print("Tuong quan AT-V:", round(df["AT"].corr(df["V"]), 3))

    plot_scatter(df, os.path.join(args.reports_dir, "scatter_AT_PE.png"))

    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # Bước 2-3: baseline bậc 1 + residual TRƯỚC
    baseline = Pipeline([("scale", StandardScaler()), ("model", LinearRegression())])
    baseline.fit(X_train, y_train)
    pred_test_baseline = baseline.predict(X_test)
    rmse_baseline = np.sqrt(mean_squared_error(y_test, pred_test_baseline))
    print(f"\nBaseline Linear bac 1: RMSE test = {rmse_baseline:.4f}")
    plot_residual(X_test, y_test.values - pred_test_baseline, "bac 1 (TRUOC)",
                  os.path.join(args.reports_dir, "residual_bac1_truoc.png"))

    # Bước 4, 6: quét bậc 1->5
    print("\n--- Quet bac 1->5 ---")
    degree_table = degree_sweep(X_train, y_train, X_test, y_test, args.reports_dir)
    print(degree_table)

    # Bước 5: validation curve + chọn bậc theo elbow
    print("\n--- Validation curve ---")
    best_degree = pick_best_degree_elbow(X_train, y_train, args.reports_dir)

    # Bước 7: Linear vs Ridge ở bậc cao
    print("\n--- Linear vs Ridge o bac cao ---")
    hd_table = linear_vs_ridge_high_degree(X_train, y_train, X_test, y_test, args.reports_dir)
    print(hd_table)

    # Bước 8: model bậc tối ưu + residual SAU
    best_pipe = make_pipeline(best_degree, alpha=1.0, use_ridge=True)
    best_pipe.fit(X_train, y_train)
    pred_test_best = best_pipe.predict(X_test)
    rmse_best = np.sqrt(mean_squared_error(y_test, pred_test_best))
    r2_best = r2_score(y_test, pred_test_best)
    print(f"\nModel bac toi uu (degree={best_degree}): RMSE test = {rmse_best:.4f}, R2 = {r2_best:.4f}")
    plot_residual(X_test, y_test.values - pred_test_best, f"bac {best_degree} (SAU)",
                  os.path.join(args.reports_dir, "residual_truoc_sau.png"))

    # Bước 9: so sánh Random Forest
    rf = RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    rmse_rf = np.sqrt(mean_squared_error(y_test, rf.predict(X_test)))
    print(f"Random Forest: RMSE test = {rmse_rf:.4f}")

    # Lưu model
    model_path = os.path.join(args.models_dir, "poly_pipeline.joblib")
    joblib.dump(best_pipe, model_path)
    print(f"\nDa luu model: {model_path}")

    print("\nHOAN TAT. Xem ket qua trong:", os.path.abspath(args.reports_dir),
          "va", os.path.abspath(args.models_dir))


if __name__ == "__main__":
    main()
