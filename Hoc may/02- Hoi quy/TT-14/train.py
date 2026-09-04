"""
TT-14 — ElasticNet: du bao tai suoi (Y1) va tai lam mat (Y2)
Ban script hoa cua notebook elasticnet_energy.ipynb — dung khi can chay
lai toan bo pipeline tu dong (vi du: tich hop CI, hoac chay tu terminal
thay vi Jupyter).

Cach chay:
    python src/train.py --data ENB2012_data.xlsx

Ket qua:
    reports/vif_table.csv
    reports/so_sanh_3_model_Y1.csv, _Y2.csv
    reports/heatmap_alpha_l1ratio.png
    reports/bootstrap_stability.csv
    models/elasticnet_Y1.joblib, elasticnet_Y2.joblib
"""
import argparse
import os

import joblib
import matplotlib
matplotlib.use("Agg")  # chay duoc tren server khong co man hinh
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import (
    ElasticNet, ElasticNetCV, Lasso, LassoCV, LinearRegression, Ridge, RidgeCV,
)
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

RANDOM_STATE = 42
NUMERIC_FEATURES = ["X1", "X2", "X3", "X4", "X5", "X7"]
CATEGORICAL_FEATURES = ["X6", "X8"]
GROUP_VARS = ["X1", "X2", "X4", "X5"]  # nhom bien dinh chat (da cong tuyen)


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Khong tim thay file du lieu tai: {os.path.abspath(path)}\n"
            "-> Truyen dung duong dan bang --data /duong/dan/ENB2012_data.xlsx"
        )
    return pd.read_excel(path)


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(drop="first"), CATEGORICAL_FEATURES),
        ]
    )


def get_feature_names(preprocessor: ColumnTransformer) -> list:
    cat_names = list(
        preprocessor.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES)
    )
    return NUMERIC_FEATURES + cat_names


def compute_vif(df: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    X_vif = df[NUMERIC_FEATURES].copy()
    X_vif = (X_vif - X_vif.mean()) / X_vif.std()
    X_vif = X_vif.assign(const=1)

    vif_data = pd.DataFrame()
    vif_data["feature"] = NUMERIC_FEATURES
    vif_data["VIF"] = [
        variance_inflation_factor(X_vif.values, i) for i in range(len(NUMERIC_FEATURES))
    ]
    vif_data = vif_data.sort_values("VIF", ascending=False).reset_index(drop=True)
    vif_data.to_csv(os.path.join(out_dir, "vif_table.csv"), index=False)
    return vif_data


def fit_regularized_models(preprocessor, X_train, y_train,
                            l1_ratios=(0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0)):
    alphas = np.logspace(-4, 1, 100)
    models = {
        "Ridge": Pipeline([("prep", preprocessor), ("model", RidgeCV(alphas=alphas, cv=5))]),
        "Lasso": Pipeline([("prep", preprocessor),
                            ("model", LassoCV(alphas=alphas, cv=5, max_iter=50000,
                                               random_state=RANDOM_STATE))]),
        "ElasticNet": Pipeline([("prep", preprocessor),
                                 ("model", ElasticNetCV(l1_ratio=list(l1_ratios), alphas=alphas,
                                                         cv=5, max_iter=50000,
                                                         random_state=RANDOM_STATE))]),
    }
    for pipe in models.values():
        pipe.fit(X_train, y_train)
    return models


def compare_models(models: dict, X_test, y_test) -> pd.DataFrame:
    rows = []
    for name, pipe in models.items():
        pred = pipe.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        r2 = r2_score(y_test, pred)
        coefs = pipe["model"].coef_
        n_nonzero = int(np.sum(np.abs(coefs) > 1e-6))
        rows.append({"Model": name, "RMSE": round(rmse, 4), "R2": round(r2, 4),
                      "So_bien_giu_lai": n_nonzero, "Tong_so_bien": len(coefs)})
    return pd.DataFrame(rows).set_index("Model")


def evaluate_baseline(preprocessor, X_train, y_train, X_test, y_test) -> pd.DataFrame:
    rows = {}
    for name, model in [("Dummy (mean)", DummyRegressor(strategy="mean")),
                         ("Linear Regression", LinearRegression())]:
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        r2 = r2_score(y_test, pred)
        rows[name] = {"RMSE": rmse, "R2": r2}
    return pd.DataFrame(rows).T


def plot_alpha_l1ratio_heatmap(preprocessor, X_train, y_train, out_path: str):
    alpha_grid = np.logspace(-3, 1, 15)
    l1_ratio_grid = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0])
    param_grid = {"model__alpha": alpha_grid, "model__l1_ratio": l1_ratio_grid}

    en_pipe = Pipeline([("prep", preprocessor),
                         ("model", ElasticNet(max_iter=50000, random_state=RANDOM_STATE))])
    grid = GridSearchCV(en_pipe, param_grid, cv=5,
                         scoring="neg_root_mean_squared_error", n_jobs=-1)
    grid.fit(X_train, y_train)

    results_df = pd.DataFrame(grid.cv_results_)
    rmse_pivot = results_df.pivot_table(
        index="param_model__l1_ratio", columns="param_model__alpha", values="mean_test_score",
    ) * -1

    plt.figure(figsize=(11, 5))
    sns.heatmap(rmse_pivot, cmap="viridis_r", xticklabels=[f"{a:.3f}" for a in alpha_grid])
    plt.title("CV RMSE theo luoi alpha x l1_ratio")
    plt.xlabel("alpha")
    plt.ylabel("l1_ratio")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return grid.best_params_, -grid.best_score_


def bootstrap_coefficients(preprocessor, X_train, y_train, best_alpha, best_l1_ratio,
                            n_boot=100, random_state=RANDOM_STATE) -> np.ndarray:
    """Bootstrap CHỈ trên tập train (co hoan lai) — khong dung ca X, y goc de tranh
    lam mat y nghia cua tap test da tach rieng truoc do."""
    rng = np.random.default_rng(random_state)
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    n = len(X_train)
    coef_list = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        pipe = Pipeline([("prep", preprocessor),
                          ("model", ElasticNet(alpha=best_alpha, l1_ratio=best_l1_ratio,
                                                max_iter=50000, random_state=random_state))])
        pipe.fit(X_train.iloc[idx], y_train.iloc[idx])
        coef_list.append(pipe["model"].coef_)
    return np.array(coef_list)


def run_for_target(df, target_col, out_dir_reports, out_dir_models):
    print(f"\n{'='*60}\nXU LY NHAN {target_col}\n{'='*60}")

    preprocessor = build_preprocessor()
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    baseline = evaluate_baseline(preprocessor, X_train, y_train, X_test, y_test)
    print("\n--- Baseline ---")
    print(baseline)

    models = fit_regularized_models(preprocessor, X_train, y_train)
    fitted_preprocessor = models["ElasticNet"]["prep"]
    feature_names = get_feature_names(fitted_preprocessor)

    print("\n--- Tham so toi uu ---")
    print("Ridge alpha      :", models["Ridge"]["model"].alpha_)
    print("Lasso alpha      :", models["Lasso"]["model"].alpha_)
    print("ElasticNet alpha :", models["ElasticNet"]["model"].alpha_,
          "| l1_ratio:", models["ElasticNet"]["model"].l1_ratio_)

    compare = compare_models(models, X_test, y_test)
    compare_path = os.path.join(out_dir_reports, f"so_sanh_3_model_{target_col}.csv")
    compare.to_csv(compare_path)
    print(f"\n--- So sanh 3 model ({target_col}) ---")
    print(compare)

    coef_table = pd.DataFrame(
        {name: pipe["model"].coef_ for name, pipe in models.items()}, index=feature_names
    )
    print(f"\n--- He so nhom bien dinh chat (hieu ung gom nhom) ({target_col}) ---")
    print(coef_table.loc[GROUP_VARS].round(4))

    model_path = os.path.join(out_dir_models, f"elasticnet_{target_col}.joblib")
    joblib.dump(models["ElasticNet"], model_path)
    print(f"\nDa luu model: {model_path}")

    return {
        "models": models,
        "compare": compare,
        "coef_table": coef_table,
        "X_train": X_train, "y_train": y_train,
        "X_test": X_test, "y_test": y_test,
        "preprocessor": fitted_preprocessor,
        "feature_names": feature_names,
    }


def main():
    parser = argparse.ArgumentParser(description="Train ElasticNet cho ENB2012 (Y1, Y2)")
    parser.add_argument("--data", default="ENB2012_data.xlsx",
                         help="Duong dan file ENB2012_data.xlsx")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--n-boot", type=int, default=100)
    args = parser.parse_args()

    os.makedirs(args.reports_dir, exist_ok=True)
    os.makedirs(args.models_dir, exist_ok=True)

    df = load_data(args.data)
    print("Kich thuoc du lieu:", df.shape)

    print("\n--- VIF (kiem tra da cong tuyen) ---")
    vif_table = compute_vif(df, args.reports_dir)
    print(vif_table)

    result_Y1 = run_for_target(df, "Y1", args.reports_dir, args.models_dir)
    result_Y2 = run_for_target(df, "Y2", args.reports_dir, args.models_dir)

    print("\n--- Heatmap alpha x l1_ratio (chi cho Y1, dung ElasticNet thuong + GridSearchCV) ---")
    heatmap_path = os.path.join(args.reports_dir, "heatmap_alpha_l1ratio.png")
    best_params, best_rmse = plot_alpha_l1ratio_heatmap(
        build_preprocessor(), result_Y1["X_train"], result_Y1["y_train"], heatmap_path,
    )
    print("Da luu:", heatmap_path, "| best params tren luoi:", best_params, "| RMSE:", best_rmse)

    print("\n--- Bootstrap on dinh he so (Y1, chi tren tap train) ---")
    en_Y1 = result_Y1["models"]["ElasticNet"]["model"]
    boot_coefs = bootstrap_coefficients(
        build_preprocessor(), result_Y1["X_train"], result_Y1["y_train"],
        en_Y1.alpha_, en_Y1.l1_ratio_, n_boot=args.n_boot,
    )
    boot_summary = pd.DataFrame({
        "feature": result_Y1["feature_names"],
        "mean_coef": boot_coefs.mean(axis=0),
        "std_coef": boot_coefs.std(axis=0),
    }).sort_values("std_coef", ascending=False)
    boot_summary.to_csv(os.path.join(args.reports_dir, "bootstrap_stability.csv"), index=False)
    print(boot_summary)

    print("\nHOAN TAT. Xem ket qua chi tiet trong thu muc:",
          os.path.abspath(args.reports_dir), "va", os.path.abspath(args.models_dir))


if __name__ == "__main__":
    main()
