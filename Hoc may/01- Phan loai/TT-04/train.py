"""
TT-04 — Logistic Regression: chẩn đoán nguy cơ bệnh tim.
Chạy: python src/train.py --data data/heart.csv
"""
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, precision_recall_curve, classification_report

RANDOM_STATE = 42
CATEGORICAL_COLS = ['cp', 'restecg', 'slope', 'thal', 'ca', 'sex', 'fbs', 'exang']
TARGET_COL = 'target'
TARGET_RECALL = 0.90


def load_data(path):
    df = pd.read_csv(path)
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Loại {n_before - len(df)} dòng trùng lặp ({n_before} -> {len(df)})")
    return df


def build_pipeline(numeric_cols):
    preprocess = ColumnTransformer([
        ('cat', OneHotEncoder(drop='first'), CATEGORICAL_COLS),
        ('num', StandardScaler(), numeric_cols),
    ])
    return Pipeline([
        ('preprocess', preprocess),
        ('model', LogisticRegressionCV(Cs=10, cv=5, penalty='l2',
                                        scoring='recall', max_iter=2000,
                                        random_state=RANDOM_STATE)),
    ])


def odds_ratio_table(pipeline):
    names = pipeline.named_steps['preprocess'].get_feature_names_out()
    coefs = pipeline.named_steps['model'].coef_[0]
    return pd.DataFrame({
        'dac_trung': names, 'he_so_w': coefs, 'odds_ratio': np.exp(coefs),
    }).sort_values('odds_ratio', ascending=False)


def choose_threshold(y_test, y_proba, target_recall=TARGET_RECALL):
    precision, recall, thr = precision_recall_curve(y_test, y_proba)
    valid = np.where(recall[:-1] >= target_recall)[0]
    idx = valid[-1]
    return thr[idx], precision[idx], recall[idx]


def main(data_path, out_path):
    df = load_data(data_path)
    numeric_cols = [c for c in df.columns if c not in CATEGORICAL_COLS + [TARGET_COL]]

    X, y = df.drop(columns=[TARGET_COL]), df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    pipeline = build_pipeline(numeric_cols)
    pipeline.fit(X_train, y_train)

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")

    threshold, precision, recall = choose_threshold(y_test, y_proba)
    print(f"Ngưỡng đạt recall >= {TARGET_RECALL}: {threshold:.3f} "
          f"(recall={recall:.3f}, precision={precision:.3f})")
    print(classification_report(y_test, (y_proba >= threshold).astype(int)))

    print("\nBảng odds ratio (top 5):")
    print(odds_ratio_table(pipeline).head())

    joblib.dump(pipeline, out_path)
    print(f"\nĐã lưu model tại {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/heart.csv')
    parser.add_argument('--out', default='models/logreg_pipeline.joblib')
    args = parser.parse_args()
    main(args.data, args.out)
