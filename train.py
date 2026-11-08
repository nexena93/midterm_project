import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = "/workspaces/midterm_project/wdbc.data"      
OUTPUT_FILE = "model.bin"
C = 1.0                          
THRESHOLD = 0.25                 
SEED = 1


COLUMNS = [
    "id", "diagnosis",
    "radius_mean", "texture_mean", "perimeter_mean",
    "area_mean", "smoothness_mean", "compactness_mean",
    "concavity_mean", "concave_points_mean", "symmetry_mean",
    "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se",
    "area_se", "smoothness_se", "compactness_se",
    "concavity_se", "concave_points_se", "symmetry_se",
    "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst",
    "area_worst", "smoothness_worst", "compactness_worst",
    "concavity_worst", "concave_points_worst", "symmetry_worst",
    "fractal_dimension_worst",
]


def load_data(path):
    df = pd.read_csv(path, header=None, names=COLUMNS)

    df["diagnosis"] = (df["diagnosis"] == "M").astype(int)

    features = [c for c in COLUMNS if c not in ("id", "diagnosis")]
    return df, features


def train(X, y):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, solver="lbfgs", max_iter=1000),
    )
    model.fit(X, y)
    return model


def main():
    df, features = load_data(DATA_PATH)

    df_full_train, df_test = train_test_split(
        df, test_size=0.2, random_state=SEED
    )

    model = train(df_full_train[features], df_full_train["diagnosis"].values)

    y_test = df_test["diagnosis"].values
    y_pred = model.predict_proba(df_test[features])[:, 1]
    print(f"Test ROC AUC: {roc_auc_score(y_test, y_pred):.3f}")
    print(f"Test F2:      {fbeta_score(y_test, y_pred >= THRESHOLD, beta=2):.3f}")

    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump({"model": model, "features": features, "threshold": THRESHOLD}, f)
    print(f"Model saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()