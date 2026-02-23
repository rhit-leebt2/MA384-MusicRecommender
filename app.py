from flask import Flask, render_template, request, jsonify
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Datasets", "dataset.csv")
TEMPLATE_DIR = os.path.join(BASE_DIR, "Data Exploration", "templates")

app = Flask(__name__, template_folder=TEMPLATE_DIR)


def load_and_prepare_data() -> pd.DataFrame:
    """Load the CSV and apply the same cleaning / feature engineering as in the notebook."""

    df = pd.read_csv(DATA_PATH)
    cols_to_drop = ["Unnamed: 0", "track_id", "mode"]
    cleaned_df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    cleaned_df = cleaned_df.dropna()

    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    Q1 = cleaned_df[numeric_cols].quantile(0.25)
    Q3 = cleaned_df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1

    df_clean = cleaned_df[
        ~(
            (cleaned_df[numeric_cols] < (Q1 - 1.5 * IQR))
            | (cleaned_df[numeric_cols] > (Q3 + 1.5 * IQR))
        ).any(axis=1)
    ]
    df_clean.reset_index(drop=True, inplace=True)

    if "track_name" in df_clean.columns and "artists" in df_clean.columns:
        df_clean = df_clean.drop_duplicates(subset=["track_name", "artists"]).reset_index(drop=True)

    if "explicit" in df_clean.columns:
        df_clean["explicit_int"] = df_clean["explicit"].astype(int)

    if "tempo" in df_clean.columns:
        tempo_bins = pd.cut(
            df_clean["tempo"],
            bins=[0, 90, 130, df_clean["tempo"].max()],
            labels=["slow", "medium", "fast"],
            include_lowest=True,
        )

        tempo_dummies = pd.get_dummies(tempo_bins, prefix="tempo")
        for col in ["tempo_slow", "tempo_medium", "tempo_fast"]:
            if col not in tempo_dummies.columns:
                tempo_dummies[col] = 0
        df_clean[["tempo_slow", "tempo_medium", "tempo_fast"]] = tempo_dummies[[
            "tempo_slow",
            "tempo_medium",
            "tempo_fast",
        ]]

    return df_clean


df_clean = load_and_prepare_data()

# Feature definitions match the final notebook
selected_feature_cols = [
    "danceability",
    "energy",
    "loudness",
    "acousticness",
    "instrumentalness",
    "valence",
    "speechiness",
    "liveness",
    "tempo_slow",
    "tempo_medium",
    "tempo_fast",
    "duration_ms",
    "popularity",
]

genre_cols = [c for c in df_clean.columns if "genre" in c.lower()]
if not genre_cols:
    raise RuntimeError("No genre column found in dataset for training.")
target_genre_col = genre_cols[0]


# Train/test split exactly as in the notebook
train_df, test_df = train_test_split(df_clean, test_size=0.2, random_state=42)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)


def _prepare_scaled_matrices(df_train: pd.DataFrame, df_eval: pd.DataFrame, feature_cols):
    """Helper that mirrors prepare_scaled_matrices from the notebook."""

    df_train_feat = df_train.dropna(subset=feature_cols).reset_index(drop=True)
    df_eval_feat = df_eval.dropna(subset=feature_cols).reset_index(drop=True)

    X_train = df_train_feat[feature_cols].values
    X_eval = df_eval_feat[feature_cols].values

    scaler_local = StandardScaler()
    X_train_scaled = scaler_local.fit_transform(X_train)
    X_eval_scaled = scaler_local.transform(X_eval)

    y_train = df_train_feat[target_genre_col].astype("category")
    y_eval = df_eval_feat[target_genre_col].astype("category")

    return X_train_scaled, X_eval_scaled, y_train, y_eval


# Tune K for KNN using the same search as the notebook
ks_to_try = [3, 5, 10, 20, 50, 100]
results_by_k = {}
for k in ks_to_try:
    X_train_scaled_k, X_eval_scaled_k, y_train_k, y_eval_k = _prepare_scaled_matrices(
        train_df, test_df, selected_feature_cols
    )
    knn_tmp = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    knn_tmp.fit(X_train_scaled_k, y_train_k)
    eval_pred = knn_tmp.predict(X_eval_scaled_k)
    eval_acc = float(np.mean(eval_pred == y_eval_k))
    results_by_k[k] = eval_acc

best_k = max(results_by_k, key=lambda kk: results_by_k[kk])


# Fit a single global scaler on the training set for live predictions
train_feat = train_df.dropna(subset=selected_feature_cols).reset_index(drop=True)
X_train_live = train_feat[selected_feature_cols].values
y_train_live = train_feat[target_genre_col].astype("category")

scaler = StandardScaler()
X_train_scaled_live = scaler.fit_transform(X_train_live)

# Final models with the same hyperparameters as in the notebook
knn_model = KNeighborsClassifier(n_neighbors=best_k, metric="euclidean")
knn_model.fit(X_train_scaled_live, y_train_live)

log_reg_model = LogisticRegression(max_iter=1000, multi_class="multinomial")
log_reg_model.fit(X_train_scaled_live, y_train_live)

dt_model = DecisionTreeClassifier(max_depth=20, random_state=42)
dt_model.fit(X_train_scaled_live, y_train_live)

rf_model = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled_live, y_train_live)


# Features exposed as sliders in the UI
slider_features = [
    "danceability",
    "energy",
    "loudness",
    "acousticness",
    "instrumentalness",
    "valence",
    "speechiness",
    "liveness",
    "tempo",
    "duration_ms",
    "popularity",
]

feature_steps = {
    "danceability": 0.01,
    "energy": 0.01,
    "acousticness": 0.01,
    "instrumentalness": 0.01,
    "valence": 0.01,
    "speechiness": 0.01,
    "liveness": 0.01,
    "loudness": 0.1,
    "tempo": 1.0,
    "duration_ms": 1000.0,
    "popularity": 1.0,
}

feature_config = {}
for feat in slider_features:
    col_name = "tempo" if feat == "tempo" else feat
    if col_name not in df_clean.columns:
        continue
    col_min = float(df_clean[col_name].min())
    col_max = float(df_clean[col_name].max())
    col_mean = float(df_clean[col_name].mean())
    step = feature_steps.get(feat, (col_max - col_min) / 100.0 if col_max > col_min else 0.01)
    feature_config[feat] = {
        "min": col_min,
        "max": col_max,
        "mean": col_mean,
        "step": step,
    }


def build_feature_vector(input_values: dict) -> np.ndarray:
    """Create a scaled feature vector in the same space the models were trained on."""

    feat_values = {}
    for col in selected_feature_cols:
        if col in {"tempo_slow", "tempo_medium", "tempo_fast"}:
            continue
        if col not in feature_config:
            continue
        raw_val = input_values.get(col, feature_config[col]["mean"])
        feat_values[col] = float(raw_val)

    if "tempo" in feature_config:
        tempo_val = float(input_values.get("tempo", feature_config["tempo"]["mean"]))
    else:
        tempo_val = 120.0

    slow = 1.0 if tempo_val <= 90 else 0.0
    medium = 1.0 if 90 < tempo_val <= 130 else 0.0
    fast = 1.0 if tempo_val > 130 else 0.0

    feat_values["tempo_slow"] = slow
    feat_values["tempo_medium"] = medium
    feat_values["tempo_fast"] = fast

    ordered = [feat_values[col] for col in selected_feature_cols]
    X_new = np.array(ordered, dtype=float).reshape(1, -1)
    X_new_scaled = scaler.transform(X_new)
    return X_new_scaled


@app.route("/", methods=["GET"])
def index():
    # Initial slider positions use the training means
    input_values = {feat: cfg["mean"] for feat, cfg in feature_config.items()}
    return render_template(
        "index.html",
        feature_config=feature_config,
        input_values=input_values,
        target_genre_col=target_genre_col,
    )


@app.route("/predict", methods=["POST"])
def predict():
    """Return live predictions for the current slider values as JSON."""

    data = request.get_json(silent=True) or {}
    # Fill missing values with means
    input_values = {}
    for feat, cfg in feature_config.items():
        try:
            val = float(data.get(feat, cfg["mean"]))
        except (TypeError, ValueError):
            val = cfg["mean"]
        input_values[feat] = val

    X_new_scaled = build_feature_vector(input_values)
    preds = {
        f"KNN (k={best_k})": str(knn_model.predict(X_new_scaled)[0]),
        "Logistic Regression": str(log_reg_model.predict(X_new_scaled)[0]),
        "Decision Tree": str(dt_model.predict(X_new_scaled)[0]),
        "Random Forest": str(rf_model.predict(X_new_scaled)[0]),
    }
    return jsonify({"predictions": preds})


if __name__ == "__main__":
    app.run(debug=True)
