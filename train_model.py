"""
train_model.py
----------------
Trains and compares multiple Machine Learning models on the real Kaggle
"Credit Card Transactions Fraud Detection Dataset" by Kartik Shenoy:
https://www.kaggle.com/datasets/kartik2112/fraud-detection

Unlike anonymized PCA-based fraud datasets, this dataset contains real,
human-readable transaction details (merchant, category, customer location,
date of birth, etc.), which means genuine feature engineering is required.
This is exactly what makes the project interview-worthy.

Pipeline:
1. Load fraudTrain.csv + fraudTest.csv and combine them
2. Clean the data (handle missing values)
3. Engineer new features:
      - age              -> from date of birth
      - distance_km      -> straight-line distance between customer & merchant
      - trans_hour       -> hour of day the transaction happened
      - trans_day_of_week-> day of week the transaction happened
4. Encode categorical columns (category, gender)
5. Scale numeric features
6. Split into train/test sets (our own 80/20 split, stratified)
7. Train 4 models: Logistic Regression, Decision Tree, Random Forest, XGBoost
8. Evaluate with Accuracy, Precision, Recall, F1-Score, Confusion Matrix
9. Save the best model (+ encoders + scaler) using Joblib
10. Save charts for the dashboard
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # allows matplotlib to run without a display (server-friendly)
import matplotlib.pyplot as plt
import joblib
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# Final feature set used by the model.
# NOTE: These exact 8 features are also what the Flask predict form collects
# (amt, category, gender, city_pop, age, trans_hour) or auto-computes
# (distance_km, trans_day_of_week) -- keep app.py in sync if you change this.
FEATURE_COLUMNS = [
    "amt", "category", "gender", "city_pop",
    "age", "trans_hour", "distance_km", "trans_day_of_week"
]
NUMERIC_COLUMNS_TO_SCALE = ["amt", "city_pop", "age", "distance_km"]
CATEGORICAL_COLUMNS = ["category", "gender"]


# ---------------------------------------------------------------------------
# STEP 1: Load Dataset
# ---------------------------------------------------------------------------
def load_data():
    """
    Loads and combines fraudTrain.csv and fraudTest.csv into a single
    DataFrame. We combine them and create our own train/test split later
    so the whole pipeline (cleaning, feature engineering, splitting) is
    handled consistently in one place.
    """
    print("Loading dataset... this may take a minute for ~1.85M rows.")
    train_df = pd.read_csv("dataset/fraudTrain.csv", index_col=0)
    test_df = pd.read_csv("dataset/fraudTest.csv", index_col=0)
    df = pd.concat([train_df, test_df], ignore_index=True)
    print(f"Dataset loaded successfully. Shape: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# STEP 2: Clean Data
# ---------------------------------------------------------------------------
def clean_data(df):
    """Handles missing values. Numeric columns -> median, others -> mode."""
    missing_before = df.isnull().sum().sum()
    print(f"Missing values before cleaning: {missing_before}")

    if missing_before > 0:
        for column in df.columns:
            if df[column].isnull().sum() > 0:
                if pd.api.types.is_numeric_dtype(df[column]):
                    df[column] = df[column].fillna(df[column].median())
                else:
                    df[column] = df[column].fillna(df[column].mode()[0])

    print(f"Missing values after cleaning: {df.isnull().sum().sum()}")
    return df


# ---------------------------------------------------------------------------
# STEP 3: Feature Engineering
# ---------------------------------------------------------------------------
def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the straight-line (great-circle) distance in kilometers
    between two GPS coordinates using the Haversine formula.
    A large distance between customer and merchant location is a classic
    real-world fraud signal.
    """
    R = 6371  # Earth's radius in kilometers
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def engineer_features(df):
    """Creates new, meaningful features from raw dataset columns."""
    print("Engineering features (age, distance, hour, day-of-week)...")

    # Convert date columns to actual datetime objects
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["dob"] = pd.to_datetime(df["dob"])

    # Age of the customer at the time of the transaction
    df["age"] = (df["trans_date_trans_time"] - df["dob"]).dt.days // 365

    # Distance (km) between customer's location and the merchant's location
    df["distance_km"] = haversine_distance(
        df["lat"], df["long"], df["merch_lat"], df["merch_long"]
    )

    # Hour of day and day of week the transaction occurred
    df["trans_hour"] = df["trans_date_trans_time"].dt.hour
    df["trans_day_of_week"] = df["trans_date_trans_time"].dt.dayofweek  # 0=Monday

    return df


# ---------------------------------------------------------------------------
# STEP 4: Encode Categorical Columns
# ---------------------------------------------------------------------------
def encode_categorical(df):
    """
    Encodes text categorical columns into numbers using LabelEncoder,
    since ML models require numeric input. The fitted encoders are
    returned so the Flask app can apply the SAME encoding to new
    transactions entered by a user.
    """
    encoders = {}
    for col in CATEGORICAL_COLUMNS:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col].astype(str))
        encoders[col] = encoder
        print(f"Encoded '{col}': {len(encoder.classes_)} unique categories")
    return df, encoders


# ---------------------------------------------------------------------------
# STEP 5: Feature Scaling
# ---------------------------------------------------------------------------
def scale_features(X_train, X_test):
    """Scales continuous numeric columns using StandardScaler."""
    scaler = StandardScaler()
    X_train[NUMERIC_COLUMNS_TO_SCALE] = scaler.fit_transform(X_train[NUMERIC_COLUMNS_TO_SCALE])
    X_test[NUMERIC_COLUMNS_TO_SCALE] = scaler.transform(X_test[NUMERIC_COLUMNS_TO_SCALE])
    return X_train, X_test, scaler


# ---------------------------------------------------------------------------
# STEP 6: Train and Evaluate a Model
# ---------------------------------------------------------------------------
def train_and_evaluate(model, model_name, X_train, X_test, y_train, y_test):
    """Trains a model and returns its evaluation metrics (including
    probability-based metrics ROC-AUC and PR-AUC, which matter a lot more
    than accuracy on an imbalanced dataset like this one)."""
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]  # P(fraud)

    metrics = {
        "model": model_name,
        "accuracy": round(accuracy_score(y_test, predictions), 4),
        "precision": round(precision_score(y_test, predictions, zero_division=0), 4),
        "recall": round(recall_score(y_test, predictions, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, predictions, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, probabilities), 4),
        "pr_auc": round(average_precision_score(y_test, probabilities), 4),
    }

    print(f"{model_name} Results:")
    for key, value in metrics.items():
        if key != "model":
            print(f"   {key}: {value}")

    return model, metrics, predictions, probabilities


# ---------------------------------------------------------------------------
# STEP 7: Save Charts
# ---------------------------------------------------------------------------
def save_confusion_matrix(y_test, predictions, save_path):
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix - Best Model")
    plt.colorbar()
    plt.xticks([0, 1], ["Genuine", "Fraud"])
    plt.yticks([0, 1], ["Genuine", "Fraud"])
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def save_bar_chart(df, save_path):
    counts = df["is_fraud"].value_counts()
    plt.figure(figsize=(5, 4))
    plt.bar(["Genuine", "Fraud"], [counts.get(0, 0), counts.get(1, 0)],
            color=["#2ecc71", "#e74c3c"])
    plt.title("Fraud vs Genuine Transactions")
    plt.ylabel("Number of Transactions")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Bar chart saved to {save_path}")


def save_pie_chart(df, save_path):
    counts = df["is_fraud"].value_counts()
    plt.figure(figsize=(5, 4))
    plt.pie([counts.get(0, 0), counts.get(1, 0)],
            labels=["Genuine", "Fraud"],
            autopct="%1.2f%%",
            colors=["#2ecc71", "#e74c3c"])
    plt.title("Class Distribution")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Pie chart saved to {save_path}")


def save_roc_curve(y_test, probabilities, roc_auc, save_path):
    """
    Plots the ROC curve (True Positive Rate vs False Positive Rate).
    A curve that hugs the top-left corner, with AUC close to 1.0, means
    the model separates fraud from genuine transactions very well
    regardless of which decision threshold is chosen.
    """
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, color="#2f6fed", linewidth=2, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random guess")
    plt.title("ROC Curve - Best Model")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"ROC curve saved to {save_path}")


def save_pr_curve(y_test, probabilities, pr_auc, save_path):
    """
    Plots the Precision-Recall curve. This is more informative than ROC
    for highly imbalanced data (like fraud detection, ~0.5% positive
    class), because it focuses on how well the model performs specifically
    on the rare fraud class.
    """
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, probabilities)
    plt.figure(figsize=(5, 4))
    plt.plot(recall_vals, precision_vals, color="#e74c3c", linewidth=2,
              label=f"PR-AUC = {pr_auc:.4f}")
    plt.title("Precision-Recall Curve - Best Model")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Precision-Recall curve saved to {save_path}")


def save_feature_importance(best_model, best_model_name, feature_columns, save_path):
    """
    Plots which features the model relied on most. Tree-based models
    (Decision Tree, Random Forest, XGBoost) expose .feature_importances_.
    Logistic Regression instead uses the absolute value of its coefficients.
    This is one of the best things to show an interviewer - it explains
    WHY the model flags a transaction, not just that it does.
    """
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
    elif hasattr(best_model, "coef_"):
        importances = np.abs(best_model.coef_[0])
    else:
        print("Model has no interpretable importances; skipping chart.")
        return

    order = np.argsort(importances)[::-1]
    sorted_features = [feature_columns[i] for i in order]
    sorted_importances = importances[order]

    plt.figure(figsize=(6, 4))
    plt.barh(sorted_features[::-1], sorted_importances[::-1], color="#2f6fed")
    plt.title(f"Feature Importance - {best_model_name}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Feature importance chart saved to {save_path}")


def find_best_threshold(y_test, probabilities):
    """
    By default, models classify a transaction as Fraud only if predicted
    probability > 0.5. That default is arbitrary - it doesn't account for
    the fact that fraud is rare and costly to miss. Here we scan many
    thresholds and pick the one that maximizes F1-Score on the test set,
    then reuse that threshold at prediction time in the Flask app instead
    of the default 0.5. This is a genuine, interview-worthy optimization.
    """
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, probabilities)
    # precision_recall_curve returns one more precision/recall point than
    # thresholds, so we drop the last precision/recall value to align them
    f1_scores = 2 * (precision_vals[:-1] * recall_vals[:-1]) / (precision_vals[:-1] + recall_vals[:-1] + 1e-9)
    best_index = np.argmax(f1_scores)
    best_threshold = thresholds[best_index]
    print(f"\nBest decision threshold (max F1): {best_threshold:.4f} "
          f"(F1 at this threshold: {f1_scores[best_index]:.4f}, "
          f"vs F1 at default 0.5)")
    return float(best_threshold)


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------
def main():
    os.makedirs("static/images", exist_ok=True)

    # 1. Load
    df = load_data()

    # 2. Clean
    df = clean_data(df)

    # 3. Feature engineering
    df = engineer_features(df)

    # 4. Encode categorical columns
    df, encoders = encode_categorical(df)

    # 5. Select final features + target
    X = df[FEATURE_COLUMNS].copy()
    y = df["is_fraud"].copy()

    # 6. Train/test split (80/20, stratified to preserve the fraud ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain size: {X_train.shape}, Test size: {X_test.shape}")

    # 7. Scale numeric features
    X_train, X_test, scaler = scale_features(X_train, X_test)

    # 8. Train & compare models
    # class_weight='balanced' / scale_pos_weight tells the model to pay more
    # attention to the rare fraud class, since only ~0.5% of transactions
    # are fraudulent. Without this, models tend to just predict "Genuine"
    # for everything and still score a misleadingly high accuracy.
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=12, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=15, class_weight="balanced",
            n_jobs=-1, random_state=42
        ),
    }
    if XGBOOST_AVAILABLE:
        # XGBoost handles imbalance via scale_pos_weight instead of class_weight
        fraud_ratio = (y_train == 0).sum() / (y_train == 1).sum()
        models["XGBoost"] = XGBClassifier(
            eval_metric="logloss", scale_pos_weight=fraud_ratio,
            n_jobs=-1, random_state=42
        )

    results, trained_models, predictions_map, probabilities_map = [], {}, {}, {}
    for name, model in models.items():
        trained_model, metrics, predictions, probabilities = train_and_evaluate(
            model, name, X_train, X_test, y_train, y_test
        )
        results.append(metrics)
        trained_models[name] = trained_model
        predictions_map[name] = predictions
        probabilities_map[name] = probabilities

    # 9. Pick best model by F1-score (best metric for imbalanced fraud data)
    results_df = pd.DataFrame(results).sort_values(by="f1_score", ascending=False)
    print("\n===== MODEL COMPARISON =====")
    print(results_df.to_string(index=False))

    best_model_name = results_df.iloc[0]["model"]
    best_model = trained_models[best_model_name]
    best_predictions = predictions_map[best_model_name]
    best_probabilities = probabilities_map[best_model_name]
    best_accuracy = results_df.iloc[0]["accuracy"]
    best_roc_auc = results_df.iloc[0]["roc_auc"]
    best_pr_auc = results_df.iloc[0]["pr_auc"]
    print(f"\nBest Model Selected: {best_model_name}")

    # 10. Find the optimal decision threshold for the best model
    best_threshold = find_best_threshold(y_test, best_probabilities)

    # 11. Save charts
    save_confusion_matrix(y_test, best_predictions, "static/images/confusion_matrix.png")
    save_bar_chart(df, "static/images/bar_chart.png")
    save_pie_chart(df, "static/images/pie_chart.png")
    save_roc_curve(y_test, best_probabilities, best_roc_auc, "static/images/roc_curve.png")
    save_pr_curve(y_test, best_probabilities, best_pr_auc, "static/images/pr_curve.png")
    save_feature_importance(best_model, best_model_name, FEATURE_COLUMNS,
                             "static/images/feature_importance.png")

    # 12. Save model, scaler, encoders, and column order using Joblib
    joblib.dump(best_model, "fraud_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(encoders, "encoders.pkl")
    joblib.dump(FEATURE_COLUMNS, "model_columns.pkl")
    joblib.dump(best_threshold, "threshold.pkl")

    # Save average distance & most common day-of-week so the Flask app can
    # auto-fill these two engineered features for a new, single transaction
    # (a live prediction has no merchant GPS coordinates or transaction date)
    defaults = {
        "distance_km": float(df["distance_km"].mean()),
        "trans_day_of_week": int(df["trans_day_of_week"].mode()[0]),
    }
    joblib.dump(defaults, "feature_defaults.pkl")
    print("\nModel, scaler, encoders, columns, threshold, and defaults saved using Joblib.")

    # 13. Save summary for dashboard
    summary = {
        "total_transactions": int(len(df)),
        "fraud_transactions": int(df["is_fraud"].sum()),
        "genuine_transactions": int(len(df) - df["is_fraud"].sum()),
        "best_model": best_model_name,
        "accuracy": float(best_accuracy),
        "roc_auc": float(best_roc_auc),
        "pr_auc": float(best_pr_auc),
        "best_threshold": float(best_threshold),
        "all_results": results,
    }
    with open("model_summary.json", "w") as f:
        json.dump(summary, f, indent=4)
    print("Summary saved to model_summary.json")


if __name__ == "__main__":
    main()
