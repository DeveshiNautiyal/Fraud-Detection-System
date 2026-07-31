"""
app.py
-------
Main Flask application for the Financial Fraud Detection System.

Routes:
    /                    -> Home page (project introduction)
    /predict             -> Form to enter ONE transaction (GET) + prediction (POST)
    /batch-predict       -> Upload a CSV of MANY transactions (GET) + batch prediction (POST)
    /download-sample-csv -> Downloads a sample CSV showing the expected batch format
    /download-batch/<f>  -> Downloads the results of a previous batch prediction
    /dashboard           -> Live dashboard with model stats, charts, and recent predictions
    /api/live-stats      -> JSON endpoint the dashboard polls to auto-refresh live counts
"""

from flask import Flask, render_template, request, jsonify, send_file, abort
import joblib
import pandas as pd
import json
import os
import io
import uuid

import database  # our custom SQLite module

app = Flask(__name__)

BATCH_RESULTS_FOLDER = "batch_results"
os.makedirs(BATCH_RESULTS_FOLDER, exist_ok=True)

# ---------------------------------------------------------------------------
# Load the trained model, scaler, encoders, and column order ONCE at startup.
# Loading these inside every request would be slow.
# ---------------------------------------------------------------------------
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")            # LabelEncoders for category & gender
model_columns = joblib.load("model_columns.pkl")  # exact feature order used in training
feature_defaults = joblib.load("feature_defaults.pkl")  # auto-filled distance_km / day-of-week
decision_threshold = joblib.load("threshold.pkl")  # tuned cutoff (not the default 0.5)

NUMERIC_COLUMNS_TO_SCALE = ["amt", "city_pop", "age", "distance_km"]
SUMMARY_PATH = "model_summary.json"

# Initialize the SQLite database (creates the table if it doesn't exist)
database.init_db()


def load_model_summary():
    """Loads the training summary (accuracy, counts, etc.) saved by train_model.py"""
    if os.path.exists(SUMMARY_PATH):
        with open(SUMMARY_PATH, "r") as f:
            return json.load(f)
    return {}


def safe_encode(encoder, value):
    """
    Encodes a category/gender value using the SAME encoder fitted during
    training. If a brand-new value shows up that the encoder has never
    seen, we safely fall back to its first known class instead of crashing
    - this matters a lot more for batch CSV uploads, where a typo or an
    unfamiliar category in someone's file shouldn't break the whole batch.
    """
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    return 0


def predict_dataframe(df):
    """
    Runs the full prediction pipeline (encode -> auto-fill engineered
    features -> scale -> predict with tuned threshold) on ANY DataFrame
    that has the columns: amount, category, gender, city_pop, age, trans_hour.

    This single function is used by BOTH the one-transaction Predict page
    and the batch CSV upload, so there's only one place the prediction
    logic can go wrong, and single vs. batch predictions can never
    silently disagree with each other.

    Returns the original DataFrame with two new columns added:
    'prediction' (Fraud/Genuine) and 'confidence' (0-100).
    """
    working = df.copy()

    # Encode category & gender using the training-time encoders
    working["category_encoded"] = working["category"].apply(
        lambda v: safe_encode(encoders["category"], v)
    )
    working["gender_encoded"] = working["gender"].apply(
        lambda v: safe_encode(encoders["gender"], v)
    )

    # Auto-fill the two engineered features a simple form/CSV can't provide
    working["distance_km"] = feature_defaults["distance_km"]
    working["trans_day_of_week"] = feature_defaults["trans_day_of_week"]

    # Build the model input in the exact column order used during training
    model_input = pd.DataFrame({
        "amt": working["amount"].astype(float),
        "category": working["category_encoded"],
        "gender": working["gender_encoded"],
        "city_pop": working["city_pop"].astype(int),
        "age": working["age"].astype(int),
        "trans_hour": working["trans_hour"].astype(int),
        "distance_km": working["distance_km"],
        "trans_day_of_week": working["trans_day_of_week"],
    })[model_columns]

    # Scale numeric columns using the SAME scaler fitted during training
    model_input[NUMERIC_COLUMNS_TO_SCALE] = scaler.transform(model_input[NUMERIC_COLUMNS_TO_SCALE])

    # Predict using the tuned threshold instead of the default 0.5 cutoff
    fraud_probabilities = model.predict_proba(model_input)[:, 1]
    predictions = ["Fraud" if p >= decision_threshold else "Genuine" for p in fraud_probabilities]
    confidences = [
        round(float(p) * 100, 2) if pred == "Fraud" else round((1 - float(p)) * 100, 2)
        for p, pred in zip(fraud_probabilities, predictions)
    ]

    df = df.copy()
    df["prediction"] = predictions
    df["confidence"] = confidences
    return df


# ---------------------------------------------------------------------------
# ROUTE: Home Page
# ---------------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


# ---------------------------------------------------------------------------
# ROUTE: Predict Page (GET shows form, POST returns result page)
# ---------------------------------------------------------------------------
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        # ---- Step 1: Read the human-friendly form inputs into a 1-row DataFrame ----
        single_row = pd.DataFrame([{
            "amount": float(request.form.get("amount", 0)),
            "category": request.form.get("category"),
            "gender": request.form.get("gender"),
            "city_pop": int(request.form.get("city_pop", 0)),
            "age": int(request.form.get("age", 0)),
            "trans_hour": int(request.form.get("trans_hour", 12)),
        }])

        # ---- Step 2: Run the SAME prediction pipeline used by batch upload ----
        result_df = predict_dataframe(single_row)
        result = result_df.iloc[0]
        amount = float(result["amount"])
        category = str(result["category"])
        gender = str(result["gender"])
        city_pop = int(result["city_pop"])
        age = int(result["age"])
        trans_hour = int(result["trans_hour"])
        prediction_result = str(result["prediction"])
        confidence = float(result["confidence"])

        # ---- Step 3: Save this prediction into the SQLite database ----
        database.insert_transaction(
            amount, category, gender, city_pop, age, trans_hour,
            prediction_result, confidence
        )

        # ---- Step 4: Show the result page ----
        return render_template(
            "result.html",
            prediction_result=prediction_result,
            confidence=confidence,
            amount=amount,
            category=category,
            gender=gender,
            city_pop=city_pop,
            age=age,
            trans_hour=trans_hour,
        )

    # GET request -> show the empty form. Category dropdown options come
    # straight from the encoder, so the form always matches what the model
    # was actually trained on.
    categories = list(encoders["category"].classes_)
    genders = list(encoders["gender"].classes_)
    return render_template("predict.html", categories=categories, genders=genders)


# ---------------------------------------------------------------------------
# ROUTE: Batch Predict Page (GET shows upload form, POST processes a CSV)
# ---------------------------------------------------------------------------
@app.route("/batch-predict", methods=["GET", "POST"])
def batch_predict():
    if request.method == "POST":
        uploaded_file = request.files.get("csv_file")

        if not uploaded_file or uploaded_file.filename == "":
            return render_template("batch_predict.html", error="Please choose a CSV file to upload.")

        if not uploaded_file.filename.lower().endswith(".csv"):
            return render_template("batch_predict.html", error="Only .csv files are supported.")

        # ---- Step 1: Read the uploaded CSV ----
        try:
            batch_df = pd.read_csv(uploaded_file)
        except Exception:
            return render_template("batch_predict.html", error="Could not read that file as a CSV.")

        # ---- Step 2: Validate it has the columns we need ----
        required_columns = {"amount", "category", "gender", "city_pop", "age", "trans_hour"}
        missing = required_columns - set(batch_df.columns)
        if missing:
            return render_template(
                "batch_predict.html",
                error=f"Your CSV is missing required column(s): {', '.join(sorted(missing))}. "
                      f"Download the sample template below to see the expected format."
            )

        # Cap batch size for a smooth demo experience (avoids an accidental
        # multi-million-row upload freezing the app in a classroom/demo setting)
        MAX_ROWS = 5000
        if len(batch_df) > MAX_ROWS:
            batch_df = batch_df.head(MAX_ROWS)

        # ---- Step 3: Run the SAME prediction pipeline as the single-transaction form ----
        result_df = predict_dataframe(batch_df)

        # ---- Step 4: Log every prediction to SQLite in one bulk insert ----
        rows_to_insert = [
            (
                float(row["amount"]), str(row["category"]), str(row["gender"]),
                int(row["city_pop"]), int(row["age"]), int(row["trans_hour"]),
                str(row["prediction"]), float(row["confidence"])
            )
            for _, row in result_df.iterrows()
        ]
        database.insert_transactions_bulk(rows_to_insert)

        # ---- Step 5: Save results to a downloadable CSV ----
        result_filename = f"batch_results_{uuid.uuid4().hex[:8]}.csv"
        result_path = os.path.join(BATCH_RESULTS_FOLDER, result_filename)
        result_df.to_csv(result_path, index=False)

        # ---- Step 6: Show a summary + preview table ----
        fraud_count = int((result_df["prediction"] == "Fraud").sum())
        genuine_count = len(result_df) - fraud_count

        return render_template(
            "batch_predict.html",
            results=result_df.head(50).to_dict(orient="records"),  # preview first 50 rows
            total_rows=len(result_df),
            fraud_count=fraud_count,
            genuine_count=genuine_count,
            result_filename=result_filename,
        )

    return render_template("batch_predict.html")


# ---------------------------------------------------------------------------
# ROUTE: Download a sample CSV template showing the expected batch format
# ---------------------------------------------------------------------------
@app.route("/download-sample-csv")
def download_sample_csv():
    sample = pd.DataFrame([
        {"amount": 999.50, "category": "shopping_net", "gender": "F", "city_pop": 50000, "age": 35, "trans_hour": 2},
        {"amount": 45.00, "category": "grocery_pos", "gender": "M", "city_pop": 80000, "age": 42, "trans_hour": 14},
        {"amount": 1500.00, "category": "misc_net", "gender": "F", "city_pop": 12000, "age": 29, "trans_hour": 3},
    ])
    buffer = io.BytesIO()
    sample.to_csv(buffer, index=False)
    buffer.seek(0)
    return send_file(buffer, mimetype="text/csv", as_attachment=True,
                      download_name="sample_transactions.csv")


# ---------------------------------------------------------------------------
# ROUTE: Download the results of a previous batch prediction
# ---------------------------------------------------------------------------
@app.route("/download-batch/<filename>")
def download_batch(filename):
    # Only allow downloading files we generated ourselves (prevents path traversal)
    safe_filename = os.path.basename(filename)
    file_path = os.path.join(BATCH_RESULTS_FOLDER, safe_filename)
    if not os.path.exists(file_path):
        abort(404)
    return send_file(file_path, mimetype="text/csv", as_attachment=True,
                      download_name=safe_filename)


# ---------------------------------------------------------------------------
# ROUTE: Dashboard Page
# ---------------------------------------------------------------------------
@app.route("/dashboard")
def dashboard():
    summary = load_model_summary()
    live_counts = database.get_summary_counts()
    recent_transactions = database.get_all_transactions(limit=10)

    return render_template(
        "dashboard.html",
        summary=summary,
        live_counts=live_counts,
        recent_transactions=recent_transactions
    )


# ---------------------------------------------------------------------------
# API ROUTE: Live stats (polled by dashboard.html via JavaScript every few
# seconds so the "live" numbers and recent-predictions table update without
# a full page reload)
# ---------------------------------------------------------------------------
@app.route("/api/live-stats")
def live_stats():
    live_counts = database.get_summary_counts()
    recent = database.get_all_transactions(limit=10)

    recent_list = [
        {
            "id": row[0], "amount": row[1], "category": row[2], "gender": row[3],
            "city_pop": row[4], "age": row[5], "trans_hour": row[6],
            "prediction": row[7], "confidence": row[8], "created_at": row[9],
        }
        for row in recent
    ]

    return jsonify({"live_counts": live_counts, "recent_transactions": recent_list})


if __name__ == "__main__":
    # debug=True is fine for local development / demos. Turn this off in production.
    app.run(debug=True)
