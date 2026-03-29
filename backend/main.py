from fastapi import FastAPI, UploadFile, File, Form
import pandas as pd
import os
import mysql.connector

from backend.services.cleaning import clean_data
from backend.services.model import run_classification, run_regression, run_clustering

app = FastAPI()


# ------------------ DB CONNECTION FUNCTION ------------------
def get_db_connection():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        port=41295
    )


# ------------------ HOME ------------------
from fastapi import Request

@app.api_route("/", methods=["GET", "HEAD"])
def home(request: Request):
    return {"message": "DataSense AI Backend Running 🚀"}


# ------------------ CLEAN ENDPOINT ------------------
@app.post("/clean")
async def clean(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        df = clean_data(df)

        return {
            "message": "Data cleaned successfully",
            "rows": df.shape[0],
            "columns": df.shape[1]
        }

    except Exception as e:
        return {"error": str(e)}


# ------------------ TRAIN ENDPOINT ------------------
@app.post("/train")
async def train_model(
    file: UploadFile = File(...),
    target_column: str = Form(""),
    problem_type: str = Form(...)
):
    try:
        # -------- READ FILE --------
        df = pd.read_csv(file.file)
        df = clean_data(df)

        # -------- VALIDATION --------
        if problem_type != "clustering":
            if not target_column:
                return {"error": "Target column required"}

            if target_column not in df.columns:
                return {"error": f"{target_column} not found"}

            X = df.drop(columns=[target_column])
            y = df[target_column]

            X = pd.get_dummies(X)

        else:
            X = pd.get_dummies(df)
            y = None

        # -------- MODEL --------
        if problem_type == "classification":
            results, best, explanation, importance = run_classification(X, y)

        elif problem_type == "regression":
            results, best = run_regression(X, y)
            explanation = "Best model selected based on highest R2 score"
            importance = "Not available"

        else:
            results = run_clustering(X)
            best = ("KMeans", 0)
            explanation = "Clustering performed"
            importance = "Not applicable"

        # -------- EXTRACT BEST --------
        if isinstance(best, (list, tuple)):
            model_name = best[0]
            score = float(best[1])
        else:
            model_name = best
            score = 0.0

        # -------- SAVE TO DATABASE --------
        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            cursor.execute(
                "INSERT INTO results (filename, best_model, score, problem_type) VALUES (%s, %s, %s, %s)",
                (file.filename, model_name, score, problem_type)
            )

            conn.commit()
            cursor.close()
            conn.close()

        except Exception as db_error:
            print("DB Error:", db_error)

        # -------- RESPONSE --------
        return {
            "results": results,
            "best_model": [model_name, score],
            "score": score,
            "explanation": explanation,
            "feature_importance": importance
        }

    except Exception as e:
        print("TRAIN ERROR:", e)
        return {"error": str(e)}


# ------------------ HISTORY ENDPOINT ------------------
@app.get("/history")
def get_history():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
        SELECT best_model, score, problem_type
        FROM results
        ORDER BY score DESC
        """)

        rows = cursor.fetchall()

        history = []
        for row in rows:
            history.append({
                "model": row[0],
                "score": float(row[1]),
                "problem_type": row[2]
            })

        cursor.close()
        conn.close()

        return history

    except Exception as e:
        print("History Error:", e)
        return {"error": str(e)}