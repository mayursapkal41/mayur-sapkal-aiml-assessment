from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os

from dotenv import load_dotenv

import google.generativeai as genai

# =========================
# LOAD ENV VARIABLES
# =========================

load_dotenv()

# =========================
# CONFIGURE GEMINI
# =========================

genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

gemini_model = genai.GenerativeModel(
    "gemini-2.5-flash"
)

# =========================
# LOAD MODEL + FEATURES
# =========================

model = joblib.load("models/model.pkl")

feature_columns = joblib.load(
    "models/feature_columns.pkl"
)

# =========================
# FASTAPI APP
# =========================

app = FastAPI(
    title="Lead Conversion Prediction API",
    description="AI-powered lead conversion prediction system",
    version="1.0"
)

# =========================
# REQUEST SCHEMA
# =========================

class PredictionRequest(BaseModel):

    avg_session_duration: float
    avg_scroll_depth: float
    total_clicks: int
    avg_mouse_activity: float
    total_sessions: int
    total_page_visits: int
    return_visitor: int

# =========================
# EXPLANATION SCHEMA
# =========================

class ExplanationRequest(BaseModel):

    conversion_probability: float
    total_sessions: int
    total_page_visits: int
    return_visitor: int

# =========================
# ROOT ENDPOINT
# =========================

@app.get("/")
def home():

    return {
        "message": "Lead Conversion Prediction API is running!"
    }

# =========================
# PREDICT ENDPOINT
# =========================

@app.post("/predict")
def predict(data: PredictionRequest):

    try:

        # Create default feature dictionary
        input_dict = {
            feature: 0
            for feature in feature_columns
        }

        # Fill important behavioral features
        input_dict["avg_session_duration"] = (
            data.avg_session_duration
        )

        input_dict["avg_scroll_depth"] = (
            data.avg_scroll_depth
        )

        input_dict["total_clicks"] = (
            data.total_clicks
        )

        input_dict["avg_mouse_activity"] = (
            data.avg_mouse_activity
        )

        input_dict["total_sessions"] = (
            data.total_sessions
        )

        input_dict["total_page_visits"] = (
            data.total_page_visits
        )

        input_dict["return_visitor"] = (
            data.return_visitor
        )

        # Convert to DataFrame
        input_df = pd.DataFrame([input_dict])

        # Prediction
        prediction = model.predict(input_df)[0]

        probability = model.predict_proba(
            input_df
        )[0][1]

        # Confidence Level
        if probability >= 0.8:
            confidence = "High"
        elif probability >= 0.5:
            confidence = "Medium"
        else:
            confidence = "Low"

        # Risk Level
        if probability >= 0.7:
            risk_level = "Low"
        elif probability >= 0.4:
            risk_level = "Medium"
        else:
            risk_level = "High"

        return {
            "prediction": int(prediction),
            "conversion_probability": round(
                float(probability), 4
            ),
            "confidence": confidence,
            "risk_level": risk_level
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

# =========================
# EXPLAIN ENDPOINT
# =========================

@app.post("/explain")
def explain(data: ExplanationRequest):

    try:

        prompt = f"""
        You are an AI sales analyst.

        Analyze this lead behavior and explain
        the conversion likelihood in simple
        business language.

        Lead Data:
        - Conversion Probability: {data.conversion_probability}
        - Total Sessions: {data.total_sessions}
        - Total Page Visits: {data.total_page_visits}
        - Return Visitor: {data.return_visitor}

        Give:
        1. Short summary
        2. Key behavioral insights
        3. Conversion intent analysis

        Keep response concise and professional.
        """

        response = gemini_model.generate_content(
            prompt
        )

        return {
            "ai_explanation": response.text
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )