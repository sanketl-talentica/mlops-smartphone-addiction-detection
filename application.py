import os
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset
from evidently.metrics import (
    DatasetDriftMetric,
    DatasetMissingValuesSummaryMetric,
    ColumnDriftMetric,
    ClassificationQualityMetric,
    ClassificationClassBalance,
    ClassificationConfusionMatrix,
)

from config.paths_config import (
    MODEL_OUTPUT_PATH,
    PROCESSED_TRAIN_DATA_PATH,
    PREDICTIONS_LOG_PATH,
    DRIFT_REPORT_PATH,
    DATA_QUALITY_REPORT_PATH,
    MODEL_PERFORMANCE_REPORT_PATH,
    MONITORING_DIR,
)

app = FastAPI(title="Smartphone Addiction Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

loaded_model = joblib.load(MODEL_OUTPUT_PATH)

# Dynamically read feature order from processed train CSV — stays in sync after every dvc repro
train_df = pd.read_csv(PROCESSED_TRAIN_DATA_PATH)
FEATURE_COLUMNS = [col for col in train_df.columns if col != "addicted_label"]

os.makedirs(MONITORING_DIR, exist_ok=True)

MIN_SAMPLES_FOR_REPORT = 5

def check_predictions_log(min_samples=MIN_SAMPLES_FOR_REPORT):
    if not os.path.exists(PREDICTIONS_LOG_PATH):
        return None, {"error": "No predictions logged yet. Make some predictions first."}
    current = pd.read_csv(PREDICTIONS_LOG_PATH)
    if len(current) < min_samples:
        return None, {"error": f"Need at least {min_samples} predictions to generate report. Currently have {len(current)}."}
    return current, None

class PredictRequest(BaseModel):
    social_media_hours: float
    daily_screen_time_hours: float
    weekend_screen_time: float
    work_study_hours: float
    sleep_hours: float
    notifications_per_day: int
    gaming_hours: float
    app_opens_per_day: int
    age: int
    academic_work_impact: int

@app.get("/")
def home():
    return FileResponse("templates/index.html")

@app.post("/predict")
def predict(data: PredictRequest):
    input_data = data.model_dump()

    # Build feature DataFrame in exact order the model was trained on
    features = pd.DataFrame([input_data])[FEATURE_COLUMNS]

    prediction = loaded_model.predict(features)[0]
    result = "Addicted" if prediction == 1 else "Not Addicted"

    # Log input + prediction for monitoring
    log_row = features.copy()
    log_row["prediction"] = int(prediction)
    log_row.to_csv(
        PREDICTIONS_LOG_PATH,
        mode="a",
        header=not os.path.exists(PREDICTIONS_LOG_PATH),
        index=False
    )

    return {
        "prediction": int(prediction),
        "result": result
    }

@app.get("/report/drift")
def drift_report():
    """Data Drift — detects if incoming user inputs shifted away from training distribution"""
    current, err = check_predictions_log()
    if err:
        return JSONResponse(status_code=400, content=err)

    reference = pd.read_csv(PROCESSED_TRAIN_DATA_PATH)[FEATURE_COLUMNS]
    current = current[FEATURE_COLUMNS]

    report = Report(metrics=[
        DatasetDriftMetric(),
        DataDriftPreset(),
    ])
    report.run(reference_data=reference, current_data=current)
    report.save_html(DRIFT_REPORT_PATH)

    return FileResponse(DRIFT_REPORT_PATH, media_type="text/html")

@app.get("/report/quality")
def data_quality_report():
    """Data Quality — checks missing values, outliers, feature distributions in incoming data"""
    current, err = check_predictions_log()
    if err:
        return JSONResponse(status_code=400, content=err)

    reference = pd.read_csv(PROCESSED_TRAIN_DATA_PATH)[FEATURE_COLUMNS]
    current = current[FEATURE_COLUMNS]

    report = Report(metrics=[
        DataQualityPreset(),
        DatasetMissingValuesSummaryMetric(),
    ])
    report.run(reference_data=reference, current_data=current)
    report.save_html(DATA_QUALITY_REPORT_PATH)

    return FileResponse(DATA_QUALITY_REPORT_PATH, media_type="text/html")

@app.get("/report/performance")
def model_performance_report():
    """Model Performance — confusion matrix, class balance, classification metrics on logged predictions"""
    current, err = check_predictions_log()
    if err:
        return JSONResponse(status_code=400, content=err)

    reference = pd.read_csv(PROCESSED_TRAIN_DATA_PATH)
    reference["prediction"] = loaded_model.predict(reference[FEATURE_COLUMNS])

    current_with_pred = current[FEATURE_COLUMNS + ["prediction"]]
    ref_with_pred = reference[FEATURE_COLUMNS + ["prediction"]]

    # Use addicted_label as target for reference, prediction as target for current
    reference["target"] = reference["addicted_label"]
    current_with_pred = current_with_pred.copy()
    current_with_pred["target"] = current_with_pred["prediction"]

    report = Report(metrics=[
        ClassificationQualityMetric(),
        ClassificationClassBalance(),
        ClassificationConfusionMatrix(),
    ])
    report.run(
        reference_data=ref_with_pred.assign(target=reference["addicted_label"]),
        current_data=current_with_pred,
        column_mapping={"target": "target", "prediction": "prediction"}
    )
    report.save_html(MODEL_PERFORMANCE_REPORT_PATH)

    return FileResponse(MODEL_PERFORMANCE_REPORT_PATH, media_type="text/html")

@app.get("/report/target-drift")
def target_drift_report():
    """Target Drift — detects if prediction distribution is shifting (more addicted/not addicted over time)"""
    current, err = check_predictions_log()
    if err:
        return JSONResponse(status_code=400, content=err)

    reference = pd.read_csv(PROCESSED_TRAIN_DATA_PATH)
    reference["prediction"] = reference["addicted_label"]

    current = current.copy()

    report = Report(metrics=[
        TargetDriftPreset(),
    ])
    report.run(
        reference_data=reference[FEATURE_COLUMNS + ["prediction"]],
        current_data=current[FEATURE_COLUMNS + ["prediction"]],
        column_mapping={"target": "prediction"}
    )
    report.save_html(DRIFT_REPORT_PATH.replace("drift", "target_drift"))

    return FileResponse(DRIFT_REPORT_PATH.replace("drift", "target_drift"), media_type="text/html")

@app.get("/monitoring")
def monitoring_dashboard():
    return FileResponse("templates/monitoring.html")

@app.get("/monitoring/stats")
def monitoring_stats():
    if not os.path.exists(PREDICTIONS_LOG_PATH):
        return JSONResponse(status_code=400, content={"error": "No predictions yet."})

    df = pd.read_csv(PREDICTIONS_LOG_PATH)
    total = len(df)
    addicted = int((df["prediction"] == 1).sum())
    not_addicted = total - addicted
    rate = f"{(addicted / total * 100):.1f}%" if total > 0 else "—"

    return {
        "total": total,
        "addicted": addicted,
        "not_addicted": not_addicted,
        "addiction_rate": rate
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
