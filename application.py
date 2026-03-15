import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from config.paths_config import MODEL_OUTPUT_PATH, PROCESSED_TRAIN_DATA_PATH

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

    return {
        "prediction": int(prediction),
        "result": result
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
