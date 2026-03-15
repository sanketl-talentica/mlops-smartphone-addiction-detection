# MLOps — Smartphone Addiction Detection

A production-grade MLOps pipeline to predict smartphone addiction based on user behavior data. Built with modular pipeline stages, config-driven execution, experiment tracking, data versioning, model monitoring, and CI/CD deployment.

---

## Tech Stack

| Area | Tool |
|---|---|
| ML Model | LightGBM |
| Experiment Tracking | MLflow + DagsHub |
| Data Versioning | DVC |
| API | FastAPI |
| Model Monitoring | Evidently AI |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Deployment | EC2 |

---

## Project Structure

```
mlops-smartphone-addiction-detection/
├── config/
│   ├── config.yaml              # Pipeline config (paths, thresholds, feature count)
│   ├── paths_config.py          # Centralised file path constants
│   └── model_params.py          # LightGBM hyperparameter search space
├── dataset/
│   └── Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv
├── artifacts/
│   ├── raw/                     # Ingestion output (raw, train, test CSVs)
│   ├── processed/               # Preprocessing output (balanced, feature-selected CSVs)
│   ├── models/                  # Saved model (lgbm_model.pkl)
│   └── monitoring/              # Evidently reports + predictions log
├── src/
│   ├── data_ingestion.py        # Stage 1: Load and split raw data
│   ├── data_preprocessing.py    # Stage 2: Clean, encode, SMOTE, feature selection
│   ├── model_training.py        # Stage 3: Train LightGBM + MLflow tracking
│   ├── logger.py                # Centralised logging
│   └── custom_exception.py      # Custom exception handler
├── pipeline/
│   └── training_pipeline.py     # Runs all 3 stages sequentially
├── utils/
│   └── common.py                # Shared utilities
├── templates/
│   ├── index.html               # Prediction UI
│   └── monitoring.html          # Monitoring dashboard
├── static/
│   └── style.css
├── application.py               # FastAPI app (predict + monitoring endpoints)
├── dvc.yaml                     # DVC pipeline definition
├── Dockerfile
├── .github/workflows/ci-cd.yml  # GitHub Actions CI/CD
└── requirements.txt
```

---

## Pipeline Stages

### Stage 1 — Data Ingestion
- Copies raw CSV to `artifacts/raw/raw.csv`
- Splits into train (80%) / test (20%)

### Stage 2 — Data Preprocessing
- Drops duplicates and nulls
- Label encodes categorical columns
- Applies `log1p` transform to skewed numerical columns
- Applies **SMOTE** on training data only to balance class distribution
- Selects top N features using **RandomForestClassifier** importance ranking

### Stage 3 — Model Training
- Trains **LightGBM** with **RandomizedSearchCV** hyperparameter tuning
- Logs params, metrics, classification report, confusion matrix to **MLflow + DagsHub**
- Saves model to `artifacts/models/lgbm_model.pkl`
- Writes `metrics.json` for DVC metrics tracking

---

## Design Decisions

**Why no scaling?** LightGBM is tree-based and scale-invariant — normalization has no effect.

**Why SMOTE on train only?** Test set must reflect real-world distribution. Applying SMOTE to test data produces fake samples and gives misleading metrics.

**Why F1 as scoring metric?** Dataset is imbalanced — accuracy alone is misleading. F1 balances precision and recall.

**Why feature selection?** Removing low-signal features reduces noise and overfitting, especially at 7500 rows.

---

## Running the Pipeline

### Option 1 — DVC (recommended)
```bash
dvc repro          # runs only changed stages
dvc metrics show   # view accuracy, F1 etc.
```

### Option 2 — Manual
```bash
python src/data_ingestion.py
python src/data_preprocessing.py
python src/model_training.py
```

---

## API Endpoints

Start the server:
```bash
python application.py
```

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Prediction UI |
| `/predict` | POST | Returns addiction prediction as JSON |
| `/monitoring` | GET | Monitoring dashboard |
| `/report/drift` | GET | Data drift report |
| `/report/quality` | GET | Data quality report |
| `/report/performance` | GET | Model performance report |
| `/report/target-drift` | GET | Target/prediction drift report |
| `/docs` | GET | Auto-generated Swagger UI |

### Sample `/predict` request
```json
{
  "age": 22,
  "daily_screen_time_hours": 8.5,
  "social_media_hours": 4.0,
  "gaming_hours": 2.0,
  "work_study_hours": 3.0,
  "sleep_hours": 6.0,
  "notifications_per_day": 150,
  "app_opens_per_day": 80,
  "weekend_screen_time": 10.0,
  "academic_work_impact": 1
}
```

### Sample response
```json
{
  "prediction": 1,
  "result": "Addicted"
}
```

---

## Docker

```bash
# Build
docker build -t smartphone-addiction-detection .

# Run
docker run -p 8080:8080 smartphone-addiction-detection
```

---

## CI/CD Pipeline (GitHub Actions)

On every push to `main`:

```
git push
    ↓
Job 1 — Train:   dvc pull → dvc repro → dvc push
    ↓
Job 2 — Build:   docker build → push to Docker Hub
    ↓
Job 3 — Deploy:  SSH into EC2 → docker pull → restart container
```

### Required GitHub Secrets

| Secret | Description |
|---|---|
| `DAGSHUB_USERNAME` | DagsHub username |
| `DAGSHUB_TOKEN` | DagsHub access token |
| `MLFLOW_TRACKING_USERNAME` | DagsHub username |
| `MLFLOW_TRACKING_PASSWORD` | DagsHub token |
| `DOCKER_USERNAME` | Docker Hub username |
| `DOCKER_PASSWORD` | Docker Hub password |
| `EC2_HOST` | EC2 public IP |
| `EC2_USER` | EC2 username (e.g. `ubuntu`) |
| `EC2_SSH_KEY` | EC2 private key (.pem contents) |

---

## Setup

```bash
# Create virtual environment
python3 -m venv venv && source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Add dataset
cp /path/to/dataset.csv dataset/

# Run pipeline
dvc repro

# Start API
python application.py
```

---

## Dependencies

| Package | Purpose |
|---|---|
| pandas, numpy | Data manipulation |
| scikit-learn | Preprocessing, feature selection, metrics |
| imbalanced-learn | SMOTE for class balancing |
| lightgbm | Gradient boosting model |
| mlflow, dagshub | Experiment tracking |
| fastapi, uvicorn | REST API server |
| evidently | Model monitoring and drift detection |
| dvc | Data and pipeline versioning |
| python-dotenv | Environment variable management |
