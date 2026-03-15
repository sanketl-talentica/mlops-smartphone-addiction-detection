import os

########################### DATA INGESTION #########################

RAW_DIR = "artifacts/raw"
RAW_FILE_PATH = os.path.join(RAW_DIR,"raw.csv")
TRAIN_FILE_PATH = os.path.join(RAW_DIR,"train.csv")
TEST_FILE_PATH = os.path.join(RAW_DIR,"test.csv")

CONFIG_PATH = "config/config.yaml"


######################## DATA PROCESSING ########################

PROCESSED_DIR = "artifacts/processed"
PROCESSED_TRAIN_DATA_PATH = os.path.join(PROCESSED_DIR,"processed_train.csv")
PROCESSED_TEST_DATA_PATH = os.path.join(PROCESSED_DIR,"processed_test.csv")


####################### MODEL TRAINING #################
MODEL_OUTPUT_PATH = "artifacts/models/lgbm_model.pkl"

####################### MONITORING #################
MONITORING_DIR = "artifacts/monitoring"
PREDICTIONS_LOG_PATH = "artifacts/monitoring/predictions_log.csv"
DRIFT_REPORT_PATH = "artifacts/monitoring/drift_report.html"
DATA_QUALITY_REPORT_PATH = "artifacts/monitoring/data_quality_report.html"
MODEL_PERFORMANCE_REPORT_PATH = "artifacts/monitoring/model_performance_report.html"