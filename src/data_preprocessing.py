import os
import time
import pandas as pd
import numpy as np
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common import read_yaml,load_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

logger = get_logger(__name__)

class DataProcessor:

    def __init__(self, train_path, test_path, processed_dir, config_path):
        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir = processed_dir

        self.config = read_yaml(config_path)

        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
        
    
    def preprocess_data(self,df):
        try:
            logger.info("Starting our Data Processing step")

            # Drop irrelevant columns — defined in config so it works across datasets
            logger.info("Dropping the unwanted columns")
            drop_cols = self.config["data_processing"]["drop_columns"]
            df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

            # Remove duplicate rows to avoid model bias from repeated data
            df.drop_duplicates(inplace=True)
            time.sleep(2)
            print()
            print()

            # Load column lists from config to keep preprocessing config-driven
            cat_cols = self.config["data_processing"]["categorical_columns"]
            num_cols = self.config["data_processing"]["numerical_columns"]

            # Drop rows with any null values — dataset is large enough that dropping is safer than imputing
            logger.info("Handling null values")
            logger.info(f"Null values before dropping: {df.isnull().sum().sum()}")
            df.dropna(inplace=True)
            logger.info("Dropped all rows with null values")
            time.sleep(2)
            print()

            # Label encode categorical columns — converts string categories to integers for model compatibility
            logger.info("Applying Label Encoding")
            label_encoder = LabelEncoder()
            mappings={}

            for col in cat_cols:
                df[col] = label_encoder.fit_transform(df[col])
                mappings[col] = {label:code for label,code in zip(label_encoder.classes_ , label_encoder.transform(label_encoder.classes_))}

            logger.info("Label Mappings are : ")
            for col,mapping in mappings.items():
                logger.info(f"{col} : {mapping}")
            time.sleep(2)
            print()

            # Apply log1p transform to highly skewed numerical columns — reduces the effect of extreme outliers
            logger.info("Doing Skewness Handling")
            skew_threshold = self.config["data_processing"]["skewness_threshold"]
            skewness = df[num_cols].apply(lambda x:x.skew())

            for column in skewness[skewness>skew_threshold].index:
                df[column] = np.log1p(df[column])
            time.sleep(2)
            print()

            return df
        
        except Exception as e:
            logger.error(f"Error during preprocess step {e}")
            raise CustomException("Error while preprocess data", e)
        
    
    def balance_data(self,df):
        try:
            logger.info("Handling Imbalanced Data")

            # Separate features and target — target column name is read from config
            target = self.config["data_processing"]["target_column"]
            X = df.drop(columns=target)
            y = df[target]

            # SMOTE generates synthetic samples for the minority class — prevents model from being biased towards majority class
            smote = SMOTE(random_state=42)
            X_resampled , y_resampled = smote.fit_resample(X,y)

            # Reconstruct the dataframe after resampling
            balanced_df = pd.DataFrame(X_resampled , columns=X.columns)
            balanced_df[target] = y_resampled

            logger.info("Data balanced sucesffuly")
            time.sleep(2)
            print()
            return balanced_df

        except Exception as e:
            logger.error(f"Error during balancing data step {e}")
            raise CustomException("Error while balancing data", e)

    def select_features(self,df):
        try:
            logger.info("Starting our Feature selection step")

            # Separate features and target — target column name is read from config
            target = self.config["data_processing"]["target_column"]
            X = df.drop(columns=target)
            y = df[target]

            # Use RandomForest to rank features by importance — it measures how much each feature reduces impurity
            model =  RandomForestClassifier(random_state=42)
            model.fit(X,y)

            feature_importance = model.feature_importances_

            # Build a sorted dataframe of features by importance score
            feature_importance_df = pd.DataFrame({
                        'feature':X.columns,
                        'importance':feature_importance
                            })
            top_features_importance_df = feature_importance_df.sort_values(by="importance" , ascending=False)

            # Number of top features to keep is driven by config — keeps it flexible without code changes
            num_features_to_select = self.config["data_processing"]["no_of_features"]

            top_10_features = top_features_importance_df["feature"].head(num_features_to_select).values

            logger.info(f"Features selected : {top_10_features}")

            # Keep only top features + target column
            top_10_df = df[top_10_features.tolist() + [target]]

            logger.info("Feature slection completed sucesfully")
            time.sleep(2)
            print()

            return top_10_df

        except Exception as e:
            logger.error(f"Error during feature selection step {e}")
            raise CustomException("Error while feature selection", e)

    def save_data(self,df , file_path):
        try:
            logger.info("Saving our data in processed folder")

            # Save without index to avoid writing an extra unnamed column in the CSV
            df.to_csv(file_path, index=False)

            logger.info(f"Data saved sucesfuly to {file_path}")
            time.sleep(2)
            print()

        except Exception as e:
            logger.error(f"Error during saving data step {e}")
            raise CustomException("Error while saving data", e)

    def process(self):
        try:
            logger.info("Loading data from RAW directory")

            # Load train and test splits produced by data ingestion
            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path)

            # Clean, encode and fix skewness for both splits
            train_df = self.preprocess_data(train_df)
            test_df = self.preprocess_data(test_df)

            # Balance class distribution using SMOTE — only applied to train, not test
            train_df = self.balance_data(train_df)

            # Select top features from train — test is aligned to same columns to avoid mismatch
            train_df = self.select_features(train_df)
            test_df = test_df[train_df.columns]

            # Persist processed data to disk for model training stage
            self.save_data(train_df,PROCESSED_TRAIN_DATA_PATH)
            self.save_data(test_df , PROCESSED_TEST_DATA_PATH)

            logger.info("Data processing completed sucesfully")
        except Exception as e:
            logger.error(f"Error during preprocessing pipeline {e}")
            raise CustomException("Error while data preprocessing pipeline", e)
              
    
    
if __name__=="__main__":
    processor = DataProcessor(TRAIN_FILE_PATH,TEST_FILE_PATH,PROCESSED_DIR,CONFIG_PATH)
    processor.process()       
    
        

