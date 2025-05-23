"""
Reading the data from a database/ file path
"""

import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformationConfig, DataTransformation
from src.components.model_trainer import ModelTrainerConfig, ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path = str = os.path.join("artifacts", "train.csv")
    test_data_path = str = os.path.join("artifacts", "test.csv")
    raw_data_path = str = os.path.join("artifacts", "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initial_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            ####### Reading the dataset
            df = pd.read_csv("notebook\data\stud.csv")
            logging.info("Read the dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok = True)

            ####### Converted the raw data into a CSV file
            df.to_csv(self.ingestion_config.raw_data_path, index = False, header = True)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            ####### Saving the train and test files
            train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)

            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e, sys)

# Testing data_ingestion.py file /// python src/components/data_ingestion.py

if __name__ == "__main__":
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initial_data_ingestion()

    # Initialising the data transforming function
    data_transformation = DataTransformation()
    train_array, test_array, preprocessor_path = data_transformation.initiate_data_transformation(
        train_path = train_data_path,
        test_path = test_data_path
    )

    # Initialising the model trainer function
    model_trainer = ModelTrainer()
    best_model_scores, model_reports = model_trainer.initiate_model_trainer(
        train_array = train_array,
        test_array = test_array,
        preprocessor_path = preprocessor_path
    )
    logging.info(f"The model report: {model_reports}")
    print(f"The model R2 score for Training dataset: {best_model_scores[0]}, and for the Testing dataset: {best_model_scores[1]}")

