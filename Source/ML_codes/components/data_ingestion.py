import os
import sys
from Source.ML_codes.exception import CustomException
from Source.ML_codes.logger import logging
import pandas as pd
from dataclasses import dataclass
from Source.ML_codes.utils import read_sql_data
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "raw.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            # Reading data from MySQL using utility function
            logging.info("Reading data from MySQL database...")
            df = read_sql_data()
            logging.info("Successfully read data from MySQL.")

            # Creating directories for saving artifacts if they don't exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            logging.info(f"Created directory for saving artifacts if not already present.")

            # Saving raw data to CSV
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info(f"Raw data saved at {self.ingestion_config.raw_data_path}.")

            # Splitting dataset into train and test sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            logging.info("Split data into training and testing sets.")

            # Saving train and test datasets to CSV files
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info(f"Train data saved at {self.ingestion_config.train_data_path}.")
            logging.info(f"Test data saved at {self.ingestion_config.test_data_path}.")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.raw_data_path  # Optionally returning raw data path
            )

        except Exception as e:
            logging.error(f"Error occurred during data ingestion: {e}")
            raise CustomException(e, sys)
