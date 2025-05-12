# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

from Source.ML_codes.exception import CustomException
from Source.ML_codes.logger import logging
import os
import sys

from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        this function is responsible for data transformation
        '''
        try:
            # Identify numerical and categorical features
            numerical_features = ['Square_Footage', 'Num_Bedrooms', 'Num_Bathrooms', 
                      'Year_Built', 'Lot_Size', 'Garage_Size']
            categorical_features = ['Neighborhood_Quality']

            # Create preprocessing pipelines for numerical and categorical features
            numerical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', MinMaxScaler())
                ])
            
            categorical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(
                    handle_unknown='ignore',
                    sparse_output=False,  # New parameter name in scikit-learn >=1.2
                    drop='first'  # Reduces multicollinearity
                ))
                ])

            logging.info(f"Categorical Column:{categorical_features}")
            logging.info(f"Numerical Column:{numerical_features}")

            preprocessor=ColumnTransformer(
                [
                    ("numerical_columns",numerical_pipeline,numerical_features)
                    ("categorical_columns",categorical_pipeline,categorical_features)

                ]
            )
        except Exception as e:
            raise CustomException(e, sys)
        
        return preprocessor