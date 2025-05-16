# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from Source.ML_codes.exception import CustomException
from Source.ML_codes.logger import logging
from Source.ML_codes.utils import save_object
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
                    ("num_pipeline",numerical_pipeline,numerical_features),
                    ("cat_pipeline",categorical_pipeline,categorical_features)

                ]
            )
            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
                                
            logging.info("Reading the train and test file")

            preprocessor_obj=self.get_data_transformer_object()

            target_column="House_Price"
            numerical_column=['Square_Footage', 'Num_Bedrooms', 'Num_Bathrooms', 
                      'Year_Built', 'Lot_Size', 'Garage_Size']
            
            #divide train dataset in dependant and independant sets

            input_feature_train_df=train_df.drop(columns=[target_column],axis=1)
            target_feature_train_df=train_df[target_column]
                                
            #divide test dataset in dependant and independant sets

            input_feature_test_df=test_df.drop(columns=[target_column],axis=1)
            target_feature_test_df=test_df[target_column]

            logging.info("Applying Preprocessing on training and testing datasets")

            input_feature_train=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test=preprocessor_obj.fit_transform(input_feature_test_df)

            train_arr=np.c_[input_feature_train, np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing info")

            save_object(
                file_path= self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            raise CustomException(e, sys)