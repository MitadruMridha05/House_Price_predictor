import os
import sys
from dataclasses import dataclass
from urllib.parse import urlparse

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

import mlflow
import mlflow.sklearn
import numpy as np

from Source.ML_codes.exception import CustomException
from Source.ML_codes.logger import logging
from Source.ML_codes.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def eval_metrics(self, actual, pred):
        rmse=np.sqrt(mean_squared_error(actual, pred))
        mae=mean_absolute_error(actual, pred)
        r2=r2_score(actual, pred)
        return rmse, mae, r2

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            x_train,y_train,x_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regressor": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            param = {
                "Random Forest": {
                    'n_estimators': [100, 200, 300],
                   # 'max_depth': [None, 10, 20, 30],
                    #'min_samples_split': [2, 5, 10],
                    #'min_samples_leaf': [1, 2, 4],
                    #'max_features': ['auto', 'sqrt', 'log2']
                },
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error'],
                   # 'splitter': ['best', 'random'],
                   # 'max_depth': [None, 10, 20, 30],
                   # 'min_samples_split': [2, 5, 10],
                   # 'min_samples_leaf': [1, 2, 4],
                    #'max_features': ['auto', 'sqrt', 'log2', None]
                },
                "Gradient Boosting": {
                    'n_estimators': [100, 200, 300],
                    #'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    #'subsample': [0.8, 1.0],
                    #'max_depth': [3, 5, 7],
                    #'min_samples_split': [2, 5, 10],
                    #'min_samples_leaf': [1, 2, 4],
                    #'max_features': ['auto', 'sqrt', 'log2']
                },
                "Linear Regressor": {
                    'fit_intercept': [True, False],
                    #'normalize': [True, False]
                },
                "XGBRegressor": {
                    'n_estimators': [100, 200, 300],
                   # 'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    #'max_depth': [3, 5, 7, 10],
                    #'subsample': [0.8, 1.0],
                    #'colsample_bytree': [0.8, 1.0],
                    #'gamma': [0, 0.1, 0.2]
                },
                "CatBoosting Regressor": {
                    'iterations': [100, 200, 300],
                    #'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    #'depth': [3, 5, 7, 10],
                    #'l2_leaf_reg': [1, 3, 5, 7],
                    #'border_count': [32, 50, 100]
                },
                "AdaBoost Regressor": {
                    'n_estimators': [50, 100, 200],
                    #'learning_rate': [0.01, 0.05, 0.1, 1.0],
                    #'loss': ['linear', 'square', 'exponential']
                }
            }

            model_report:dict=evaluate_models(x_train, y_train, x_test, y_test, models,param)

            #To get the best model score from the dict
            best_model_score = max(sorted(model_report.values()))

            ##To get the best model name from the dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            print("This is the Best Model: ",best_model_name)
            models_names=list(param.keys())
            actual_model=""

            for model in models_names:
                if best_model_name==model:
                    actual_model=actual_model+model

            best_params = param[actual_model]

            mlflow.set_registry_uri("https://dagshub.com/MitadruMridha05/House_Price_predictor.mlflow")
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

             # mlflow

            with mlflow.start_run():

                predicted_qualities = best_model.predict(x_test)

                (rmse, mae, r2) = self.eval_metrics(y_test, predicted_qualities)

                mlflow.log_params(best_params)

                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("r2", r2)
                mlflow.log_metric("mae", mae)


                # Model registry does not work with file store
                if tracking_url_type_store != "file":

                    # Register the model
                    # There are other ways to use the Model Registry, which depends on the use case,
                    # please refer to the doc for more information:
                    # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                    mlflow.sklearn.log_model(best_model, "model", registered_model_name=actual_model)
                else:
                    mlflow.sklearn.log_model(best_model, "model")

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(x_test)

            r2_square = r2_score(y_test, predicted)
            
            return r2_square
        
        except Exception as e:
            raise CustomException(e,sys)
