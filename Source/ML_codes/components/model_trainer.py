import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from Source.ML_codes.exception import CustomException
from Source.ML_codes.logger import logging
from Source.ML_codes.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

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
