"""
In here we train the model
"""

import os
import sys
import pandas as pd
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors  import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array, preprocessor_path):
        try:
            logging.info("Split training and test input data")

            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )
            models = {
                "Random Forest" : RandomForestRegressor(),
                "Decision Tree" : DecisionTreeRegressor(),
                "Gradient Boosting" : GradientBoostingRegressor(),
                "Linear Regression" : LinearRegression(),
                "K-Neighbors Regressor" : KNeighborsRegressor(),
                "XGBoosting Regressor" : XGBRegressor(),
                "CatBoosting Regressor" : CatBoostRegressor(),
                "AdaBoost Regressor" : AdaBoostRegressor(),
            }

            params={
                "Decision Tree": {
                    "criterion":["squared_error", "friedman_mse", "absolute_error", "poisson"],
                    # "splitter":["best","random"],
                    # "max_features":["sqrt","log2"],
                },
                "Random Forest":{
                    # "criterion":["squared_error", "friedman_mse", "absolute_error", "poisson"],
                    # "max_features":["sqrt","log2",None],
                    "n_estimators": [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # "loss":["squared_error", "huber", "absolute_error", "quantile"],
                    "learning_rate":[.1,.01,.05,.001],
                    "subsample":[0.6,0.7,0.75,0.8,0.85,0.9],
                    # "criterion":["squared_error", "friedman_mse"],
                    # "max_features":["auto","sqrt","log2"],
                    "n_estimators": [8,16,32,64,128,256]
                },
                "Linear Regression": {},
                "XGBoosting Regressor":{
                    "learning_rate":[.1,.01,.05,.001],
                    "n_estimators": [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    "depth": [6,8,10],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "iterations": [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    "learning_rate":[.1,.01,0.5,.001],
                    # "loss":["linear","square","exponential"],
                    "n_estimators": [8,16,32,64,128,256]
                },
                "K-Neighbors Regressor": {},
            }

            model_report:dict = evaluate_models(
                x_train = x_train,
                y_train = y_train,
                x_test = x_test,
                y_test = y_test,
                models = models,
                params = params,
            )

            ### To get best model score from the dictionary
            # Since model_report.values() returns a 1D array with two value [train_score, test_score]
            # I converted the dictionary to dataframe to access the second column (train_score) and find the max value
            best_model_test_score = max(pd.DataFrame(model_report.values())[1])

            ### here it will return an array containing the training and testing score of the model with best test score
            best_model_score = list(model_report.values())[
                list(pd.DataFrame(model_report.values())[1]).index(best_model_test_score)
            ]

            ### To get best model name from the dictionary
            best_model_name = pd.DataFrame(model_report).columns[
                list(pd.DataFrame(model_report.values())[1]).index(best_model_test_score)
            ]
            best_model = models[best_model_name]

            if best_model_test_score < 0.6:
                raise CustomException("No best model Found")
            logging.info(f"Best found model on both training and testing dataset: {best_model_name}")

            ### Saving the best model with the path and name made in the config class
            save_object(
                filepath = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            # predicted = best_model.predict(x_test)
            # r2_square = r2_score(y_test, predicted)

            return best_model_score, model_report
        except Exception as e:
            raise CustomException(e, sys)