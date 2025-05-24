"""
Common Functions that you will use throughout the project

"""

import os
import sys
import numpy as np
import pandas as pd
import dill
from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


####### saving object files by taking the filepath we want to save it in and the object
def save_object(filepath, obj):
    try:
        dir_path = os.path.dirname(filepath)

        os.makedirs(dir_path, exist_ok = True)

        with open(filepath, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


####### Evaluating the metrics of the models we are using
def evaluate_models(x_train, y_train, x_test, y_test, models, params):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]

            gs = GridSearchCV(estimator = model, param_grid = param, cv = 3)        # Hyperparameter tuning

            gs.fit(x_train, y_train)
            model.set_params(**gs.best_params_)

            model.fit(x_train, y_train)         #Train the model

            y_train_pred = model.predict(x_train)

            y_test_pred = model.predict(x_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = [train_model_score, test_model_score]
        return report
    except Exception as e:
        raise CustomException(e, sys)