import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression,Lasso,ElasticNet,LassoCV,Ridge,RidgeCV

from sklearn.metrics import r2_score,mean_absolute_error

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainConfig:
    trained_model_file_path = os.path.join('artifacts',"model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("split training and test input data.")
            X_train,y_train,X_test,y_test =(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Linear Regression":LinearRegression(),
                "Lasso" : Lasso(),
                "ElasticNet" : ElasticNet(),
                "ridge" : Ridge(),
                
            }

            params = {
    "Linear Regression": {},

    "Lasso": {
        "alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        "max_iter": [100, 500, 1000]
    },

    "ElasticNet": {
        "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
        "l1_ratio": [0.2, 0.4, 0.5, 0.6, 0.8, 1.0],
        "max_iter": [100, 500, 1000]
    },

        "lassocv": {
    "cv": [3, 5, 10],
    "alphas": [
        [0.001, 0.01, 0.1, 1, 10]
    ],
    "max_iter": [100, 500]
},

    "ridge": {
        "alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        "solver": ["auto", "svd", "cholesky", "lsqr", "sag", "saga"]
    },

  "ridgecv": {
    "alphas": [
        [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    ],
    "cv": [3, 5, 10]
}
}
            

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)
            mae  = mean_absolute_error(y_test,predicted)
            r2_square = r2_score(y_test, predicted)
            return(
                 r2_square,
                self.model_trainer_config.trained_model_file_path
            )        
        except Exception as e:
            raise CustomException(e,sys)
            