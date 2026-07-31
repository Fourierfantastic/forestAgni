import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            model_path = os.path.join("artifacts","model.pkl")
            preprocessor_path = os.path.join("artifacts","preprocessor.pkl")
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds            
        except Exception as e:
            raise CustomException(e,sys)


class CustomData:
    def __init__(
        self,
        day: int,
        month: int,
        year: int,
        Temperature: float,
        RH: float,
        Ws: float,
        Rain: float,
        FFMC: float,
        DMC: float,
        DC: float,
        ISI: float,
        BUI: float,
        Region: int,
        Classes: str
    ):

        self.day = day
        self.month = month
        self.year = year
        self.Temperature = Temperature
        self.RH = RH
        self.Ws = Ws
        self.Rain = Rain
        self.FFMC = FFMC
        self.DMC = DMC
        self.DC = DC
        self.ISI = ISI
        self.BUI = BUI
        self.Region = Region
        self.Classes = Classes

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "day": [self.day],
                "month": [self.month],
                "year": [self.year],
                "Temperature": [self.Temperature],
                "RH": [self.RH],
                "Ws": [self.Ws],
                "Rain": [self.Rain],
                "FFMC": [self.FFMC],
                "DMC": [self.DMC],
                "DC": [self.DC],
                "ISI": [self.ISI],
                "BUI": [self.BUI],
                "Region": [self.Region],
                "Classes": [self.Classes]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)