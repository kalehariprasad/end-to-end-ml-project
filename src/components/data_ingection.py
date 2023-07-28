import os
import sys
from src.exeption import CustomException
from src.logger import logging
import pandas as pd 
from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationconfig
from src.components.model_trainer import ModelTrainerconfig
from src.components.model_trainer import Modeltrainer

from dataclasses import dataclass
@dataclass
class Datainjectionconfig:
    train_data_path:str=os.path.join('artifacts',"train.csv") 
    test_data_path:str=os.path.join('artifacts',"test.csv") 
    raw_data_path:str=os.path.join('artifacts',"data.csv") 


class Datainjection:
    def __init__(self):
        self.injection_config = Datainjectionconfig()
    def intiate_data_injection(self):
        logging.info("entered the data injection part")
        try:
            df=pd.read_csv('notebook/students-performance-in-exams/StudentsPerformance.csv')
            logging.info('Read the data set as dataframe')
            os.makedirs(os.path.dirname(self.injection_config.train_data_path),exist_ok=True)

            df.to_csv(self.injection_config.raw_data_path,index=False,header=True)
            logging.info('train test split intiated')
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.injection_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.injection_config.test_data_path,index=False,header=True)
            logging.info('train test and raw data injection compleeted')
            return(
                self.injection_config.train_data_path,
                self.injection_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":

    obj=Datainjection()
    train_data,test_data=obj.intiate_data_injection()
    data_transformations=DataTransformation()
    train_arr,test_arr=data_transformations.initiate_data_transformation(train_data, test_data)
    modeltrainer=Modeltrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))