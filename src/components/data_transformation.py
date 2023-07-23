import sys
import os 
import pandas as pd 
import numpy as np
from dataclasses import dataclass
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from src.exeption import CustomException
from src.logger import logging
from src.utils import save_object 
@dataclass
class DataTransformationconfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:

    def __init__(self):
    
        self.data_transformation_config = DataTransformationconfig()

    def get_data_transformer_object(self):
        """
        This function will return the preprocessor.
        """
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                   "lunch",
                "test_preparation_course",
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scalar", StandardScaler())
                ]
            )

            logging.info("Numerical column missing values imputation and scaling completed")

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoding", OneHotEncoder()),
                    ("scaling", StandardScaler())
                ]
            )

            logging.info("Categorical column missing values imputation, One hot encoding, and scaling completed")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            logging.info('Combined both pipelines as preprocessor')
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info('read train and test_data compleeted')
            logging.info(" started obtaining preprocessor object")
            preprocessor_obj=self.get_data_transformer_object()
            logging.info("compleeted obtaining preprocessor objet")
            
            targrt_column_name=['math score']
            numerical_columns = ["writing score", "reading score"]
            input_feature_train_df=train_df.drop(targrt_column_name,axis=1)
            output_feature_train_df=train_df[targrt_column_name]
            input_feature_test_df=test_df.drop(targrt_column_name,axis=1)
            output_feature_test_df=test_df[targrt_column_name]
            logging.info("applying preprocessor to train and test data")
            input_features_train_array =preprocessor_obj.fit_transform(input_feature_train_df)
            input_features_test_array =preprocessor_obj.fit_transform(input_feature_test_df)

            train_arr = np.c_[
                input_features_train_array, np.array(output_feature_test_df)
            ]
            test_arr= np.c_[
                input_features_test_array, np.array(output_feature_train_df)
            ]
            logging.info("Saving preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
                )
            return (train_arr,
                    test_arrm,
                    self.data_transformation_config.preprocessor_obj_file_path)
                    

        except Exception as e:
            raise CustomException(e, sys)
