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
            numerical_columns = ["writing score", "reading score"]
            categorical_columns = [
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course",
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
                    ("encoding", OneHotEncoder(handle_unknown='ignore')),
                ]
            )

            logging.info("Categorical column missing values imputation and one-hot encoding completed")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            logging.info('Combined both pipelines as preprocessor')
            logging.info(f"preprocessor object: {preprocessor}")
            return preprocessor
        
        

        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('Read train and test_data completed')

            # Log the column names of the train_df DataFrame
            logging.info("Train DataFrame columns: {}".format(train_df.columns))

            # Log the column names of the test_df DataFrame
            logging.info("Test DataFrame columns: {}".format(test_df.columns))

            logging.info("Started obtaining preprocessor object")

            preprocessor_obj = self.get_data_transformer_object()

        
            logging.info("Completed obtaining preprocessor object")
            
            target_column_name = ['math score'] 
            numerical_columns = ["writing score", "reading score"]
            input_feature_train_df = train_df.drop(target_column_name, axis=1)
            output_feature_train_df = train_df[target_column_name]
            input_feature_test_df = test_df.drop(target_column_name, axis=1)

            logging.info(f"input test data :{input_feature_train_df}")
            output_feature_test_df = test_df[target_column_name]
            logging.info("Applying preprocessor to train and test data")
            input_features_train_array = preprocessor_obj.fit_transform(input_feature_train_df)
            logging.info(f'input train features:{input_features_train_array} and shape is {input_features_train_array.shape}')
            input_features_test_array = preprocessor_obj.transform(input_feature_test_df)  # Changed fit_transform to transform
            logging.info(f'input test features:{input_features_test_array}and shpae is {input_features_test_array.shape}')
            train_arr = np.c_[
                input_features_train_array, np.array(output_feature_train_df)
            ]
            test_arr = np.c_[
                input_features_test_array, np.array(output_feature_test_df)
            ]
            logging.info("Saving preprocessing object")
            

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            return (train_arr,
                    test_arr,
                    )

        except Exception as e:
            raise CustomException(e, sys)