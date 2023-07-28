import os
import sys 
import pandas as pd
from src.exeption import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
       pass
    def Predict(self,features):
        try:
            
            model_path="artifacts\model.pkl"
            preprocrssor_path="artifacts\preprocessor.pkl"
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocrssor_path)
            data_scaled=preprocessor.transform(features)
            pred=model.predict(data_scaled)
        

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 gender: str,
                 race_ethnicity: str,
                 parental_level_of_education: str,
                 lunch: str,
                 test_preparation_course: str,
                 reading_score: int,
                 writing_score: int
                 ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_dataframe(self):
        custom_data_input_dict = {
            "gender": self.gender,
            "race/ethnicity": self.race_ethnicity,
            "parental level of education": self.parental_level_of_education,
            "lunch": self.lunch,
            "test preparation course": self.test_preparation_course,
            "reading score": self.reading_score,
            "writing score": self.writing_score
        }

        # Create a DataFrame from the input dictionary
        df = pd.DataFrame([custom_data_input_dict])

        return df

    def __str__(self):
        return f"CustomData(gender='{self.gender}', " \
               f"race/ethnicity='{self.race_ethnicity}', " \
               f"parental level of education='{self.parental_level_of_education}', " \
               f"lunch='{self.lunch}', " \
               f"test preparation course='{self.test_preparation_course}', " \
               f"reading score={self.reading_score}, " \
               f"writing score={self.writing_score})"




     


    
     
    



    
