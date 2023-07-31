import os
import sys 
import pandas as pd
import streamlit as st
from src.exeption import CustomException
from src.logger import logging
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
            logging.info(f"inut variables for prediction are :{data_scaled}")
            pred=model.predict(data_scaled)
            return pred 
        

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self):
        pass


    def get_dataframe(self):
        gender_list=['female','male']
        race_ethnicity_list=['group B', 'group C', 'group A', 'group D' ,'group E']
        parental_level_of_education_list=["bachelor's degree" , 'some college' "master's degree" ,"associate's degree",'high school' ,'some high school']
        lunch_list=['standard' ,'free/reduced']
        test_preparation_course_list=['none','completed']


        gender=st.selectbox('select your Gender',sorted(gender_list))
        race_ethnicity=st.selectbox('select your race/ethnicity',sorted(race_ethnicity_list))
        parental_level_of_education=st.selectbox('student parental level of education',sorted(parental_level_of_education_list))
        lunch=st.selectbox('select lunch type',sorted(lunch_list))
        test_preparation_course=st.selectbox('Does student gone test preparation course',sorted(test_preparation_course_list))
        reading_score=st.number_input(label='Student Reading score',min_value=0,max_value=100)
        writing_score=st.number_input(label='Student Writing score' ,min_value=0,max_value=100)


        data={
        'gender':gender,
        'race/ethnicity':race_ethnicity,
        'parental level of education':parental_level_of_education,
        'lunch':lunch,
        'test preparation course':test_preparation_course,
        'reading score':reading_score,
        'writing score':writing_score

        }
        DataFrame=pd.DataFrame(data=data,index=[0])
        return pd.DataFrame(DataFrame)

        


    
     
    



    





     


    
     
    



    
