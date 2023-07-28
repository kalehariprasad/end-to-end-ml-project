import streamlit as st
import pickle 
import pandas as pd 
import numpy  as np
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import PredictPipeline,CustomData


st.title('Student performance indicator')

gender=st.selectbox('select your Gender',['female','male'])
race_ethnicity=st.selectbox('select your race/ethnicity',['group B', 'group C', 'group A', 'group D' ,'group E'])
parental_level_of_education=st.selectbox('student parental level of education',["bachelor's degree" ,
                                        'some college' "master's degree" ,
                                        "associate's degree",
                                        'high school' ,
                                        'some high school'])
lunch=st.selectbox('select lunch type',['standard' ,'free/reduced'])
test_preparation_course=st.selectbox('Does student gone test preparation course',['none','completed'])
reading_score=st.slider(label='Student Reading score',min_value=0,max_value=100)
writing_score=st.slider(label='Student Writing score' ,min_value=0,max_value=100)

data_instance = CustomData(
    gender=gender,
    race_ethnicity=race_ethnicity,
    parental_level_of_education=parental_level_of_education,
    lunch=lunch,
    test_preparation_course=test_preparation_course,
    reading_score=reading_score,
    writing_score=writing_score
)

st.write(data_instance)
input_features = data_instance.get_data_as_dataframe().values

st.write(input_features)

predictor=PredictPipeline()
if st.button('predict'):
    try:
        
        predictions = predictor.Predict(features=input_features)
        st.write("Predicted Scores:")
        st.write(predictions)
    except Exception as e:
        st.error(f"Prediction error: {e}")