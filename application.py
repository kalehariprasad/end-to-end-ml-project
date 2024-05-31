import streamlit as st
import pickle 
import pandas as pd 
import numpy  as np
from sklearn.preprocessing import StandardScaler
import warnings
from src.logger import logging
from src.pipeline.predict_pipeline import CustomData,PredictPipeline


data_instance = CustomData()
input_data = data_instance.get_dataframe()
predictor = PredictPipeline()

if st.button('predict'):
    try:
        prediction = predictor.Predict(features=input_data)
        logging.info(f'processed data is: {input_data}')
        
        if prediction is not None:
            st.write(f"expected math score will be: {np.round(prediction[0], decimals=2)}")
    except Exception as e:
        st.error(f"Error occurred during prediction: {e}")









