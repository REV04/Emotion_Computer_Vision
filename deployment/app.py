import streamlit as st
import eda
import prediction

navigation = st.sidebar.selectbox('Choose Pages:',('EDA','Prediction'))

if navigation == 'EDA':
    eda.run()
else:
    prediction.run()