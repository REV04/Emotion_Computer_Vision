# Import Libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from io import StringIO
import tensorflow as tf

# Load the model
model = load_model("model3.h5")

# Classify the class
class_names = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]

# Function for predict from file
def predict_from_file(uploaded_file):
    img = image.load_img(uploaded_file, target_size=(110, 110))
    # Show the image
    img
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0) 
    img_batch = img_batch / 255.0
    prediction_inf = model.predict(img_batch)
    result_max_proba = prediction_inf.argmax(axis=-1)[0]
    result_class = class_names[result_max_proba]
    return prediction_inf, result_class

# Function to run the prediction
def run():
    st.title("Identify Your Emotion")
    uploaded_file = st.file_uploader("Choose a file", type=['png', 'jpg'])
    if uploaded_file is not None:
        st.write('Prediction result is: ', predict_from_file(uploaded_file)[1])
    
if __name__ == '__main__':
  run()


