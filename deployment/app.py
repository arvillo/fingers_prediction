import streamlit as st
import tensorflow as tf
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

st.set_page_config(
    page_title='Fingers Prediction',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Load Label Encoder
with open('label_encoder.pkl', 'rb') as file_1:
  label_encoder = joblib.load(file_1)
# Load All Models
model_cnn = load_model('model_cnn.h5')

img_file_buffer = st.camera_input("Take a picture")

if img_file_buffer is not None:
    # To read image file buffer with OpenCV:
    img = image.load_img(img_file_buffer, target_size=(128, 128))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    prediction = model_cnn.predict(img_batch)
    result_max_proba = np.argmax(prediction,axis=-1)[0]
    result_class = label_encoder.classes_[result_max_proba]
    results = {'0L':'0 from Left Hand',
               '1L':'1 from Left Hand',
               '2L':'2 from Left Hand',
               '3L':'3 from Left Hand',
               '4L':'4 from Left Hand',
               '5L':'5 from Left Hand',
               '0R':'0 from Right Hand',
               '1R':'1 from Right Hand',
               '2R':'2 from Right Hand',
               '3R':'3 from Right Hand',
               '4R':'4 from Right Hand',
               '5R':'5 from Right Hand'}
    st.write('The finger is', results[result_class])
    st.write('')

