# import libraries
import streamlit as st
import tensorflow as tf

# define class names as seen in the data
class_names = ['Covid', 'Normal', 'Viral Pneumonia']

# create function that allows us to load a saved model
@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('Covid19_ComputerVisionProject_Version2_model.hdf5')
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()

# header
st.write("""
         # Covid-19 Computer Vision Project
         """
         )

# allowing a space for people to upload images
file = st.file_uploader("Please upload an X-ray scan file", type=["jpg", "png", 'jpeg'])

# importing libraries to handle image data
import cv2
from PIL import Image, ImageOps
import numpy as np

# creating function to import an image and predict what class it belongs to
st.set_option('deprecation.showfileUploaderEncoding', False)
def import_and_predict(image_data, model):

        size = (256,256)
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.

        img_reshape = img[np.newaxis,...]

        prediction = model.predict(img_reshape)

        return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    score = tf.nn.softmax(predictions[0])
    st.write(predictions)
    st.write(score)

# adding output text
if score[0] > score[1] > score[2]:
    st.write(
    "{} Infected with an {:.2f} percent chance of infection."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
    )
elif score[1] > score[0] > score[2]:
    st.write(
    "Person is {} with an {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
    )
else:
    st.write(
    "Person has {} with an {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
    )
