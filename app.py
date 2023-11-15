import streamlit as st
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
import wget
import zipfile
class_names = ["COVID-19", "Normal", "Pneumonia"]

@st.cache(suppress_st_warning=True)

def predict(model_path):
    
    test_gen =ImageDataGenerator(featurewise_center=False, samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    zca_epsilon=0.0,
    rotation_range=0,
    width_shift_range=0.0,
    height_shift_range=0.0,
    brightness_range=None,
    shear_range=0.0,
    zoom_range=0.0,
    channel_shift_range=0.0,
    fill_mode="nearest",
    cval=0.0,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=1./255,
    preprocessing_function=None,
    data_format=None,
    validation_split=0.0)

    data = test_gen.flow_from_directory('test_set',target_size=(224, 224),
        color_mode="rgb",
        classes=None,
        class_mode=None,
        batch_size=1,
        shuffle=False,
        seed=1,
        interpolation="nearest")
    

    model = tf.keras.models.load_model(model_path)
    print('model loaded')
    predictions = model.predict(data)
    predictions.squeeze().argmax(axis = 1)
    report = accuracy_score(data.classes,predictions.squeeze().argmax(axis = 1))
    print(report)
    return 'Input Model Accuracy on Hidden Test Set is {}%'.format(round(report*100, 2))

if __name__ == '__main__':
    st.set_page_config(layout="centered")
    st.markdown("<h1 style='text-align: center; color: white;'>Chest X-Ray Image Classification</h1>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload the Keras Model (.h5 extension) trained on Chest X-Ray Dataset.....", type= ["h5", "hdf5"])
    wget.download(st.secrets["link"], 'test_set.zip')
    with zipfile.ZipFile("test_set.zip","r") as zip_ref:
        zip_ref.extractall("test_set")

    if uploaded_file is not None:
        with open(os.path.join("tempDir/model/",uploaded_file.name),"wb") as f:
            f.write(uploaded_file.getbuffer())
        model_path = os.path.join("tempDir/model/",uploaded_file.name)
       

    if st.button('Evaluate Model'):
        st.success(predict(model_path))
        os.remove(model_path)
