import streamlit as st
import pickle
from skimage.feature import graycomatrix, graycoprops
import cv2
import pandas as pd
import numpy as np

# Load model
filename = "finalized_model.sav"
model = pickle.load(open(filename, 'rb'))

def calc_glcm_all_agls(img, props, dists=[1], agls=[0, np.pi/4, np.pi/2, 3*np.pi/4], lvl=256, sym=True, norm=True):

    glcm = graycomatrix(img,
                        distances=dists,
                        angles=agls,
                        levels=lvl,
                        symmetric=sym,
                        normed=norm)
    feature = []
    glcm_props = [propery for name in props for propery in graycoprops(glcm, name)[0]]
    for item in glcm_props:
            feature.append(item)

    return feature

def predict_animal(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resize = cv2.resize(gray, (150,150))

    # ----------------- call calc_glcm_all_agls() for all properties ----------------------------------
    properties = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']

    glcm_all_agls = []
    glcm_all_agls.append(
            calc_glcm_all_agls(resize,
                                props=properties)
                            )

    columns = []
    angles = ['0', '45', '90','135']
    for name in properties :
        for ang in angles:
            columns.append(name + "_" + ang)

    glcm_df = pd.DataFrame(glcm_all_agls,
                        columns = columns)

    # glcm_df.head()

    X = glcm_df
    pred = model.predict(X)

    if pred == ['Elephant']:
        st.write("Model detect Elephant")
    else:
        st.write("Model detect Zebra")

st.title('Animal Detection')

uploaded_images = st.file_uploader('Upload images', type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

st.write('Image preview')
for image in uploaded_images:
    st.write(image.name)
    st.image(image)

    # Convert uploaded image to bytes
    file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    predict_animal(img)

    # with open(image.name,'wb') as f:
    #     f.write(image.read())
    #     img = cv2.imread(image.name)
    #     predict_animal(img)