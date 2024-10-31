import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import streamlit as st
from sklearn.preprocessing import StandardScaler


model = load_model('thyroid_cancer_model.h5')


label_encoders = {}
categorical_columns = ['Gender', 'Smoking', 'Hx Smoking', 'Hx Radiotherapy',
                       'Thyroid Function', 'Physical Examination',
                       'Adenopathy', 'Pathology', 'Focality',
                       'Risk', 'T', 'N', 'M', 'Stage', 'Response']

for column in categorical_columns:
    with open(f'{column}_encoder.pkl', 'rb') as file:
        label_encoders[column] = pickle.load(file)


scaler_mean = np.load('scaler_mean.npy')
scaler_scale = np.load('scaler_scale.npy')
scaler = StandardScaler()
scaler.mean_ = scaler_mean
scaler.scale_ = scaler_scale


st.title('Thyroid Cancer Recurrence Prediction')
st.write('Enter patient details to predict recurrence.')

age = st.number_input('Age:', min_value=0)
gender = st.selectbox('Gender:', ['M', 'F'])
smoking = st.selectbox('Smoking:', ['Yes', 'No'])
hx_smoking = st.selectbox('Hx Smoking:', ['Yes', 'No'])
hx_radiotherapy = st.selectbox('Hx Radiotherapy:', ['Yes', 'No'])
thyroid_function = st.selectbox('Thyroid Function:', ['Euthyroid', 'Subclinical Hypothyroidism', 'Clinical Hyperthyroidism'])
physical_exam = st.selectbox('Physical Examination:', ['Single nodular goiter-right', 'Multinodular goiter', 'Normal', 'Diffuse goiter'])
adenopathy = st.selectbox('Adenopathy:', ['No', 'Right', 'Left', 'Bilateral', 'Extensive', 'Posterior'])
pathology = st.selectbox('Pathology:', ['Micropapillary', 'Papillary', 'Follicular', 'Hurthel cell'])
focality = st.selectbox('Focality:', ['Uni-Focal', 'Multi-Focal'])
risk = st.selectbox('Risk:', ['Intermediate', 'Low', 'High'])
tumor_classification = st.selectbox('Tumor Classification (T):', ['T1a', 'T1b', 'T2', 'T3a', 'T3b', 'T4a', 'T4b'])
nodal_classification = st.selectbox('Nodal Classification (N):', ['N0', 'N1b', 'N1a'])
metastasis_classification = st.selectbox('Metastasis Classification (M):', ['M0', 'M1'])
stage = st.selectbox('Stage:', ['I', 'II', 'III', 'IVB', 'IVA'])
response = st.selectbox('Response:', ['Structural Incomplete', 'Biochemical Incomplete', 'Indeterminate', 'Excellent'])


input_data = np.array([[age,
                         label_encoders['Gender'].transform([gender])[0],
                         label_encoders['Smoking'].transform([smoking])[0],
                         label_encoders['Hx Smoking'].transform([hx_smoking])[0],
                         label_encoders['Hx Radiotherapy'].transform([hx_radiotherapy])[0],
                         label_encoders['Thyroid Function'].transform([thyroid_function])[0],
                         label_encoders['Physical Examination'].transform([physical_exam])[0],
                         label_encoders['Adenopathy'].transform([adenopathy])[0],
                         label_encoders['Pathology'].transform([pathology])[0],
                         label_encoders['Focality'].transform([focality])[0],
                         label_encoders['Risk'].transform([risk])[0],
                         label_encoders['T'].transform([tumor_classification])[0],
                         label_encoders['N'].transform([nodal_classification])[0],
                         label_encoders['M'].transform([metastasis_classification])[0],
                         label_encoders['Stage'].transform([stage])[0],
                         label_encoders['Response'].transform([response])[0]]])

input_data = scaler.transform(input_data)


if st.button('Predict'):
    prediction = model.predict(input_data)
    result = 'Yes' if prediction[0][0] > 0.5 else 'No'
    st.write(f'The prediction for cancer recurrence is: {result}')
