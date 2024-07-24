import streamlit as st
import pandas as pd
import joblib

# Load the dataset
data_path = 'cleaned_plant_disease_dataset.csv'
data = pd.read_csv(data_path)

# Load the model
model_path = 'best_random_forest_model.pkl'
model = joblib.load(model_path)

# Streamlit app
st.title('Plant Disease Diagnosis')

# User inputs
disease = st.selectbox('Disease', data['disease'].unique())
plant = st.selectbox('Plant', data['plants'].unique())
region = st.selectbox('Region', data['region'].unique())
severity = st.selectbox('Severity', data['severity'].unique())

# Button to predict
if st.button('Predict'):
    input_data = pd.DataFrame({
        'disease': [disease],
        'plants': [plant],
        'region': [region],
        'severity': [severity]
    })
    
    input_data = pd.get_dummies(input_data)
    input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)
    
    prediction = model.predict(input_data)[0]
    
    result = data[
        (data['disease'] == disease) & 
        (data['plants'] == plant) & 
        (data['region'] == region) & 
        (data['severity'] == severity)
    ]
    
    if not result.empty:
        symptoms = result['symptoms'].values[0]
        description = result['disease_description'].values[0]
        control_measures = result['control_measures'].values[0]
        
        st.subheader('Prediction Results')
        st.write(f'**Symptoms:** {symptoms}')
        st.write(f'**Disease Description:** {description}')
        st.write(f'**Control Measures:** {control_measures}')
    else:
        st.write('No matching records found in the dataset.')
