import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.datasets import load_breast_cancer

# Load the trained model from the pickle file
with open('breast_cancer_detector.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit app header with CSS animation
st.markdown(
    """
    <style>
    @keyframes fade-in {
        0% {opacity: 0;}
        100% {opacity: 1;}
    }

    @keyframes netflix-animation {
        0% {opacity: 0; transform: translateY(-20px);}
        100% {opacity: 1; transform: translateY(0);}
    }

    .title-container {
        animation: fade-in 2s ease-in-out;
    }

    .paragraph-container {
        animation: netflix-animation 2s ease-in-out;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Animated title
st.markdown('<h1 class="title-container">Breast Cancer Predictor</h1>', unsafe_allow_html=True)

#Animated title image
st.image('Designer.jpg', width=400)

# Animated paragraph with Netflix animation
st.markdown(
    """
    <div class="paragraph-container">
        <p>
        Breast cancer is one of the most common cancers among women worldwide. 
        Early detection plays a crucial role in improving the chances of successful treatment. 
        This application uses machine learning to predict whether a breast tumor is benign or malignant based on various 
        features. Please enter the relevant information in the sidebar and click the 'Predict' button to get the 
        prediction. You can also just click 'Predict' directly and it generates random values for the input fields if 
        in case filling all the fields is tiresome.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar for user input features with animation
st.sidebar.markdown('<h2 class="title-container">User Input Features</h2>', unsafe_allow_html=True)

# Load the dataset
breast_cancer = load_breast_cancer()
X = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)

# Function to get user input features using text input fields
def user_input_features():
    input_features = {}
    for feature in X.columns:
        min_val = float(X[feature].min())
        max_val = float(X[feature].max())
        default_val = (min_val + max_val) / 2  # Set default value as the midpoint of min and max
        value = st.sidebar.slider(f'Enter {feature}', min_value=min_val, max_value=max_val, value=default_val)
        input_features[feature] = value
    return pd.DataFrame([input_features])

# Get user input features
input_df = user_input_features()

# Predict function
def predict(input_df):
    # Convert input values to float
    input_values = {feature: float(value) for feature, value in input_df.iloc[0].items()}
    input_df_float = pd.DataFrame([input_values], columns=X.columns)

    prediction = model.predict(input_df_float)
    prediction_proba = model.predict_proba(input_df_float)
    return prediction[0], prediction_proba

# Predict button with animation
if st.button('Predict', key='predict_button'):
    # Make predictions
    prediction, prediction_proba = predict(input_df)

    # Display prediction
    st.subheader('Prediction')
    if prediction == 0:
        st.write("The patient result is 'Benign' condition.")
    else:
        st.write("The patient result is 'Malignant' condition.")
