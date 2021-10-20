# Import necessary modules
import streamlit as st
from preprocessing import laod_data
from functions import get_input_data, model_selector, prediction_button

# Configure the web page.
st.set_page_config(
    page_title = 'Penguin Species Prediction',
    page_icon = 'penguin',
    layout = 'centered',
    initial_sidebar_state = 'auto'
)

# Add title widget
st.title("Penguin Species Prediction App")  

# Add sidbar title
st.sidebar.title("Penguin Species Prediction Values")

# Load the data from the dataset 
df, X, y = laod_data()

# Get input from the user
feature_list = get_input_data(df)

# Add Classifier selector, to get model type input from the user
clf = st.sidebar.selectbox('Classifier', ('Support Vector Machine', 'Logistic Regression', 'Random Forest Classifier'))

# Get model and model score according selected model
model, score = model_selector(clf, X, y)

# if clicked on predict button precict the value
if st.sidebar.button("Predict"):
	prediction_button(feature_list, model, score)
else:
	cols = st.columns([1, 6, 1])
	with cols[1]:
		st.image("./images/penguin.png")