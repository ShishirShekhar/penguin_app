# Import necessary modules
import streamlit as st
from functions import laod_data, get_input_data, prediction, clf_s

# Add title widget
st.title("Penguin Species Prediction App")  

# Add sidbar title
st.sidebar.title("Penguin Species Prediction Values")

# Load the data from the dataset 
df, X, y = laod_data()

# Get input from the user
feature_list = get_input_data(df)

# Get input of model from user
model, score = clf_s(X, y)

# if clicked on predict button precict the value
if st.sidebar.button("Predict"):

	# Show given data
	st.markdown('### Given Data')
	st.write('Island:', feature_list[0])
	st.write('Bill Length:', feature_list[1], 'mm')
	st.write('Bill Depth:', feature_list[2], 'mm')
	st.write('Flipper Length:', feature_list[3], 'mm')
	st.write('Body Mass:', feature_list[4], 'gram')
	st.write('Sex:', feature_list[5])

	# Show processing values
	st.markdown('### Processing Values')
	st.write('Classifier used:', model)
	st.write("Accuracy score of this model is:", score)

	# Process Data
	if feature_list[0] == 'Biscoe':
		feature_list[0] = 0
	elif feature_list[0] == 'Dream':
		feature_list[0] = 1
	else:
		feature_list[0] = 2

	if feature_list[5] == 'Male':
		feature_list[5] = 0
	else:
		feature_list[5] = 1

	species_type = prediction(model, feature_list)

	# Show result
	st.markdown('### Result')
	st.success(f"Species predicted: {species_type}")
