import streamlit as st
from prep_data import df, X_train, X_test, y_train, y_test
from functions import prediction, clf_s

# Add title widget
st.title("Penguin Species Prediction App")  

# Add sidbar title
st.sidebar.title("Penguin Species Prediction Values")

# Add 4 sliders and 2 selectbox.
bill_length_mm = st.sidebar.slider("Bill Length(in mm)", float(df["bill_length_mm"].min()), float(df["bill_length_mm"].max()))
bill_depth_mm = st.sidebar.slider("Bill Depth(in mm)", float(df["bill_depth_mm"].min()), float(df["bill_depth_mm"].max()))
flipper_length_mm = st.sidebar.slider("Flipper Length(in mm)", float(df["flipper_length_mm"].min()), float(df["flipper_length_mm"].max()))
body_mass_g = st.sidebar.slider("Body Mass(in g)", float(df["body_mass_g"].min()), float(df["body_mass_g"].max()))
island = st.sidebar.selectbox('Island', ['Biscoe', 'Dream', 'Torgersen'])
sex = st.sidebar.selectbox('Sex', ['Male', 'Female'])

# Add Classifier selector
clf = st.sidebar.selectbox('Classifier',('Support Vector Machine', 'Logistic Regression', 'Random Forest Classifier'))

# Get values
model, score = clf_s(clf, X_train, y_train)

# When 'Predict' button is pushed, the 'prediction()' function must be called 
# and the value returned by it must be stored in a variable, say 'species_type'. 
# Print the value of 'species_type' and 'score' variable using the 'st.write()' function.
if st.sidebar.button("Predict"):

	# Show given data
	st.markdown('### Given Data')
	st.write('Island:', island)
	st.write('Bill Length:', bill_length_mm, 'mm')
	st.write('Bill Depth:', bill_depth_mm, 'mm')
	st.write('Flipper Length:', flipper_length_mm, 'mm')
	st.write('Body Mass:', body_mass_g, 'gram')
	st.write('Sex:', sex)

	# Show processing values
	st.markdown('### Processing Values')
	st.write('Classifier used:', clf)
	st.write("Accuracy score of this model is:", score)

	# Process Data
	if island == 'Biscoe':
		island = 0
	elif island == 'Dream':
		island = 1
	else:
		island = 2

	if sex == 'Male':
		sex = 0
	else:
		sex = 1

	species_type = prediction(model, island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex)

	# Show result
	st.markdown('### Result')
	st.write("Species predicted:", species_type)
