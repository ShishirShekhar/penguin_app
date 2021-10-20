"""This module contains the functions"""

# Import necessary modules
import streamlit as st
from models import svc_score, lr_score, rf_score


def get_input_data(df):
    """This function take input from the user and return that input"""
    # Add 4 sliders and 2 selectbox.
    bill_length_mm = st.sidebar.slider("Bill Length(in mm)", float(df["bill_length_mm"].min()), float(df["bill_length_mm"].max()))
    bill_depth_mm = st.sidebar.slider("Bill Depth(in mm)", float(df["bill_depth_mm"].min()), float(df["bill_depth_mm"].max()))
    flipper_length_mm = st.sidebar.slider("Flipper Length(in mm)", float(df["flipper_length_mm"].min()), float(df["flipper_length_mm"].max()))
    body_mass_g = st.sidebar.slider("Body Mass(in g)", float(df["body_mass_g"].min()), float(df["body_mass_g"].max()))
    island = st.sidebar.selectbox('Island', ['Biscoe', 'Dream', 'Torgersen'])
    sex = st.sidebar.selectbox('Sex', ['Male', 'Female'])

    return [island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex]


def model_selector(clf, X, y):
    """
    This take the input of ml model user want to use
    and return model
    """
    if clf == 'Support Vector Machine':
        return svc_score(X, y)
    elif clf == 'Logistic Regression':
        return lr_score(X, y)
    else:
        return rf_score(X, y)


def prediction(model, feature_list):
    """This function returns the preddicted value."""
    species = model.predict([feature_list])
    species = species[0]
    if species == 0:
        return "'Adelie'"
    elif species == 1:
        return "Chinstrap"
    else:
        return "Gentoo"

def prediction_button(feature_list, model, score):
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