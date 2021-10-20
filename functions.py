"""This module contains the functions"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

@st.cache()
def laod_data():
    """This function loads the data and return preprocessed data."""
    # Loading the dataset.
    csv_file = 'penguin.csv'
    df = pd.read_csv(csv_file)

    # Drop the NAN values
    df = df.dropna()

    # Add numeric column 'label' to resemble non numeric column 'species'
    df['label'] = df['species'].map({'Adelie': 0, 'Chinstrap': 1, 'Gentoo':2})


    # Convert the non-numeric column 'sex' to numeric in the DataFrame
    df['sex'] = df['sex'].map({'Male':0,'Female':1})

    # Convert the non-numeric column 'island' to numeric in the DataFrame
    df['island'] = df['island'].map({'Biscoe': 0, 'Dream': 1, 'Torgersen':2})


    # Create X and y variables
    X = df[['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]
    y = df['label']

    return df, X, y


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


@st.cache()
def svc_score(X, y):
    """This function process svc model"""
    svc_model = SVC(kernel = 'linear')
    svc_model.fit(X, y)
    score = svc_model.score(X, y)
    return svc_model, score


@st.cache()
def rf_score(X, y):
    """This function process rf_clf model"""
    rf_clf = RandomForestClassifier(n_jobs = -1, n_estimators = 100)
    rf_clf.fit(X, y)
    score = rf_clf.score(X, y)
    return rf_clf, score


@st.cache()
def lr_score(X, y):
    """This function process rf_clf model"""
    log_reg = LogisticRegression(n_jobs = -1)
    log_reg.fit(X, y)
    score = log_reg.score(X, y)
    return log_reg, score


def clf_s(X, y):
    """
    This take the input of ml model user want to use
    and return model
    """
    # Add Classifier selector
    clf = st.sidebar.selectbox('Classifier', 
                                ('Support Vector Machine', 
                                'Logistic Regression', 
                                'Random Forest Classifier')
    )
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

