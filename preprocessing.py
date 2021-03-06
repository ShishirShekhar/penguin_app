# Import necessary modules
import pandas as pd
import streamlit as st


@st.cache()
def load_data():
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
