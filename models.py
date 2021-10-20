"""This modules contain different ml model"""

# Import necessary modules
import streamlit as st
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

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