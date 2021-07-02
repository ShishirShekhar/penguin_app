"""This module contains the functions"""

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def prediction(model, island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex):
    """This function returns the preddicted value."""
    species = model.predict([[island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex]])
    species = species[0]
    if species == 0:
        return "'Adelie'"
    elif species == 1:
        return "Chinstrap"
    else:
        return "Gentoo"

def svc_score(X_train, y_train):
    """This function process svc model"""
    svc_model = SVC(kernel = 'linear')
    svc_model.fit(X_train, y_train)
    score = svc_model.score(X_train, y_train)
    return svc_model, score

def rf_score(X_train, y_train):
    """This function process rf_clf model"""
    rf_clf = RandomForestClassifier(n_jobs = -1, n_estimators = 100)
    rf_clf.fit(X_train, y_train)
    score = rf_clf.score(X_train, y_train)
    return rf_clf, score

def lr_score(X_train, y_train):
    """This function process rf_clf model"""
    log_reg = LogisticRegression(n_jobs = -1)
    log_reg.fit(X_train, y_train)
    score = log_reg.score(X_train, y_train)
    return log_reg, score

def clf_s(clf, X_train, y_train):
    if clf == 'Support Vector Machine':
        return svc_score(X_train, y_train)
    elif clf == 'Logistic Regression':
        return lr_score(X_train, y_train)
    else:
        return rf_score(X_train, y_train)
