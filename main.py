# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 18:43:35 2023

@author: wittmann
"""

import pandas as pd
from utils import Data_Preprocessor, Multi_Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


vectorizers = [
    CountVectorizer(),
    TfidfVectorizer(),
    HashingVectorizer()]

classifiers_param_grid = [
    (RandomForestClassifier(), {
        'n_estimators': [10, 50, 100],
        'max_depth': [None, 5, 10]
    }),
    (LinearSVC(), {
        'C': [0.1, 1, 10],
        'penalty': ['l1', 'l2'],
        'dual': [False, True]
    }),
    (SVC(), {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf']
    }),
    (MultinomialNB(), {
        'alpha': [0.1, 0.5, 1.0],
        'fit_prior': [True, False]
    }),
    (LogisticRegression(), {
        'C': [0.1, 1, 10],
        'penalty': ['l1', 'l2']
    }),
    (KNeighborsClassifier(), {
        'n_neighbors': [3, 5, 10],
        'weights': ['uniform', 'distance']
    }),
    (DecisionTreeClassifier(), {
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10]
    }),
    (GradientBoostingClassifier(), {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2]
    }),
    (MLPClassifier(), {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01]
    }),
    (AdaBoostClassifier(), {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2]
    })
]


data = Data_Preprocessor("train.csv").preproc_data()

grid_search = Multi_Pipeline(data, vectorizers, classifiers_param_grid, results_df = None, scoring='accuracy', cv=10)

final_model = grid_search.fit_final_model()



        
        
