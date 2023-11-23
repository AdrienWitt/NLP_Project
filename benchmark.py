# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 10:16:48 2023

@author: adywi
"""

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from utils import Data_Preprocessor, Multi_Pipeline, predict_test
import os


os.chdir(r"C:\Users\wittmann\OneDrive - unige.ch\Documents\Courses\MACHINE LEARNING\Project")

data = Data_Preprocessor("train.csv").preproc_data()
X, y = data["post"], data["tags"]

# Create a pipeline with a TfidfVectorizer and an SVM classifier
pipeline = make_pipeline(CountVectorizer(), SVC(kernel='linear'))

# Define the hyperparameters to search through
param_grid = {'svc__kernel': ['linear', 'rbf'],
    'svc__C': [0.2, 0.5, 1, 4, 6, 8, 10]
}

# Use GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(pipeline, param_grid, cv=10)
grid_search.fit(X, y)


print(grid_search.best_params_)


best_model = grid_search.best_estimator_
best_model.fit(X, y)


test = Data_Preprocessor("test.csv").preproc_data()

y_pred = best_model.predict(test["post"])  
solution = test
solution.insert(2, "tags", y_pred)
solution = solution[["Id", "tags"]]
solution.to_csv("solution_svm.csv", index = False)

