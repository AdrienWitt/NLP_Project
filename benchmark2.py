# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 17:12:14 2023

@author: wittmann
"""

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from utils import Data_Preprocessor, Multi_Pipeline

# Define the pipeline with CountVectorizer, TfidfTransformer, and MLPClassifier
text_clf_nn = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MLPClassifier(max_iter=100)),
])

# Set up the parameters for the grid search
parameters_nn = {
    'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'vect__min_df': [1, 2, 3],
    'vect__stop_words': [None, 'english'],
    'tfidf__use_idf': [True, False],
    'clf__hidden_layer_sizes': [(50,), (100,), (50, 50)],
    'clf__activation': ['relu', 'tanh'],
    'clf__alpha': [0.0001, 0.001, 0.01],
}

# Create the grid search with cross-validation
grid_search_nn = GridSearchCV(text_clf_nn, parameters_nn, cv=5, scoring='accuracy')

data = Data_Preprocessor("train.csv").preproc_data()
X, y = data["post"], data["tags"]



# Fit the grid search to the data
grid_search_nn.fit(X, y)

# Print the best parameters and the corresponding performance
print("Best parameters found: ", grid_search_nn.best_params_)
print("Best cross-validation score: {:.2f}".format(grid_search_nn.best_score_))

print(grid_search_nn.best_params_)


best_model = grid_search_nn.best_estimator_
best_model.fit(X, y)

# print(grid_search.best_params_)
# {'svc__C': 1, 'tfidfvectorizer__max_features': 10000}

test = Data_Preprocessor("test.csv").preproc_data()

y_pred = best_model.predict(test["post"])  
solution = test
solution.insert(2, "tags", y_pred)
solution = solution[["Id", "tags"]]
solution.to_csv("solution_2.csv", index = False)

