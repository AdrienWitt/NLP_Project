# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 11:49:56 2023

@author: adywi
"""

import pandas as pd
import matplotlib.pyplot as plt
from utils import Data_Preprocessor, Multi_Pipeline
import nltk
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB

data = Data_Preprocessor("train.csv").preproc_data()

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC


# Define the list of vectorizers and classifiers
vectorizer_list = [
    CountVectorizer(),
    TfidfVectorizer(),
    HashingVectorizer(),
    TfidfTransformer(),classifiers = [RandomForestClassifier()]

# Define parameter grids for each classifier
param_grids = [
    {'classifier__n_estimators': [50, 100]}
]


# Create an instance of the VectorizerClassifierGridSearchCV class
grid_search = Multi_Pipeline(data, vectorizer_list, classifiers, param_grids, scoring='accuracy', cv=5)

# Perform the grid search
results_df = grid_search.fit_transform()


final_model = grid_search.fit_final_model(results_df)

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


classifiers_param_grid = [
    (RandomForestClassifier(), {
        'n_estimators': [10, 50, 100],
        'max_depth': [None, 5, 10]
    }),
    (SVC(), {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf']
    }),
    (KNeighborsClassifier(), {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance']
    }),
    (LogisticRegression(), {
        'C': [0.1, 1, 10],
        'penalty': ['l1', 'l2']
    }),
    (DecisionTreeClassifier(), {
        'max_depth': [None, 5, 10]
    }),
    (GaussianNB(), {}),
    (AdaBoostClassifier(), {
        'n_estimators': [50, 100, 200]
    }),
    (GradientBoostingClassifier(), {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7]
    }),
    (MLPClassifier(), {
        'hidden_layer_sizes': [(50, 50), (100, 100)],
        'alpha': [0.0001, 0.001, 0.01]
    }),
    (XGBClassifier(), {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7]
    })
]

from sklearn.svm import LinearSVC, SVC

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
    (XGBClassifier(), {
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


 
## Test BERT

from transformers import BertForSequenceClassification, AdamW, BertConfig,BertTokenizer,get_linear_schedule_with_warmup
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler,random_split
from utils import Data_Preprocessor, Multi_Pipeline, Transformer
import numpy as np

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


data = Data_Preprocessor("train.csv").preproc_data_transformer()


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = 20, # The number of output labels--2 for binary classification.
                    # You can increase this for multi-class tasks.   
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
)

transformer = Transformer(data, model, 512)

transformer.train(epochs = 5)


train_dataloader, validation_dataloader = transformer.data_loader()

for step, batch in enumerate(train_dataloader):
    batch = tuple(t.to(device) for t in batch)
    input_ids, input_mask, labels = batch


