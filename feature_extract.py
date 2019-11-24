# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import time
from tools import plot
from sklearn.feature_extraction import stop_words
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tools import clean_data as clean
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix


df = pd.read_csv('dataset/raw_training.csv')
N, p = df.shape
print(df.shape)

#df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
for index, row in df.iterrows():
    if row['username'] == '' or row['username'] == None:
        row['username'] = 'NaN'