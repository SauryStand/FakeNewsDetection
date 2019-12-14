#-*- coding:utf-8 -*-
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  LogisticRegression

df = pd.read_csv('dataset/raw_training.csv')
df_features = pd.concat([df[['label', 'tweetText']]], axis=0)
df_features = df_features.reset_index(drop=True)
print(df_features.head(10))
print(df_features.tail(2))
y = df_features['label']
countVector = CountVectorizer()
train_count = countVector.fit_transform(df_features['tweetText'].values)
logR_pipeline = Pipeline([
        ('LogRCV',countVector),
        ('LogR_clf',LogisticRegression())
        ])

X_train, X_test, y_train, y_test = train_test_split(df_features['tweetText'], y, test_size=0.33, random_state=53)

count_vectorizer = CountVectorizer(stop_words='english')
count_train = count_vectorizer.fit_transform(X_train)
count_test = count_vectorizer.transform(X_test)