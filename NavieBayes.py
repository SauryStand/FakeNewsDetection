#-*- coding:utf-8 -*-
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv('dataset/raw_training.csv')
df_features = pd.concat([df[['label', 'tweetText']]], axis=0)
df_features = df_features.reset_index(drop=True)
print(df_features.head(10))
print(df_features.tail(2))
#y = df_features['label']
countVector = CountVectorizer()
train_count = countVector.fit_transform(df_features['tweetText'].values)
# logR_pipeline = Pipeline([
#         ('LogRCV',countVector),
#         ('LogR_clf',LogisticRegression())
#         ])

X_train, X_test, y_train, y_test = train_test_split(df_features['tweetText'], df_features['label'], test_size=0.33, random_state=53)

count_vectorizer = CountVectorizer(stop_words='english')
count_train = count_vectorizer.fit_transform(X_train)
count_test = count_vectorizer.transform(X_test)

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

mn_count_clf = MultinomialNB(alpha=0.1)
mn_count_clf.fit(count_train, y_train)
pred = mn_count_clf.predict(count_test)
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)

# logR_pipeline.fit(X_train['tweetText'],y_train['label'])
# predicted_LogR = logR_pipeline.predict(X_test['tweetText'])
# np.mean(predicted_LogR == y_test['label'])
