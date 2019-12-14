#-*- coding:utf-8 -*-
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix, f1_score, classification_report
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
#
#
# def build_confusion_matrix(classifier):
#     k_fold = KFold(n_splits=5)
#     scores = []
#     confusion = np.array([[0, 0], [0, 0]])
#
#     for train_ind, test_ind in k_fold.split(X_train):
#         train_text = X_train.iloc[train_ind]['tweetText']
#         train_y = y_train.iloc[train_ind]['label']
#
#         test_text = X_test.iloc[test_ind]['tweetText']
#         test_y = y_test.iloc[test_ind]['label']
#
#         classifier.fit(train_text, train_y)
#         predictions = classifier.predict(test_text)
#
#         confusion += confusion_matrix(test_y, predictions)
#         score = f1_score(test_y, predictions)
#         scores.append(score)
#
#     return (print('Total statements classified:', len(df_features)),
#             print('Score:', sum(scores) / len(scores)),
#             print('score length', len(scores)),
#             print('Confusion matrix:'),
#             print(confusion))
#
# build_confusion_matrix(logR_pipeline)