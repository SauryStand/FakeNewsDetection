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

clean.clean_data(df)

#time.struct_time(tm_year=2017, tm_mon=3, tm_mday=16, tm_hour=18, tm_min=35, tm_sec=10, tm_wday=3, tm_yday=75, tm_isdst=0)

print(df['timestamp'].sample(10))

tl = time.localtime(time.time())
print(tl)
print(time.strftime("%Y-%m-%d %H:%M:%S", tl))


#statistic plot useless
# df_authors = df['username'].value_counts()
# df_authors = df_authors.sort_values(ascending=False)
# df_authors_index = list(df_authors.index)
# plot.bar_plot(df_authors.values, df_authors_index, "Most active username", 'r')


df_features = pd.concat([df[['label', 'tweetText']]], axis=0)
df_features = df_features.reset_index(drop=True)
print(df_features.head(10))
print(df_features.tail(2))

df_features["label"] = df_features["label"].map({"real": 0, "fake": 1, "humor": 0})

mask_on_1 = df_features['label'] == 1
mask_on_0 = df_features['label'] == 0
mask_on_2 = df_features['label'] == 0

print(df_features.sample(10))
df_tweetText_1 = df_features[mask_on_1]['tweetText']
df_tweetText_0 = df_features[mask_on_0]['tweetText']
df_tweetText_2 = df_features[mask_on_2]['tweetText']

#print(df_tweetText)
# # Instantiate a CountVectorizer
# # data cleaning
cv1 = CountVectorizer(stop_words = 'english')
cv0 = CountVectorizer(stop_words = 'english')
cv2 = CountVectorizer(stop_words = 'english')
df_tweetText_cvec_1 = cv1.fit_transform(df_tweetText_1)
df_tweetText_cvec_0 = cv0.fit_transform(df_tweetText_0)
df_tweetText_cvec_2 = cv2.fit_transform(df_tweetText_2)

tweet_cvec_df_1 = pd.DataFrame(df_tweetText_cvec_1.toarray(), columns=cv1.get_feature_names())
tweet_cvec_df_0 = pd.DataFrame(df_tweetText_cvec_0.toarray(), columns=cv0.get_feature_names())
tweet_cvec_df_2 = pd.DataFrame(df_tweetText_cvec_2.toarray(), columns=cv2.get_feature_names())
print(tweet_cvec_df_0.shape)

isFakeTweet_wc = tweet_cvec_df_1.sum(axis = 0)
fake_tops = isFakeTweet_wc.sort_values(ascending=False).head(11)
#fake_top_5 = fake_top_5.drop('http')
print('fake tops:\n', fake_tops)
isTrueTweet_wc = tweet_cvec_df_0.sum(axis = 0)
true_tops = isTrueTweet_wc.sort_values(ascending=False).head(11)
#true_top_5 = true_top_5.drop('http')
print('true tops:\n', true_tops)

# most frequent fake news top 5, there was a bug, cannot be plotted
#plot.bar_plot(fake_top_5.values, fake_top_5.index, 'Top 5 unigrams on r/FakeTweet','r')


fake_top_list = set(fake_tops.index)
true_top_list = set(true_tops.index)

# Return common words
common_bigrams = fake_top_list.intersection(true_top_list)
common_unigrams = true_top_list.intersection(fake_top_list)
print(common_bigrams, common_unigrams)
custom = stop_words.ENGLISH_STOP_WORDS
custom = list(custom)
common_unigrams = list(common_unigrams)
common_bigrams = list(common_bigrams)


# Append unigrams to list
for i in common_unigrams:
    custom.append(i)

# Append bigrams to list
for i in common_bigrams:
    split_words = i.split(" ")
    for word in split_words:
        custom.append(word)
print("custom: ", custom)
#
#

#baseline score
scores = df_features['label'].value_counts(normalize=True)
print(scores)
X = df_features['tweetText']
y = df_features['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)

pipe = Pipeline([('cvec', CountVectorizer()), ('lr', LogisticRegression(solver='liblinear'))])# Tune GridSearchCV
pipe_params = {'cvec__stop_words': [None, 'english', custom],
               'cvec__ngram_range': [(1,1), (2,2), (1,3)],
               'lr__C': [0.01, 1]}

gs = GridSearchCV(pipe, param_grid=pipe_params, cv=3)
gs.fit(X_train, y_train)


#cross validation todo
from sklearn.model_selection import cross_val_score
scores = cross_val_score(gs, X, y, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#other models apply todo

print("Best score:", gs.best_score_)
print("Train score", gs.score(X_train, y_train))
print("Test score", gs.score(X_test, y_test))
