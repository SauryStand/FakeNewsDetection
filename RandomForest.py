import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix, f1_score, classification_report

df = pd.read_csv('dataset/raw_training.csv')
df_features = pd.concat([df[['label', 'tweetText']]], axis=0)
df_features = df_features.reset_index(drop=True)
print(df_features.head(10))
print(df_features.tail(2))
countVector = CountVectorizer()
train_count = countVector.fit_transform(df_features['tweetText'].values)

X_train, X_test, y_train, y_test = train_test_split(df_features['tweetText'], df_features['label'], test_size=0.28, random_state=53)
print(X_train)
random_forest = Pipeline([
    ('rfCV' ,countVector),
    ('rf_clf' ,RandomForestClassifier(n_estimators=200 ,n_jobs=3))
])

random_forest.fit(X_train,y_train)
predicted_rf = random_forest.predict(X_test)
np.mean(predicted_rf == y_test)

score = metrics.accuracy_score(y_test, predicted_rf)
print("accuracy:   %0.3f" % score)

# score = f1_score(y_test, predicted_rf)
# print("F1-score:", score)