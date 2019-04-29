import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

# classes_dict = {}

# заполняем dict классами (названия директорий)
# def fill_classes(input_dir):
#     for root, subdirs, files in os.walk(input_dir):
#         for subdir in subdirs:
#             classes_dict[subdir] = subdir
#
# input_txt = '/home/dr/Documents/train/actual_train'
#
# fill_classes(input_txt)

col_types = {'filename': str, 'data': str, 'target': str}
processed_data = pd.read_csv('data.csv')

print(processed_data.info())
# y = processed_data.target
# X = processed_data.drop('target', axis=1)
# processed_data.info()

#
test_data = processed_data[:300]
y = test_data.target
X = test_data.drop('target', axis=1)
# test_data.info()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print('MultinomialNB:')
multi_NB_pipe = Pipeline([('count_vect', CountVectorizer(ngram_range=(1, 4))),
                          ('tfidf', TfidfTransformer()),
                          ('multi_naive', MultinomialNB()), ])

multi_NB_pipe = multi_NB_pipe.fit(X_train.data, y_train)

multi_NB_predicted = multi_NB_pipe.predict(X_test.data)

# for doc, category_id in zip(X_test.filename, multi_NB_predicted):
#     print('%r => %s' % (doc, target_names_ext[category_id]))


print(np.mean(multi_NB_predicted == y_test))

# print(metrics.classification_report(y_test, multi_NB_predicted, target_names=target_names_ext))
# metrics.confusion_matrix(y_test, multi_NB_predicted)

print('SGD:')
sgd_pipe = Pipeline([('count_vect', CountVectorizer(ngram_range=(1, 2))),
                     ('tfidf', TfidfTransformer()),
                     ('sgd', SGDClassifier(loss='hinge', penalty='l2',
                                           alpha=1e-3, max_iter=1000, tol=1e-3, random_state=42)), ])

sgd_pipe = sgd_pipe.fit(X_train.data, y_train)
sgd_predicted = sgd_pipe.predict(X_test.data)
print(np.mean(sgd_predicted == y_test))

# for doc, category in zip(X_test.filename, sgd_pipe_predicted):
#     print('%r => %s' % (doc, target_names[category]))


# print(metrics.classification_report(y_test, sgd_predicted, target_names=target_names))
# metrics.confusion_matrix(y_test, sgd_predicted)

# 5555555555555555555555555555555
#
# rf_pipe = Pipeline([('count_vec', CountVectorizer(ngram_range=(1, 2))),
#                     ('tfidf', TfidfTransformer()),
#                     ('rf', RandomForestClassifier(n_estimators=100)), ])
# rf_pipe = rf_pipe.fit(X_train.data, y_train)
#
# rf_predicted = rf_pipe.predict(X_test.data)
# print(np.mean(rf_predicted == y_test))
# 555555555555555555555555555555
#
#
# print(metrics.classification_report(y_test, rf_predicted, target_names=target_names))
# metrics.confusion_matrix(y_test, rf_predicted)


# def iter_minibatches(hs, chunksize):
#     numtrainingpoints = 15429
#     chunkstartmarker = 0
#     while chunkstartmarker < numtrainingpoints:
#         X_chunk = hs[chunkstartmarker:chunkstartmarker+chunksize]
#         y_chunk = y_train[chunkstartmarker:chunkstartmarker+chunksize]
#
#         chunkstartmarker += chunksize
#         yield X_chunk, y_chunk


# def getrows(r):
#     chunk = X_train[r]
#     y_chunk = chunk.target
#     X_chunk = chunk.drop('target', axis=1) 
#     yield X_chunk, y_chunk


# sgd = SGDClassifier(loss='hinge', shuffle=True, penalty='l2',
#                                          alpha=1e-3, max_iter=1000, tol=1e-3, random_state=42)
#
# vectorizer = HashingVectorizer()
# count_vect = CountVectorizer()
# X_train_counts = count_vect.fit_transform(X_train.data)
# tfidf_transformer = TfidfTransformer()
# X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# hs = vectorizer.fit_transform(X_train.data)

# batcherator = iter_minibatches(hs, chunksize=100)


# classes = np.unique(list(classes_dict.values()))
# for X_chunk, y_chunk in batcherator:
#     if X_chunk.size == None:
#         print(X_chunk)
#     else:
#         try:
#             sgd = sgd.partial_fit(X_chunk, y_chunk, classes=classes)
#         except Exception:
#             print(X_chunk)
#


# sgd.predict([X_train.data])
