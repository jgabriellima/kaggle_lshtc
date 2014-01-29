# Very basic script to run initial results for Kaggle's LSHTC.
# https://www.kaggle.com/c/lshtc
# Code still in early testing mode

# SETTINGS
ROOT_DIR = '/Users/ling/lshtc/'

from sklearn.datasets import load_svmlight_file
from sklearn import preprocessing
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier

# train_1000.csv is the first 1000 lines of train
# To preprocess:
# (head -1000 train.csv ) > train_1000.csv
# delete first line (Data)
# Change all comma space to comma :%s/, /,/g
print "Loading data..."
data = load_svmlight_file(ROOT_DIR + "train_1000.csv", multilabel = True)

# Turn labels into binary columns
print "Binarizing labels..."
lb = preprocessing.LabelBinarizer()
lb.fit(data[1])
y_train = lb.fit_transform(data[1])

# Multi-label classifier learning
print "Fitting classifier..."
classif = OneVsRestClassifier(KNeighborsClassifier())
classif.fit(data[0], y_train)

# Testing if plain works for now - predict on training set
print "Classifying..."
prediction = classif.predict(data[0])

# Scoring - need to finish
print "Scoring..."
data[0].toarray()