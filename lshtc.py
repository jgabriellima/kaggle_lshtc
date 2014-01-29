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

data = load_svmlight_file(ROOT_DIR + "train_1000.csv", multilabel = True)

# Turn labels into binary columns
lb = preprocessing.LabelBinarizer()
lb.fit(data[1])
x = lb.fit_transform(data[1])

# Multi-label classifier learning
classif = OneVsRestClassifier(KNeighborsClassifier())
classif.fit(data[0], x)

# Testing if plain works for now - predict on training set
prediction = classif.predict(x)

# Scoring - need to finish
data[0].toarray()