# SETTINGS
ROOT_DIR = '/Users/ling/kaggle/lshtc/'

from sklearn.datasets import load_svmlight_file
from sklearn import preprocessing
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier

data = load_svmlight_file(ROOT_DIR + "train_1000.csv", multilabel = True)

# Turn labels into binary columns
lb = preprocessing.LabelBinarizer()
lb.fit(data[1])
x = lb.fit_transform(data[1])

classif = OneVsRestClassifier(KNeighborsClassifier())
classif.fit(data[0], transformed)
prediction = classif.predict(data[0])

# Scoring
data[0].toarray()