import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

cancer = datasets.load_breast_cancer()

# print(cancer.feature_names)
# print(cancer.target_names)

x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(cancer.data, cancer.target, test_size = 0.1)

classes = ['malignant' 'benign']

# Optimization by giving parameters in SVC

#using kernel = linear

clf = svm.SVC(kernel="linear")

clf.fit(x_train, y_train)

prediction = clf.predict(x_test)

accuracy = metrics.accuracy_score(y_test, prediction)

print(f'Linear accuracy: {accuracy}')

#using kernel = poly

clf = svm.SVC(kernel="poly")

#svm- support vector machine
#SVC- support vector classifier

clf.fit(x_train, y_train)

prediction = clf.predict(x_test)

accuracy = metrics.accuracy_score(y_test, prediction)

print(f'poly accuracy: {accuracy}')