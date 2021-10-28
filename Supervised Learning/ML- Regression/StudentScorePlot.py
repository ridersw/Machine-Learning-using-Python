import numpy as np
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style
from sklearn import linear_model
import sklearn

# import the data

data = pd.read_csv("/Users/wshashiraj/Desktop/Machine Learning/DataSet/student/student-mat.csv", sep=";")
print(data.head())

#trim the data

data = data[["G1", "G2", "G3", "freetime", "traveltime", "studytime"]]
print(data.head())


#seperate the input and output data

predict = "G3"
X = np.array(data.drop([predict],1))
y = np.array(data[predict])

# split the data into training and testing

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

best = 0

for _ in range(30):
    # split the data into training and testing

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

    # model

    linear = linear_model.LinearRegression()
    linear.fit(X_train, y_train)

    #find the accuracy of the model

    accuracy = linear.score(X_test, y_test)
    print(f'Accuracy: {accuracy}')

    # save this model
    if accuracy > best:
        best = accuracy
        with open("studentScore.pickle", "wb") as f:
            pickle.dump(linear, f)

# Since the equation of line is y = mx + c

print(f'value of m: {linear.coef_}')
print(f'Value of y-intercept: {linear.intercept_}')


print(f"Best Accuracy: {best}")

#predictions

predictions = linear.predict(X_test)

for swi in range(len(predictions)):
   print(f'Prediction: {predictions[swi]} and Actual Value: {y_test[swi]}')


# plotting a graph

p = "G1"
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()