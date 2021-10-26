import pandas as pd
import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn import linear_model, preprocessing
import pickle

data = pd.read_csv("car.data")
print(data.head())
print(len(data))


#since many of the csv file data is not in integer format, we are using the preprocessing for converting the data into integer values so as to do the numerical operations easily.

le = preprocessing.LabelEncoder()

#the preprocessing takes in the list and currently we have panda's data frames. So we are updating the columns to repective integer values

buying = le.fit_transform(list(data["buying"]))
maintenance = le.fit_transform(list(data["maintenance"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

# check if the non integer values have been converted into integer values or not

#print(f'buying: {buying}')
#print(f'maintenance: {maintenance}')
#print(f'door: {door}')
#print(f'persons: {persons}')
#print(f'lug_boot: {lug_boot}')
#print(f'safety: {safety}')
#print(f'cls: {cls}')


# define the parameter which we want to predict

predict = "class"

# now that data is in integer form, divide the data into input and output

X = list(zip(buying, maintenance, door, persons, lug_boot, safety))
y = list(cls)

#check if the data is properly got like we wanted

#print(X)

#print(y)

# now the data has been prepared, divide the data into training and testing

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1) #or can use train_size as well


#print the training and testing data or its length to confirm

#print(len(x_test), len(y_test))

best = 0

for _ in range(30):

#define the model
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)
    model = KNeighborsClassifier(n_neighbors=8)

    model.fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)

    print(f'Accuracy: {accuracy}')

    if accuracy > best:
        best = accuracy
        with open("CarClassification.pickle", "wb") as f:
            pickle.dump(model, f)

print(f'Best Accuracy: {best}')

names = ["unacc", "acc", "good", "vgood"]

predicted = model.predict(x_test)

for swi in range(len(x_test)):
    print(f'Actual Value: {names[y_test[swi]]} Predicted: {names[predicted[swi]]}')