# Income predictions
#This analysis predicts whether an individual makes more that $50000 using Census Data

#Importing the libraries
import pandas as pd
import numpy as py
import matplotlib.pyplot as plt

#Import dataset
dataset = pd.read_csv('census_income.csv')
x = dataset.iloc[:, [0, 1, 3, 6, 11]].values
y = dataset.iloc[:, 14].values

#Coding categorical data
#Encoding x
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x_1 = LabelEncoder()
x[:, 1] = labelencoder_x_1.fit_transform(x[:, 1])
labelencoder_x_2 = LabelEncoder()
x[:, 2] = labelencoder_x_2.fit_transform(x[:, 2])
labelencoder_x_3 = LabelEncoder()
x[:, 3] = labelencoder_x_3.fit_transform(x[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [4])
x = onehotencoder.fit_transform(x).toarray()
x = x[:, 0:]
#Encodeing y
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#Splitting dataset into training and test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0) 

#Fitting the Random Forest classification to the training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 300, criterion = 'entropy', random_state = 0)
classifier.fit(x_train, y_train)

#Predicting the test set results
y_pred = classifier.predict(x_test)

#Making the confusion matrix - tp predict accuracy of model
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
