import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

df = pd.read_csv('emails.csv')
df

df.shape

df.isnull().any()

df.drop(columns='Email No.', inplace=True)
df

df.columns

df.Prediction.unique()

df['Prediction'] = df['Prediction'].replace({0:'Not spam', 1:'Spam'})

df

"""# KNN"""

X = df.drop(columns='Prediction',axis = 1)
Y = df['Prediction']

X.columns

Y.head()

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

KN = KNeighborsClassifier
knn = KN(n_neighbors=7)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)

print("Prediction: \n")
print(y_pred)

# Accuracy

M = metrics.accuracy_score(y_test,y_pred)
print("KNN accuracy: ", M)

C = metrics.confusion_matrix(y_test,y_pred)
print("Confusion matrix: ", C)

"""# SVM Classifier"""

model = SVC(C = 1)   # cost C = 1

model.fit(x_train, y_train)

y_pred = model.predict(x_test)      # predict

kc = metrics.confusion_matrix(y_test, y_pred)
print("SVM accuracy: ", kc)
