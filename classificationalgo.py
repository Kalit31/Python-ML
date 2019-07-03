import pandas as pd
import matplotlib.pyplot as plt

# store data in fruits
fruits = pd.read_table('fruit_data.txt')

# print some data
print(fruits.head())

# size of data
print(fruits.shape)

# number of uniques fruits
print(fruits['fruit_name'].unique())

# quantity of each unique fruit
print(fruits.groupby('fruit_name').size())

import seaborn as sns

# bar graph of various fruit and their count
sns.countplot(fruits['fruit_name'], label="Count")
plt.savefig('graph_fruit_count')

# box plot: detect outliers
fruits.drop('fruit_label', axis=1).plot(kind='box', subplots=True, layout=(2, 2), sharex=False, figsize=(9, 9))
plt.savefig('fruits_box')

import pylab as pl

# histogram of various features

fruits.drop('fruit_label', axis=1).hist(bins=30, figsize=(9, 9))
pl.suptitle("Historam for each numeric input variable")
plt.savefig('fruits_hist')

# spliting data
feature_names = ['mass', 'width', 'height', 'color_score']
x = fruits[feature_names]
y = fruits['fruit_label']

# data slicing

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

# scale data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

# logistic regression

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_train, y_train)

print('Accuracy of Logistic regression classifier on training set: {:.2f}'.format(logreg.score(x_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'.format(logreg.score(x_test, y_test)))
print()

# Decision Tree

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier().fit(x_train,y_train)
print('Accuracy of Decision tree classifier on training set: {:.2f}'.format(clf.score(x_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'.format(clf.score(x_test, y_test)))
print()

# KNN Classifier

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier().fit(x_train,y_train)
print('Accuracy of K-NN classifier on training set: {:.2f}'.format(knn.score(x_train, y_train)))
print('Accuracy of K-NN classifer on test set: {:.2f}'.format(knn.score(x_test, y_test)))
print()


# Naive Bayes

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB().fit(x_train, y_train)
print('Accuracy of GNB  classifier on training set: {:.2f}'.format(gnb.score(x_train, y_train)))
print('Accuracy of GNB classifier on test set: {:.2f}'.format(gnb.score(x_test, y_test)))
print()


# Support Vector Machine Classifier

from sklearn.svm import SVC

svm = SVC().fit(x_train, y_train)
print('Accuracy of SVM classifier on training set: {:.2f}'.format(svm.score(x_train, y_train)))
print('Accuracy of SVM classifier on test set: {:.2f}'.format(svm.score(x_test, y_test)))
print()


# Confusion matrix for knn classifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

pred = knn.predict(x_test)
print(confusion_matrix(y_test,pred))
print(classification_report(y_test, pred))
