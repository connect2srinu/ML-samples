
from sklearn import tree 
from sklearn import  neighbors
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score
import random
import numpy as np



clf = tree.DecisionTreeClassifier()


# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

# Create test data
x_test=[[random.randint(100,199),random.randint(50,99),random.randint(30,59)] for _ in range(11)]

# x_test = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
#      [190, 90, 47], [175, 64, 39],
#      [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]
   

# train them on our data
clf = clf.fit(X, Y)

print("When input:" ,x_test)

# compare their reusults and print the best one!
pred_tree = clf.predict(x_test)
acc_tree = accuracy_score(Y, pred_tree) * 100

pred_knei = neighbors.KNeighborsClassifier().fit(X, Y).predict(x_test)
acc_knei = accuracy_score(Y, pred_knei) * 100
pred_lg = LogisticRegression().fit(X, Y).predict(x_test)
acc_lg = accuracy_score(Y, pred_lg) * 100
pred_GNB = GaussianNB().fit(X, Y).predict(x_test)
acc_GNB = accuracy_score(Y, pred_GNB) * 100
pred_RFC = RandomForestClassifier().fit(X, Y).predict(x_test)
acc_RFC = accuracy_score(Y, pred_RFC) * 100
pred_ABC = AdaBoostClassifier().fit(X, Y).predict(x_test)
acc_ABC = accuracy_score(Y, pred_ABC) * 100
pred_QD = QuadraticDiscriminantAnalysis().fit(X, Y).predict(x_test)
acc_QD = accuracy_score(Y, pred_QD) * 100


print("From DecisionTreeClassifier:" ,pred_tree)
print("From KNeighborsClassifier:" ,pred_knei)
print("From LogisticRegression:" ,pred_lg)
print("From GaussianNB:" ,  pred_GNB)
print("From RandomForestClassifier:" ,  pred_RFC)
print("From AdaBoostClassifier:" ,  pred_ABC)
print("From QuadraticDiscriminantAnalysis:" ,  pred_QD)

print('Accuracy for DecisionTree: {}'.format(acc_tree))
print('Accuracy for KNeighborsClassifier: {}'.format(acc_knei))
print('Accuracy for LogisticRegression: {}'.format(acc_lg))
print('Accuracy for GaussianNB: {}'.format(acc_GNB))
print('Accuracy for RandomForestClassifier: {}'.format(acc_RFC))
print('Accuracy for AdaBoostClassifier: {}'.format(acc_ABC))
print('Accuracy for QuadraticDiscriminantAnalysis: {}'.format(acc_QD))


# The best classifier from svm, per, KNN
index = np.argmax([acc_tree, acc_knei, acc_lg, acc_GNB, acc_RFC, acc_ABC, acc_QD])
classifiers = {0: 'Tree', 1: 'KNEI', 2: 'LG', 3: 'GNB', 4: 'RFC', 5: 'ABC', 6: 'QD'}
print('Best gender classifier is {}'.format(classifiers[index]))
