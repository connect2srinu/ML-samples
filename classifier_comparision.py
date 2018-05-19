
from sklearn import tree 
from sklearn import  neighbors
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import random



clf = tree.DecisionTreeClassifier()


# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

# Create test data
x_test=[[random.randint(100,199),random.randint(50,99),random.randint(30,59)] for _ in range(5)]
   

# train them on our data
clf = clf.fit(X, Y)

print("When input:" ,x_test)
prediction = clf.predict(x_test)

# compare their reusults and print the best one!

print("From DecisionTreeClassifier:" ,prediction)
print("From KNeighborsClassifier:" ,neighbors.KNeighborsClassifier().fit(X, Y).predict(x_test))
print("From LogisticRegression:" ,LogisticRegression().fit(X, Y).predict(x_test))
print("From GaussianNB:" ,  GaussianNB().fit(X, Y).predict(x_test))
print("From RandomForestClassifier:" ,  RandomForestClassifier().fit(X, Y).predict(x_test))
print("From AdaBoostClassifier:" ,  AdaBoostClassifier().fit(X, Y).predict(x_test))
print("From QuadraticDiscriminantAnalysis:" ,  QuadraticDiscriminantAnalysis().fit(X, Y).predict(x_test))

