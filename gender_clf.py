import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

# hiegth , weigth and shoe size
X = [[180, 78, 43], [170, 70, 41], [176, 80, 42], [160, 60, 42], [182, 85, 44], [160, 60, 38], [174, 76, 41],
     [179, 70, 41], [167, 74, 42], [188, 86, 44], [190, 90, 45]]

y = ['male', 'female', 'male', 'male', 'male', 'female', 'male', 'female', 'female', 'female', 'male']

# pre processing the data, splitting the data into the GenderClassifier and Train parts
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)


# train the models
# decision tree classifier
clf_tree = tree.DecisionTreeClassifier()
clf_tree = clf_tree.fit(X_train, y_train)
# mlp classier
mlp = MLPClassifier()
mlp = mlp.fit(X_train, y_train)
# RBF classifier
kn = KNeighborsClassifier()
kn = kn.fit(X_train, y_train)


# testing by using the test data
y_pred_tree = clf_tree.predict(X_test)
tree_accuracy = accuracy_score(y_test, y_pred_tree)
print('Accuracy for decision tree classifier is: ', tree_accuracy)

y_pred_mlp = mlp.predict(X_test)
mlp_accuracy = accuracy_score(y_test, y_pred_mlp)
print('Accuracy for MLP classifier is: ', mlp_accuracy)

y_pred_kn = kn.predict(X_test)
kn_accuracy = accuracy_score(y_test, y_pred_kn)
print('Accuracy for KNeighbours classifier is: ', kn_accuracy)


# choosing the best classifier using NumPy
max_index = np.argmax([tree_accuracy, mlp_accuracy, kn_accuracy])
classifiers = {0: 'decision tree', 1: 'mlp', 2: 'KNeighbours'}
print('The selected classifier with the highest accuracy is: ', classifiers[max_index])