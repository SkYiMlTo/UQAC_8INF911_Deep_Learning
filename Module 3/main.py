import numpy as np
from scipy.io import arff
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

data = arff.loadarff('RFID_Features_windows5.arff')
df = pd.DataFrame(data[0])

# df = pd.read_csv('csv_result-RFID_Features_windows5.csv')
classes = list(df['class'].array)
# classes = df['classe']
values = df.iloc[:, 1:len(df.keys())-1]

# temp = list(set(classes))
# temp.sort()
# for k in temp:
#     print(k, end=", ")
# print()

X_train, X_test, y_train, y_test = train_test_split(values, classes, test_size=0.7, random_state=0)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# nb_diff = 0
# for i, j in zip(y_test, y_pred):
#     if i != j:
#         nb_diff += 1

# tree = DecisionTreeClassifier(criterion='entropy', max_depth=100)
# tree.fit(X_train_std, y_train)
# y_pred = tree.predict(X_test_std)
# print('Misclassified samples: %d' % (y_test != y_pred).sum())
# print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

# lr = LogisticRegression(C=1000, random_state=0)
# lr.fit(X_train_std, y_train)
# y_pred = lr.predict(X_test_std)
# print('Misclassified samples: %d' % (y_test != y_pred).sum())
# print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

# svm = SVC(kernel="rbf", C=1.0, gamma=2.0, random_state=0)
# svm.fit(X_train_std, y_train)
# y_pred = svm.predict(X_test_std)
# print('Misclassified samples: %d' % (y_test != y_pred).sum())
# print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn.fit(X_train_std, y_train)
y_pred = knn.predict(X_test_std)
print('Misclassified samples: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
