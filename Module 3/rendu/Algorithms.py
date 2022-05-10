from scipy.io import arff
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle

from timing import timing


class Algorithms:
    def __init__(self, input_file, key_classes):
        self.temp = 0
        if input_file.endswith('.arff'):
            data = arff.loadarff(input_file)
            self.__data_frame = pd.DataFrame(data[0])
        elif input_file.endswith('.csv'):
            self.__data_frame = pd.read_csv(input_file)
        else:
            print("Format du fichier d'input non supporté. (uniquement .arff ou .csv)")
            return

        self.__data_frame = shuffle(self.__data_frame)

        x = self.__data_frame.iloc[:, 1:len(self.__data_frame.keys())-1]
        y = list(self.__data_frame[key_classes].array)

        self.__print_classes(y)

        X_train, X_test, self.__y_train, self.__y_test = train_test_split(x, y, test_size=0.3, random_state=0)
        sc = StandardScaler()
        sc.fit(X_train)
        self.__X_train_std = sc.transform(X_train)
        self.__X_test_std = sc.transform(X_test)

    def exec_logistic_regression(self):
        lr = LogisticRegression(C=1000, random_state=0)

        @timing
        def inner_func():
            lr.fit(self.__X_train_std, self.__y_train)
            return lr.predict(self.__X_test_std)
        self.__print_results(inner_func(), "logistic regression")

    def exec_c_support_vector(self):
        # gamma -> Impact sur over fitting
        # kernel -> linear or rbf
        # C 100 | gamma 1 | precision 0.95 | mal classes 692
        svm = SVC(kernel="linear", C=100.0, gamma=1.0, random_state=0)

        @timing
        def inner_func():
            svm.fit(self.__X_train_std, self.__y_train)
            return svm.predict(self.__X_test_std)
        self.__print_results(inner_func(), "C-support vector")

    def exec_decision_tree(self):
        tree = DecisionTreeClassifier(criterion='entropy', max_depth=100)

        @timing
        def inner_func():
            tree.fit(self.__X_train_std, self.__y_train)
            return tree.predict(self.__X_test_std)
        self.__print_results(inner_func(), "decision tree")

    def exec_knn(self):
        knn = KNeighborsClassifier(n_neighbors=5, p=1, metric='minkowski')  # p 1 | n 8 | res 0.75

        @timing
        def inner_func():
            knn.fit(self.__X_train_std, self.__y_train)
            return knn.predict(self.__X_test_std)
        self.__print_results(inner_func(), "knn")

    def __print_results(self, y_pred, name_algo):
        print("\nResults for " + name_algo + " : ")
        print('Misclassified samples: %d' % (self.__y_test != y_pred).sum())
        print('Accuracy: %.2f' % accuracy_score(self.__y_test, y_pred))
        print("")

    def __print_classes(self, classes):
        print("Liste des classes :")
        temp = sorted(set(classes))
        for k in temp:
            print(k.decode("utf-8"), end=", ")
        print('\n')
