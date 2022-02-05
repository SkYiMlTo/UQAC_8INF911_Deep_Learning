from Algorithms import Algorithms


def main():
    algo = Algorithms('RFID_Features_windows5.arff', 'class')
    algo.exec_logistic_regression()
    algo.exec_c_support_vector()
    algo.exec_decision_tree()
    algo.exec_knn()


if __name__ == '__main__':
    main()
