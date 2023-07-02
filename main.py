from functions import *

data = read_data('bank.csv')

X, Y = preproccessing(data)
X = normalize(X)

X_new, selected_features = selectFeatures(X, Y)
print('selected_features:', selected_features)

k = 5
X_Train, X_Test, Y_Train, Y_Test = split_data(X_new, Y, k)

metric_decision_tree = 0
metric_gaussian = 0
metric_SVM = 0
metric_KNN = 0

for i in range(k):
    print(i+1, '-fold')
    print('===================== decision tree =====================')
    clf = decision_tree(X_Train[i], Y_Train[i])
    metric_decision_tree += test_data(clf, X_Test[i], Y_Test[i], 'decision_tree')

    print('===================== gaussian =====================')
    clf = gaussian(X_Train[i], Y_Train[i])
    metric_gaussian += test_data(clf, X_Test[i], Y_Test[i], 'gaussian')

    print('===================== SVM =====================')
    clf = SVM(X_Train[i], Y_Train[i])
    metric_SVM += test_data(clf, X_Test[i], Y_Test[i], 'SVM')

    print('===================== KNN =====================')
    clf = KNN(X_Train[i], Y_Train[i])
    metric_KNN += test_data(clf, X_Test[i], Y_Test[i], 'KNN')

print('===================== mean metric =====================')
print('metric_decision_tree =', metric_decision_tree / k)
print('metric_gaussian =', metric_gaussian / k)
print('metric_SVM =', metric_SVM / k)
print('metric_KNN =', metric_KNN / k)