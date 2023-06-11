from functions import *


data = read_data('bank.csv')

X, Y = preproccessing(data)
X = normalize(X)
# print(X)
# print(Y)

X_new, selected_features = selectFeatures(X, Y)
# print(X_new)
# print(selected_features)

X_Train, X_Test, Y_Train, Y_Test = split_data(X_new, Y, 70)


print('===================== decision tree =====================')
clf = decision_tree(X_Train, Y_Train)
test_data(clf, X_Test, Y_Test, 'decision_tree')

# print('===================== gaussian =====================')
# clf = gaussian(X_Train, Y_Train)
# test_data(clf, X_Test, Y_Test, 'gaussian')
#
# print('===================== SVM =====================')
# clf = SVM(X_Train, Y_Train)
# test_data(clf, X_Test, Y_Test, 'SVM')
#
# print('===================== KNN =====================')
# clf = KNN(X_Train, Y_Train)
# test_data(clf, X_Test, Y_Test, 'KNN')
