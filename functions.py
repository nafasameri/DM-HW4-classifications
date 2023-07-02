import dtreeviz
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn import svm
from sklearn import tree
from sklearn.metrics import roc_auc_score, confusion_matrix, jaccard_score
from sklearn.metrics import matthews_corrcoef, precision_recall_curve, classification_report

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split, KFold, GridSearchCV


def read_data(filename):
    data = pd.read_csv(filename, sep=';')
    return data


def preproccessing(data):
    # print(data.isnull().sum())
    # jobs = data.groupby('job')['job'].head()
    jobs = data['job'].value_counts().index
    job_labels = {job: i for i, job in enumerate(jobs)}

    maritals = data['marital'].value_counts().index
    marital_labels = {marital: i for i, marital in enumerate(maritals)}

    educations = data['education'].value_counts().index
    education_labels = {education: i for i, education in enumerate(educations)}

    defaults = data['default'].value_counts().index
    default_labels = {default: i for i, default in enumerate(defaults)}

    housings = data['housing'].value_counts().index
    housing_labels = {housing: i for i, housing in enumerate(housings)}

    loans = data['loan'].value_counts().index
    loan_labels = {loan: i for i, loan in enumerate(loans)}

    contacts = data['contact'].value_counts().index
    contact_labels = {contact: i for i, contact in enumerate(contacts)}

    months = data['month'].value_counts().index
    month_labels = {month: i for i, month in enumerate(months)}

    poutcomes = data['poutcome'].value_counts().index
    poutcome_labels = {poutcome: i for i, poutcome in enumerate(poutcomes)}

    ys = data['y'].value_counts().index
    y_labels = {y: i for i, y in enumerate(ys)}
    print('y_labels', y_labels)

    # تبدیل مقادیر رشته‌ای به برچسب‌های عددی
    data["job"] = data["job"].map(job_labels)
    data["marital"] = data["marital"].map(marital_labels)
    data["education"] = data["education"].map(education_labels)
    data["default"] = data["default"].map(default_labels)
    data["housing"] = data["housing"].map(housing_labels)
    data["loan"] = data["loan"].map(loan_labels)
    data["contact"] = data["contact"].map(contact_labels)
    data["month"] = data["month"].map(month_labels)
    data["poutcome"] = data["poutcome"].map(poutcome_labels)
    data["y"] = data["y"].map(y_labels)

    X = data.iloc[:, :-1]
    Y = data.iloc[:, -1]
    return X, Y


def normalize(data):
    scaler = MinMaxScaler()

    numeric_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    data[numeric_features] = scaler.fit_transform(data[numeric_features])

    return data


def split_data(X, Y, k=5):
    X_Train, X_Test, Y_Train, Y_Test = [], [], [], []
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    for train_index, test_index in kfold.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        X_Train.append(X_train)
        X_Test.append(X_test)
        Y_Train.append(y_train)
        Y_Test.append(y_test)

    print("Number of train data:", np.array(X_Train).shape)
    print("Number of test data:", np.array(X_Test).shape)

    # train_size = Percent / 100
    # X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, train_size=train_size, random_state=42)
    # print("Number of train data:", len(X_Train))
    # print("Number of test data:", len(X_Test))

    return X_Train, X_Test, Y_Train, Y_Test


def selectFeatures(X, y):
    # انتخاب ویژگی‌های مهم
    selector = SelectKBest(score_func=chi2, k=10)  # انتخاب 10 ویژگی برتر
    X_new = selector.fit_transform(X, y)

    selected_features = X.columns[selector.get_support()]

    return X_new, selected_features


def decision_tree(X_Train, Y_Train):
    model_dt = DecisionTreeClassifier(criterion='entropy')
    model_dt.fit(X_Train, Y_Train)

    viz = dtreeviz.model(model_dt, X_Train, Y_Train)
    v = viz.view()
    v.save("Decision tree.svg")

    # plt.figure(figsize=(200, 150))
    # tree.plot_tree(model_dt, filled=True)
    # plt.title("Decision tree trained")
    # plt.savefig("Decision-Tree.png")
    # plt.show()

    # text_representation = tree.export_text(model_dt)
    # print('text_representation:', text_representation)

    # from sklearn.inspection import DecisionBoundaryDisplay
    # feature_1, feature_2 = np.meshgrid(np.linspace(iris.data[:, 0].min(), iris.data[:, 0].max()), np.linspace(iris.data[:, 1].min(), iris.data[:, 1].max()))
    # grid = np.vstack([feature_1.ravel(), feature_2.ravel()]).T
    # tree = DecisionTreeClassifier().fit(iris.data[:, :2], iris.target)
    # y_pred = np.reshape(tree.predict(grid), feature_1.shape)
    # display = DecisionBoundaryDisplay(xx0 = feature_1, xx1 = feature_2, response = y_pred)
    # display.plot()
    # display.ax_.scatter(iris.data[:, 0], iris.data[:, 1], c = iris.target, edgecolor = "black")
    # plt.show()

    return model_dt


def gaussian(X_train, y_train):
    model_nb = GaussianNB()
    model_nb.fit(X_train, y_train)
    return model_nb


def SVM(X_Train, Y_Train):
    clf = svm.SVC().fit(X_Train, Y_Train)
    return clf


def KNN(X_Train, Y_Train):
    model_knn = KNeighborsClassifier()

    param_grid = {'n_neighbors': range(1, 20)}

    # جستجوی شبکه برای پیدا کردن مقدار k بهینه
    grid_search = GridSearchCV(estimator=model_knn, param_grid=param_grid, scoring='accuracy')
    grid_search.fit(X_Train, Y_Train)

    # مقدار k بهینه
    best_k = grid_search.best_params_['n_neighbors']
    print('best_k:', best_k)

    # ساختن مدل با k بهینه
    model_knn = KNeighborsClassifier(n_neighbors=best_k)

    # آموزش مدل بر روی داده‌های آموزش
    model_knn.fit(X_Train, Y_Train)

    return model_knn


def test_data(clf, X_Test, Y_Test, method):
    y_pred = clf.predict(X_Test)
    # print("prediction labels:", y_pred)

    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(Y_Test, y_pred)
    AUC_ROC = roc_auc_score(Y_Test, y_pred)
    roc_curve = plt.figure()
    plt.plot(fpr, tpr, '-', label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC)
    plt.title('ROC curve')
    plt.xlabel("FPR (False Positive Rate)")
    plt.ylabel("TPR (True Positive Rate)")
    plt.legend(loc="lower right")
    plt.savefig('results/' + method + "-ROC.png")
    # plt.show()

    precision, recall, thresholds = precision_recall_curve(Y_Test, y_pred)
    precision = np.fliplr([precision])[0]  # so the array is increasing (you won't get negative AUC)
    recall = np.fliplr([recall])[0]  # so the array is increasing (you won't get negative AUC)
    AUC_prec_rec = np.trapz(precision, recall)
    print("Area under Precision-Recall curve:", AUC_prec_rec)
    prec_rec_curve = plt.figure()
    plt.plot(recall, precision, '-', label='Area Under the Curve (AUC = %0.4f)' % AUC_prec_rec)
    plt.title('Precision - Recall curve')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower right")
    plt.savefig('results/' + method + "-Precision_recall.png")
    # plt.show()

    confusion = confusion_matrix(Y_Test, y_pred)
    print("confusion_matrix: ", confusion)
    tn, fp, fn, tp = confusion.ravel()

    accuracy = 0
    if float(np.sum(confusion)) != 0:
        accuracy = float(tp + tn) / float(np.sum(confusion))
    print("Accuracy: " + str(accuracy))

    specificity = 0
    if float(tn + fp) != 0:
        specificity = float(tn) / float(tn + fp)
    print("Specificity: " + str(specificity))

    sensitivity = 0
    if float(tp + fn) != 0:
        sensitivity = float(tp) / float(tp + fn)
    print("Sensitivity: " + str(sensitivity))

    precision = 0
    if float(tp + fp) != 0:
        precision = float(tp) / float(tp + fp)
    print("Precision: " + str(precision))

    NPV = 0
    if float(tn + fn) != 0:
        NPV = float(tn) / float(tn + fn)
    print("NPV: " + str(NPV))

    f1score = 0
    if float(tp + fp + fn) != 0:
        f1score = float((2. * tp)) / float((2. * tp) + fp + fn)
    print("F1-Score: " + str(f1score))

    error_rate = 0
    if float(np.sum(confusion)) != 0:
        error_rate = float(fp + fn) / float(np.sum(confusion))
    print("Error Rate: " + str(error_rate))

    jaccard_index = jaccard_score(Y_Test, y_pred, average='weighted')
    print("Jaccard similarity score: " + str(jaccard_index))

    corrcoef = matthews_corrcoef(Y_Test, y_pred)
    print("The Matthews correlation coefficient: " + str(corrcoef))

    file_perf = open('results/' + method + '-performances.txt', 'w')
    file_perf.write("Jaccard similarity score: " + str(jaccard_index)
                    + "\nConfusion matrix: " + str({"Real Pos": {"tp": tp, "fn": fn}, "Real Neg": {"fp": fp, "tn": tn}})
                    + "\nACCURACY: " + str(accuracy)
                    + "\nSENSITIVITY: " + str(sensitivity)
                    + "\nSPECIFICITY: " + str(specificity)
                    + "\nPRECISION: " + str(precision)
                    + "\nNPV: " + str(NPV)
                    + "\nError Rate: " + str(error_rate)
                    + "\nThe Matthews correlation coefficient: " + str(corrcoef)
                    + "\nF1 score: " + str(f1score))
    file_perf.close()

    rep = classification_report(Y_Test, y_pred)
    print(rep)
    file_perf = open('results/' + method + '-classification_report.txt', 'w')
    file_perf.write(rep)
    file_perf.close()

    return precision