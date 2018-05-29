import get_mat
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# para_list:
# assign_mode = divided, undivided
# user_mode = all, active, passive
# week_mode = one, two
# user_mode_threshold = 1,2,3,...
# video_mode = m1s1, threshold

# model: svm_l, svm_g, lr, dt, bayes, rf, knn, adaboost, mlp, gpc, qda

def transfer(train_course, test_course, assign_mode, user_mode, user_mode_threshold_train, user_mode_threshold_test, week_mode, video_mode, classifier):
    # print(train_course, test_course, assign_mode, user_mode, user_mode_threshold_train, user_mode_threshold_test, week_mode, video_mode, classifier)
    # =============================test set==========================
    test_mat, test_label = transfer_get_mat.file2mat(course_id = test_course, assign_mode = assign_mode, user_mode = user_mode, user_mode_threshold = user_mode_threshold_test, week_mode = week_mode, video_mode = video_mode)
    test_list_mat = test_mat.fillna(0).values.tolist()
    test_mat = np.asarray(test_list_mat, dtype=np.float64)
    test_list_label = test_label['label'].tolist()
    test_label = np.array(test_list_label)

    test_mat[np.isinf(test_mat) == True] = 0
    test_min_max_scaler = preprocessing.MinMaxScaler(copy=True, feature_range=(0, 1))
    test_X_minmax = test_min_max_scaler.fit_transform(test_mat)

    # ============================training set==================================
    train_mat, train_label = transfer_get_mat.file2mat(course_id = train_course, assign_mode = assign_mode, user_mode = user_mode, user_mode_threshold = user_mode_threshold_train, week_mode = week_mode, video_mode = video_mode)
    train_list_mat = train_mat.fillna(0).values.tolist()
    train_mat = np.asarray(train_list_mat, dtype=np.float64)
    train_list_label = train_label['label'].tolist()
    train_label = np.array(train_list_label)

    train_mat[np.isinf(train_mat) == True] = 0
    train_min_max_scaler = preprocessing.MinMaxScaler(copy=True, feature_range=(0, 1))
    train_X_minmax = train_min_max_scaler.fit_transform(train_mat)

    # =====================build model===================
    X_train, X_test, Y_train, Y_test = train_X_minmax, test_X_minmax, train_label, test_label
    # X_train = np.vstack([X_minmax_1021x4T2015, X_minmax_101x_1T2016[:230]])
    # X_test = X_minmax_101x_1T2016[231:]
    # Y_train = np.concatenate([emnlp_label_1021x4T2015,emnlp_label_101x_1T2016[:230]])
    # Y_test = emnlp_label_101x_1T2016[231:]

    clf = classifier.fit(X_train, Y_train)
    y_pred = clf.predict(X_test)

    with open('/Users/xuyujie/Desktop/result_two.txt', 'a') as out:
        out.write("%s %s %s %s %s %s\n" % (train_course, test_course, assign_mode, user_mode, week_mode, classifier))
        out.write("%s %s\n" % ('Accuracy', str(accuracy_score(Y_test, y_pred))))
        out.write("%s %s\n" % ('Precision', str(precision_score(Y_test, y_pred, average="macro"))))
        out.write("%s %s\n" % ('Recall', str(recall_score(Y_test, y_pred, average="macro"))))
        out.write("%s %s\n" % ('F1 score', str(f1_score(Y_test, y_pred, average="macro"))))
        out.write("%s\n" % (str(X_train.shape)))
        out.write("%s\n" % (str(X_test.shape)))
    out.close()

    # print('Accuracy', accuracy_score(Y_test, y_pred))
    # print('Precision', precision_score(Y_test, y_pred, average="macro"))
    # print('Recall', recall_score(Y_test, y_pred, average="macro"))
    # print('F1 score', f1_score(Y_test, y_pred, average="macro"))
    # print('X_train.shape', X_train.shape)
    # print('X_test.shape', X_test.shape)

    # =====================future work===================
    # TODO: feature selection
    from sklearn.feature_selection import VarianceThreshold
    # sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    # sel.fit_transform(X_scaled)
    # clf = svm.SVC(kernel='linear', C=1)
    # scores = cross_val_score(clf, X_scaled, emnlp_label, cv=5)
    # print(scores)
    # print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    # TODO: try scoring methods on 1 and -1 separately
    # result = []
    #
    # for p in range(len(emnlp_label)*0.8+1, len(label)):
    #     result = clf.predict([emnlp_mat[p]])
    #     print(result)
    #
    # count = 0
    # for i in range(result):
    #     if result[i] == label[len(label)*0.8+1+i]:
    #         count += 1
    #     else:
    #         count = count
    # print("accuracy = ", count/(len(label)*0.2))

    # TODO: (transfer learning) do cross validation on another course
    return 0

if __name__ == '__main__':
    train_course = '102_1x_4T2015'
    test_course = '101x_1T2016'
    # test_course = '102x_1T2016'

    classifiers = [
        # # SVC(kernel="linear", C=1),
        # SVC(kernel="linear", C=0.025),
        # # SVC(gamma=2, C=0.025),
        # LogisticRegression(C=1.0),
        # # DecisionTreeClassifier(max_depth=5),
        # # GaussianNB(),
        # # RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        # KNeighborsClassifier(3),
        # AdaBoostClassifier(),
        # MLPClassifier(alpha=1),
        GaussianProcessClassifier(1.0 * RBF(1.0))
        # QuadraticDiscriminantAnalysis()
        ]

    for classifier in classifiers:
        # transfer(train_course, test_course, 'divided', 'active', 2, 3, 'one', 'threshold', classifier)
        transfer(train_course, test_course, 'divided', 'active', 2, 3, 'two', 'threshold', classifier)
        # transfer(train_course, test_course, 'undivided', 'active', 2, 3, 'one', 'threshold', classifier)
        transfer(train_course, test_course, 'undivided', 'active', 2, 3, 'two', 'threshold', classifier)
        # transfer(train_course, test_course, 'divided', 'passive', 2, 3, 'one', 'threshold', classifier)
        transfer(train_course, test_course, 'divided', 'passive', 2, 3, 'two', 'threshold', classifier)
        # transfer(train_course, test_course, 'undivided', 'passive', 2, 3, 'one', 'threshold', classifier)
        transfer(train_course, test_course, 'undivided', 'passive', 2, 3, 'two', 'threshold', classifier)
        # transfer(train_course, test_course, 'divided', 'all', 2, 3, 'one', 'threshold', classifier)
        transfer(train_course, test_course, 'divided', 'all', 2, 3, 'two', 'threshold', classifier)
        # transfer(train_course, test_course, 'undivided', 'all', 2, 3, 'one', 'threshold', classifier)
        transfer(train_course, test_course, 'undivided', 'all', 2, 3, 'two', 'threshold', classifier)
