import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import metrics

import eda
import regression

X_train = regression.X_train
X_test = regression.X_test
y_train = regression.y_train
y_test = regression.y_test


# Decision Tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# clf = DecisionTreeClassifier(random_state=1)
#
# grid_param=[{'max_depth': [1, 2, 3, 4, 5],
#              'min_samples_split': [2, 4, 6, 8, 10],
#              'min_samples_leaf': [i for i in range(1, 20)],
#              'max_features': [4],
#              'splitter': ['best', 'random'],
#              'criterion': ['gini', 'entropy', 'log_loss']}]
#
# grid_search=GridSearchCV(estimator=clf,param_grid=grid_param,cv=5)
# grid_search.fit(X_train,y_train)
# print(grid_search.best_params_)

clf = DecisionTreeClassifier(criterion= 'gini', max_depth= 5, max_features= 4,
                             min_samples_leaf= 3, min_samples_split= 2,
                             splitter= 'best')

model = clf.fit(X_train, y_train)
y_pred_tree = model.predict(X_test)

cnf_matrix = metrics.confusion_matrix(y_test, y_pred_tree)
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap='YlGnBu', fmt = ".1f")
plt.title('Confusion Matrix - Pre-prunned Deecision Tree')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

TN = cnf_matrix[0][0]
TP = cnf_matrix[1][1]
FN = cnf_matrix[1][0]
FP = cnf_matrix[0][1]
specificity = TN/(TN+FP)

tree_accuracy = metrics.accuracy_score(y_test, y_pred_tree)
tree_precision = metrics.precision_score(y_test, y_pred_tree)
tree_recall = metrics.recall_score(y_test, y_pred_tree)
tree_f1 = metrics.f1_score(y_test, y_pred_tree)
tree_specificity = specificity

print(f'Decision Tree Accuracy {tree_accuracy:.3f}')
print(f'Decision Tree Precision {tree_precision:.3f}')
print(f'Decision Tree Recall: {tree_recall:.3f}')
print(f'Decision Tree F1 Score: {tree_f1:.3f}')
print(f'Decision Tree Specificity: {tree_specificity:.3f}')

y_pred_proba_tree = model.predict_proba(X_test)[::,1]
tree_fpr, tree_tpr, _ = metrics.roc_curve(y_test, y_pred_proba_tree)
tree_auc = metrics.roc_auc_score(y_test, y_pred_proba_tree)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
model_logistic = LogisticRegression()
model_logistic.fit(X_train, y_train)
y_pred_logistic = model_logistic.predict(X_test)

cnf_matrix = metrics.confusion_matrix(y_test, y_pred_logistic)
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap='YlGnBu', fmt = ".1f")
plt.title('Confusion Matrix - Logistic Regression')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()
print(cnf_matrix)

TN = cnf_matrix[0][0]
TP = cnf_matrix[1][1]
FN = cnf_matrix[1][0]
FP = cnf_matrix[0][1]
specificity = TN/(TN+FP)

logistic_accuracy = metrics.accuracy_score(y_test, y_pred_logistic)
logistic_precision = metrics.precision_score(y_test, y_pred_logistic)
logistic_recall = metrics.recall_score(y_test, y_pred_logistic)
logistic_f1 = metrics.f1_score(y_test, y_pred_logistic)
logistic_specificity = specificity

print(f'Logistic Regression Accuracy {logistic_accuracy:.3f}')
print(f'Logistic Regression Precision {logistic_precision:.3f}')
print(f'Logistic Regression Recall: {logistic_recall:.3f}')
print(f'Logistic Regression F1 Score: {logistic_f1:.3f}')
print(f'Logistic Regression Specificity: {logistic_specificity:.3f}')

y_pred_proba_logistic = model_logistic.predict_proba(X_test)[::,1]
logistic_fpr, logistic_tpr, _ = metrics.roc_curve(y_test, y_pred_proba_logistic)
logistic_auc = metrics.roc_auc_score(y_test, y_pred_proba_logistic)

# KNN
from sklearn.neighbors import KNeighborsClassifier
# knn = KNeighborsClassifier()
# k_range = list(range(1,31))
# param_grid = dict(n_neighbors=k_range)
# grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy',
#                     return_train_score=False, verbose=1)
# grid_search = grid.fit(X_train, y_train)
# print(f'The best k is: {grid_search.best_params_}')
knn = KNeighborsClassifier(n_neighbors=28)
model = knn.fit(X_train, y_train)
y_pred_knn = model.predict(X_test)

cnf_matrix = metrics.confusion_matrix(y_test, y_pred_knn)
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap='YlGnBu', fmt = ".1f")
plt.title('Confusion Matrix - KNN')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

TN = cnf_matrix[0][0]
TP = cnf_matrix[1][1]
FN = cnf_matrix[1][0]
FP = cnf_matrix[0][1]
specificity = TN/(TN+FP)

knn_accuracy = metrics.accuracy_score(y_test, y_pred_knn)
knn_precision = metrics.precision_score(y_test, y_pred_knn)
knn_recall = metrics.recall_score(y_test, y_pred_knn)
knn_f1 = metrics.f1_score(y_test, y_pred_knn)
knn_specificity = specificity

print(f'KNN Accuracy {knn_accuracy:.3f}')
print(f'KNN Precision {knn_precision:.3f}')
print(f'KNN Recall: {knn_recall:.3f}')
print(f'KNN F1 Score: {knn_f1:.3f}')
print(f'KNN Specificity: {knn_specificity:.3f}')

y_pred_proba_knn = model.predict_proba(X_test)[::,1]
knn_fpr, knn_tpr, _ = metrics.roc_curve(y_test, y_pred_proba_knn)
knn_auc = metrics.roc_auc_score(y_test, y_pred_proba_knn)

# SVM
from sklearn.svm import SVC
svm = SVC(random_state=1, probability=True)
model = svm.fit(X_train, y_train)
y_pred_svm = model.predict(X_test)

cnf_matrix = metrics.confusion_matrix(y_test, y_pred_svm)
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap='YlGnBu', fmt = ".1f")
plt.title('Confusion Matrix - SVM')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

TN = cnf_matrix[0][0]
TP = cnf_matrix[1][1]
FN = cnf_matrix[1][0]
FP = cnf_matrix[0][1]
specificity = TN/(TN+FP)

svm_accuracy = metrics.accuracy_score(y_test, y_pred_svm)
svm_precision = metrics.precision_score(y_test, y_pred_svm)
svm_recall = metrics.recall_score(y_test, y_pred_svm)
svm_f1 = metrics.f1_score(y_test, y_pred_svm)
svm_specificity = specificity

print(f'SVM Accuracy {svm_accuracy:.3f}')
print(f'SVM Precision {svm_precision:.3f}')
print(f'SVM Recall: {svm_recall:.3f}')
print(f'SVM F1 Score: {svm_f1:.3f}')
print(f'SVM Specificity: {svm_specificity:.3f}')

y_pred_proba_svm = model.predict_proba(X_test)[::,1]
svm_fpr, svm_tpr, _ = metrics.roc_curve(y_test, y_pred_proba_svm)
svm_auc = metrics.roc_auc_score(y_test, y_pred_proba_svm)

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
naive = GaussianNB()
model = naive.fit(X_train, y_train)
y_pred_naive = model.predict(X_test)

cnf_matrix = metrics.confusion_matrix(y_test, y_pred_naive)
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap='YlGnBu', fmt = ".1f")
plt.title('Confusion Matrix - Naive Bayes')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

TN = cnf_matrix[0][0]
TP = cnf_matrix[1][1]
FN = cnf_matrix[1][0]
FP = cnf_matrix[0][1]
specificity = TN/(TN+FP)

naive_accuracy = metrics.accuracy_score(y_test, y_pred_naive)
naive_precision = metrics.precision_score(y_test, y_pred_naive)
naive_recall = metrics.recall_score(y_test, y_pred_naive)
naive_f1 = metrics.f1_score(y_test, y_pred_naive)
naive_specificity = specificity

print(f'Naive Bayes Accuracy {naive_accuracy:.3f}')
print(f'Naive Bayes Precision {naive_precision:.3f}')
print(f'Naive Bayes Recall: {naive_recall:.3f}')
print(f'Naive Bayes F1 Score: {naive_f1:.3f}')
print(f'Naive Bayes Specificity: {naive_specificity:.3f}')

y_pred_proba_naive = model.predict_proba(X_test)[::,1]
naive_fpr, naive_tpr, _ = metrics.roc_curve(y_test, y_pred_proba_naive)
naive_auc = metrics.roc_auc_score(y_test, y_pred_proba_naive)

# ROC/AUC
plt.plot(tree_fpr, tree_tpr, label=f'Decision Tree, auc={tree_auc}')
plt.plot(logistic_fpr, logistic_tpr, label=f'Logistic Regression, auc={logistic_auc}')
plt.plot(knn_fpr, knn_tpr, label=f'KNN, auc={knn_auc}')
plt.plot(svm_fpr, svm_tpr, label=f'SVM, auc={svm_auc}')
plt.plot(naive_fpr, naive_tpr, label=f'Naive Bayes, auc={naive_auc}')
plt.plot([0, 1], [0, 1],'r--')
plt.legend(loc=4)
plt.show()

from prettytable import PrettyTable
x = PrettyTable(['Metric', 'Decision Tree', 'Logistic Regression', 'KNN', 'SVM', 'Naive Bayes'])
x.add_row(['Accuracy:', np.round(tree_accuracy, 3), np.round(logistic_accuracy, 3), np.round(knn_accuracy, 3),
           np.round(svm_accuracy, 3), np.round(naive_accuracy, 3)])
x.add_row(['Precision:', np.round(tree_precision, 3), np.round(logistic_precision, 3), np.round(knn_precision, 3),
           np.round(svm_precision, 3), np.round(naive_precision, 3)])
x.add_row(['Recall:', np.round(tree_recall, 3), np.round(logistic_recall, 3), np.round(knn_recall, 3),
           np.round(svm_recall, 3), np.round(naive_recall, 3)])
x.add_row(['F1 Score:', np.round(tree_f1, 3), np.round(logistic_f1, 3), np.round(knn_f1, 3),
           np.round(svm_f1, 3), np.round(naive_f1, 3)])
x.add_row(['Specificity:', np.round(tree_specificity, 3), np.round(logistic_specificity, 3), np.round(knn_specificity, 3),
           np.round(svm_specificity, 3), np.round(naive_specificity, 3)])
print(x.get_string(title = 'Metric Comparisions'))

