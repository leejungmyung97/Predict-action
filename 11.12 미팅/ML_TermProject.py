import warnings

from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings(action='ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from yellowbrick.classifier import ROCAUC
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns

dataset = pd.read_csv('mpii_human_pose.csv')

category_df = dataset
category_df['new'] = np.nan

cate_list = dataset['Activity'].astype('category').values.categories

# yoga
category_df.loc[category_df['Activity'] == 'yoga, Power', 'new'] = 'yoga'
category_df.loc[category_df['Activity'] == 'yoga, Nadisodhana', 'new'] = 'yoga'
category_df.loc[category_df['Activity'] == 'stretching', 'new'] = 'yoga'
# category_df.loc[category_df['Activity'] == 'pilates, general', 'new'] = 'yoga'
print('yoga: ' + str(len(category_df.loc[category_df['new'] == 'yoga', :])))  # 340

# rowing
category_df.loc[category_df['Activity'] == 'rowing, stationary', 'new'] = 'rowing'
print('rowing: ' + str(len(category_df.loc[category_df['new'] == 'rowing', :])))  # 150

# running
category_df.loc[category_df['Category'] == 'running', 'new'] = 'running'
print('running: ' + str(len(category_df.loc[category_df['new'] == 'running', :])))  # 291

# golf
category_df.loc[category_df['Activity'] == 'golf', 'new'] = 'golf'
print('golf: ' + str(len(category_df.loc[category_df['new'] == 'golf', :])))  # 138

category_df.dropna(axis=0, inplace=True)
X = category_df.drop(columns=['ID', 'NAME', 'Scale', 'Activity', 'Category', 'new'])
y = category_df['new']

scaledX = pd.DataFrame(StandardScaler().fit_transform(X.transpose()))
# encode_y = LabelEncoder().fit_transform(y)

sm = SMOTE()
x_resample, y_resample = sm.fit_resample(scaledX.transpose(), y)

print(dataset.head())
print(dataset.describe())
print(dataset.info())




# Decision Tree Entropy
def dteClassifier(X_train, Y_train, X_test, Y_test):
    dte = DecisionTreeClassifier(criterion="entropy")
    dte.fit(X_train, Y_train)
    print(dte.score(X_test, Y_test))


# Decision Tree Gini
def dtgClassifier(X_train, Y_train, X_test, Y_test):
    dtg = DecisionTreeClassifier(criterion="gini")
    dtg.fit(X_train, Y_train)
    print(dtg.score(X_test, Y_test))


# Logistic Regression
def logisticRegr(X_train, y_train, X_test, y_test):
    logisticRegr = LogisticRegression(solver='lbfgs')
    logisticRegr.fit(X_train, y_train)
    logisticRegr.predict(X_train[0].reshape(1, -1))
    logisticRegr.predict(X_train[0:10])
    predictions = logisticRegr.predict(y_test)
    score = logisticRegr.score(X_test, y_test)
    print(score)


# SVC
def svc(X_train, y_train, X_test, y_test):
    # create an SVC classifier model
    svclassifier = SVC(kernel='linear')
    # fit the model to train dataset
    svclassifier.fit(X_train, y_train)
    # make predictions using the trained model
    y_pred = svclassifier.predict(X_test)
    print(svclassifier.score(X_test, y_test))



# evaluation each model
def evaluation(x, y, classifier):
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=1)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    score = cross_val_score(classifier, X_train, Y_train, cv=skf)
    print(classifier, '\nCross validation score :', score)
    classifier.fit(X_train, Y_train)
    print('Accuracy on test set :', classifier.score(X_test, Y_test))
    print('')


listClassifier = [DecisionTreeClassifier(criterion="entropy"), DecisionTreeClassifier(criterion="gini"),
                  LogisticRegression(), SVC()]

# listBestDf index 0=DecisionTree(entropy), 1=DecisionTree(gini) 2=LogisticRegression, 3=SVC

for i in range(len(listClassifier)):
    evaluation(x_resample, y_resample, listClassifier[i])


# autoML for decision tree
def autoDT(X, y, min_d, max_d):
    """
    find best depth of decision tree using grid search

    Param
    --------
    data: X, y
        data for train

    min_d: int
        minimum depth of decision tree

    max_d: int
        maxiumn depth of decision tree

    Return
    --------
    dict:
         return best depth for grid search
        None means can not find best d in given range(min_d, max_d)
        {'best depth ' = __, 'best score ' = __}
    """
    X_train = X
    y_train = y
    param_grid = [{'max_depth': np.arange(min_d, max_d)}]
    dt_gscv = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=2)
    dt_gscv.fit(X_train, y_train)
    print('Best depth :', dt_gscv.best_params_)
    print('Best score :', dt_gscv.best_score_)

# autoML for logistic regression
def autoLR(X, y, min_C, max_C):
    """
    find best C of logistic regression using grid search

    Param
    --------
    data: X, y
        data for train

    min_C: int
        minimum C of logistic regression

    max_C: int
        maxiumn C of logistic regression

    Return
    --------
    dict:
         return best C for grid search
        None means can not find best d in given range(min_C, max_C)
        {'best C ' = __, 'best score ' = __}
    """
    X_train = X
    y_train = y
    param_grid = [{'C': np.arange(min_C, max_C)}]
    lr_gscv = GridSearchCV(LogisticRegression(), param_grid, cv=2)
    lr_gscv.fit(X_train, y_train)
    print('Best C :', lr_gscv.best_params_)
    print('Best score :', lr_gscv.best_score_)



# grid Decision Tree Gini
X_train, X_test, y_train, y_test = train_test_split(x_resample, y_resample, test_size=0.2, shuffle=True, random_state=1)
param_grid = [{'max_features': np.arange(1, len(X_test.columns)), 'max_depth': np.arange(1, 10)}]
dt_gini_gscv = GridSearchCV(listClassifier[1], param_grid, cv=2, n_jobs=2)
dt_gini_gscv.fit(X_train, y_train)
print('Best parameter :', dt_gini_gscv.best_params_)
print('Best score :', dt_gini_gscv.best_score_)

# grid Logistic Regression
X_train, X_test, y_train, y_test = train_test_split(x_resample, y_resample, test_size=0.2, shuffle=True, random_state=1)
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
              'penalty': ['l1', 'l2']}
lr_gscv = GridSearchCV(listClassifier[2], param_grid, cv=2, n_jobs=2)
lr_gscv.fit(X_train, y_train)
print('Best parameter :', lr_gscv.best_params_)
print('Best score :', lr_gscv.best_score_)

# grid SVC
X_train, X_test, y_train, y_test = train_test_split(x_resample, y_resample, test_size=0.2, shuffle=True, random_state=1)
param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf', 'poly', 'sigmoid']}
svc_gscv = GridSearchCV(listClassifier[3], param_grid, cv=2, n_jobs=2)
svc_gscv.fit(X_train, y_train)
print('Best parameter :', svc_gscv.best_params_)
print('Best score :', svc_gscv.best_score_)

X_train, X_test, y_train, y_test = train_test_split(x_resample, y_resample, test_size=0.2, shuffle=True, random_state=1)
autoDT(X_train, y_train)

# print('\n---------After GridSearchCV---------\n')
# dt_e = DecisionTreeClassifier(max_depth=10, max_features=29, criterion="entropy")
# dt_g = DecisionTreeClassifier(max_depth=9, max_features=24, criterion="gini")
# lr = LogisticRegression(C=100, penalty="l2")
# svc = SVC(C=10, gamma=0.1, kernel='rbf')

# evaluation(listBestDf[0], y, dt_e)
# evaluation(listBestDf[1], y, dt_g)
# evaluation(listBestDf[2], y, lr)
# evaluation(listBestDf[3], y, svc)

# visulization

# Decision Tree(entropy)
dte_best = DecisionTreeClassifier(max_depth=10, max_features=29, criterion="entropy")
dte_best.fit(x_resample, y_resample)
X_train, X_test, Y_train, Y_test = train_test_split(x_resample, y_resample, test_size=0.2, shuffle=True, random_state=1)
print(confusion_matrix(Y_test, dte_best.predict(X_test)))
plt.figure(figsize=(2, 2))
ax = sns.heatmap(metrics.confusion_matrix(Y_test, dte_best.predict(X_test)), annot=True, fmt='.2f', linewidths=.1, cmap='Blues')
plt.title("Heatmap of Decision Tree(entropy)")
plt.show()

visualizer = ROCAUC(dte_best, classes=['golf', 'rowing', 'running', 'yoga'], micro=False, macro=True, per_class=False)
visualizer.fit(x_resample, y_resample)
visualizer.score(x_resample, y_resample)
visualizer.show()

print("classification_report using Decision Tree(entropy)")
print(classification_report(Y_test, dte_best.predict(X_test)))

# Decision Tree(gini)
dtg_best = DecisionTreeClassifier(max_depth=9, max_features=24, criterion="gini")
dtg_best.fit(x_resample, y_resample)
X_train, X_test, Y_train, Y_test = train_test_split(x_resample, y_resample, test_size=0.2, shuffle=True, random_state=1)
print(confusion_matrix(Y_test, dtg_best.predict(X_test)))
plt.figure(figsize=(2, 2))
ax = sns.heatmap(metrics.confusion_matrix(Y_test, dtg_best.predict(X_test)), annot=True, fmt='.2f', linewidths=.1, cmap='Blues')
plt.title("Heatmap of Decision Tree(gini)")
plt.show()

visualizer = ROCAUC(dte_best, classes=['golf', 'rowing', 'running', 'yoga'], micro=False, macro=True, per_class=False)
visualizer.fit(x_resample, y_resample)
visualizer.score(x_resample, y_resample)
visualizer.show()

print("classification_report using Decision Tree(gini)")
print(classification_report(Y_test, dte_best.predict(X_test)))

# LogisticRegression
lr_best = LogisticRegression(C=100, penalty="l2")
lr_best.fit(x_resample, y_resample)
X_train, X_test, Y_train, Y_test = train_test_split(x_resample, y_resample, test_size=0.2, shuffle=True, random_state=1)
print(confusion_matrix(Y_test, lr_best.predict(X_test)))
plt.figure(figsize=(2, 2))
ax = sns.heatmap(metrics.confusion_matrix(Y_test, lr_best.predict(X_test)), annot=True, fmt='.2f', linewidths=.1, cmap='Blues')
plt.title("Heatmap of Logistic Regression")
plt.show()

visualizer = ROCAUC(lr_best, classes=['golf', 'rowing', 'running', 'yoga'], micro=False, macro=True, per_class=False)
visualizer.fit(x_resample, y_resample)
visualizer.score(x_resample, y_resample)
visualizer.show()

print("classification_report using Logistic Regression")
print(classification_report(Y_test, lr_best.predict(X_test)))

# SVM
svc_best = SVC(C=10, gamma=0.1, kernel='rbf')
svc_best.fit(x_resample, y_resample)
X_train, X_test, Y_train, Y_test = train_test_split(x_resample, y_resample, test_size=0.2, shuffle=True, random_state=1)
print(confusion_matrix(Y_test, svc_best.predict(X_test)))
plt.figure(figsize=(2, 2))
ax = sns.heatmap(metrics.confusion_matrix(Y_test, svc_best.predict(X_test)), annot=True, fmt='.2f', linewidths=.1, cmap='Blues')
plt.title("Heatmap of Support Vector Machine")
plt.show()

visualizer = ROCAUC(svc_best, classes=['golf', 'rowing', 'running', 'yoga'], micro=False, macro=True, per_class=False)
visualizer.fit(x_resample, y_resample)
visualizer.score(x_resample, y_resample)
visualizer.show()

print("classification_report using Support Vector Machine")
print(classification_report(Y_test, svc_best.predict(X_test)))

