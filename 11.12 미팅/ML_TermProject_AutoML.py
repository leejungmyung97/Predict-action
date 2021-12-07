from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score, plot_confusion_matrix
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# AutoML for DT
def autoDT(data: pd.DataFrame, min_D: int, max_D: int, min_gamma, max_gamma) -> dict:
    """
    find best depth of DecisionTree using grid score

    Param
    --------
    data: DataFrame
        data for train

    min_D: int
        minimum Depth

    max_D: int
        maximum Depth

    Return
    --------
    dict:
        return best C & Gamma components and best kernel type
        {'best D components': __}
    """



    max_grid_scores = {}
    k = min_D
    for kernel_type in ['entropy', 'gini']:
        while True:
            model = DecisionTreeClassifier(c_components=k, gamma_components=k, covariance_type=kernel_type)
            model.fit(data)

            new_score = grid_score(data, model.fit_predict(data))

            if k != min_C and new_score < last_score:
                max_grid_scores[last_score] = (k - 1, kernel_type)
                break

            if k >= max_C:
                max_grid_scores[new_score] = (k, kernel_type)
                break

            last_score = new_score
            k += 1

    # packing result to return
    best_k, best_kernel_type = max_grid_scores[max(max_grid_scores.keys())]

    return {'best n components': best_k, 'best covariance type': best_kernel_type}


# CG = C & Gamma in SVM
def autoSVM(data: pd.DataFrame, min_C: int, max_C: int, min_gamma, max_gamma) -> dict:
    """
    find best C, Gamma, Kernel of SVM using grid score

    Param
    --------
    data: DataFrame
        data for train

    min_C: int
        minimum C & Gamma of SVM C & Gamma components

    max_C: int
        maximum C & Gamma of SVM C & Gamma components

    Return
    --------
    dict:
        return best C & Gamma components and best kernel type
        {'best C components': __, 'best gamma components': __, 'best kernel type': __}
    """

    # svm = SVC(kernel='rbf', random_state=1,
    #           gamma=0.008, C=0.1)


    max_grid_scores = {}
    k = min_C
    l = min_gamma
    for kernel_type in ['poly', 'rgb', 'sigmoid']:
        while True:
            model = SVC(c_components=k, gamma_components=k, covariance_type=kernel_type)
            model.fit(data)

            new_score = grid_score(data, model.fit_predict(data))

            if k != min_C and new_score < last_score:
                max_grid_scores[last_score] = (k - 1, kernel_type)
                break

            if k >= max_C:
                max_grid_scores[new_score] = (k, kernel_type)
                break

            last_score = new_score
            k += 1

    # packing result to return
    best_k, best_kernel_type = max_grid_scores[max(max_grid_scores.keys())]

    return {'best n components': best_k, 'best covariance type': best_kernel_type}



def drawHistogram(data):
    plt.figure()
    plt.hist(data, histtype='bar')
    plt.show()



def getContainList(list, key) -> list:
    result = []
    for word in list:
        if key in word:
            result.append(word)

    return result


def preprocessingTest():
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

    # category_df.dropna(axis=0, inplace=True)
    # category_df['nose_X'] = (category_df['upper neck_X'] + category_df['head top_X'])/2
    # category_df['nose_Y'] = (category_df['upper neck_Y'] + category_df['head top_Y'])/2
    # X = category_df.drop(columns=['ID', 'NAME', 'Scale', 'Activity', 'Category', 'new',
    #                             'pelvis_X','pelvis_Y', 'thorax_X', 'thorax_Y',
    #                             'upper neck_X', 'upper neck_Y', 'head top_X', 'head top_Y'])

    X.info()
    y = category_df['new']

    scaledX = pd.DataFrame(StandardScaler().fit_transform(X.transpose()))
    # encode_y = LabelEncoder().fit_transform(y)

    sm = SMOTE()
    x_resample, y_resample = sm.fit_resample(scaledX.transpose(), y)

    return x_resample, y_resample


if __name__ == '__main__':
    x, y = preprocessingTest()

    model = RandomForestClassifier(max_features='log2', n_estimators=200).fit(x, y)
    joblib.dump(model, 'poseModel.pkl')

    param = {'n_estimators': [50, 100, 150, 200],
            'criterion': ['gini', 'entropy'],
            'max_features': ['sqrt', 'log2']}


    gscv =GridSearchCV(estimator=RandomForestClassifier(), param_grid=param, cv=10, n_jobs=-1)
    gscv.fit(x,y)
    print(gscv.best_params_)
    print(gscv.best_score_)
    kmeans_result = autoSVM(x, 2, 20, 0.9)
    print(kmeans_result)


    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, shuffle=True)

