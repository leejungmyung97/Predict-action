from typing import Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, plot_confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
import pickle, joblib


def autoKmeans(data: pd.DataFrame, min_k: int, max_k: int, elbow_threshold: float) -> dict:
    """
    find best k of kmeans using elbow method and silhouette score

    Param
    --------
    data: DataFrame
        data for train

    min_k: int 
        minimum k of kmeans

    max_k: int
        maxiumn k of kmeans

    elbow_threshold: float 
        threshold to find elbow point

    Return
    --------
    dict: 
        return best k for elbow and silhouette
        None means can not find best k in given range(min_k, max_k)
        {'best k for elbow' = __ , 'best k for silhouette': __}
    """
    elbow_k = False
    max_silhouette_k = False

    silhouette_scores = []
    inertias = []
    k = min_k

    while not(elbow_k and max_silhouette_k):
        model = KMeans(n_clusters=k)
        model.fit(data)

        if not max_silhouette_k:
            score = silhouette_score(data, model.fit_predict(data))
            silhouette_scores.append(score)
            if max(silhouette_scores) != score:
                max_silhouette_k = k-1

        if not elbow_k:
            inertias.append(model.inertia_)

            if len(inertias) >= 3:
                now = model.inertia_
                prev = inertias[k-3]
                pprev = inertias[k-4]

                if (now - prev) < elbow_threshold*(prev - pprev):
                    elbow_k = k - 1

        if k >= max_k:
            break

        k += 1

    # packing result to return
    if elbow_k == False:
        elbow_k = None
    if max_silhouette_k == False:
        max_silhouette_k = None
    result = {}
    result['best k for elbow'] = elbow_k
    result['best k for silhouette'] = max_silhouette_k

    return result

def autoGM(data: pd.DataFrame, min_k: int, max_k: int) -> dict:
    """
    find best k of gaussian mixture using silhouette score

    Param
    --------
    data: DataFrame
        data for train

    min_k: int 
        minimum k of gaussian mixture n componets

    max_k: int
        maxiumn k of gaussian mixture n componets

    Return
    --------
    dict:
        return best n components and best covarinace type
        {'best n components': __ , 'best covariance type': __}

    """

    max_silhouette_scores = {}
    k = min_k
    for covar_type in ['full', 'tied', 'diag', 'spherical']:
        while True:
            model = GaussianMixture(n_components=k, covariance_type=covar_type)
            model.fit(data)

            new_score = silhouette_score(data, model.fit_predict(data))
            
            if k != min_k and new_score < last_score:
                max_silhouette_scores[last_score] = (k-1,covar_type)
                break
            
            if k >= max_k:
                max_silhouette_scores[new_score] = (k,covar_type)
                break

            last_score = new_score
            k += 1

    # packing result to return
    best_k, best_covar_type = max_silhouette_scores[max(max_silhouette_scores.keys())]
    
    return {'best n components': best_k, 'best covariance type': best_covar_type}

def autoAgglomertive(data: pd.DataFrame, min_k: int, max_k: int) -> dict:
    max_silhouette_scores = {}
    k = min_k
    for linkage in ['ward', 'complete', 'average', 'single']:
        while True:
            model = AgglomerativeClustering(n_clusters=k, linkage=linkage)
            model.fit(data)

            new_score = silhouette_score(data, model.fit_predict(data))
            
            if k != min_k and new_score < last_score:
                max_silhouette_scores[last_score] = (k-1,linkage)
                break
            
            if k >= max_k:
                max_silhouette_scores[new_score] = (k,linkage)
                break

            last_score = new_score
            k += 1

    # packing result to return
    best_k, best_linkage = max_silhouette_scores[max(max_silhouette_scores.keys())]
    
    return {'best n components': best_k, 'best linkage': best_linkage}

def drawHistogram(data):
    plt.figure()
    plt.hist(data, histtype='bar')
    plt.show()

def testAutoML():
    from sklearn import datasets
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # test kmeans
    kmeans_result = autoKmeans(X, min_k=2, max_k=30, elbow_threshold=0.85)
    print(kmeans_result)

    # test gausian mixture
    gm_result = autoGM(X, min_k=2, max_k=30)
    print(gm_result)

    # test agglomertive clustering
    agglo_result = autoAgglomertive(X, min_k=2, max_k=30)
    print(agglo_result)

def getContainList(list, key) -> list:
    result = []
    for word in list:
        if key in word:
            result.append(word)
    
    return result

def preprocessingTest():
    dataset = pd.read_csv('Dataset/mpii_human_pose.csv')

    category_df = dataset
    category_df['new'] = np.nan

    cate_list = dataset['Activity'].astype('category').values.categories

    # bicycling    
    # category_df.loc[category_df['Category'] == 'bicycling', 'new'] = 'bicycling' #516
    # category_df.loc[category_df['Activity'] == 'bicycling, stationary', 'new'] = 'bicycling' # 102
    # category_df.loc[category_df['Activity'] == 'upper body exercise, stationary bicycle - Airdyne (arms only) 40 rpm, moderate', 'new'] = 'bicycling' # 32
    # print('bicycling: ' + str(len(category_df.loc[category_df['new'] == 'bicycling', :]))) # 650

    # # yoga
    category_df.loc[category_df['Activity'] == 'yoga, Power', 'new'] = 'yoga' 
    category_df.loc[category_df['Activity'] == 'yoga, Nadisodhana', 'new'] = 'yoga'
    category_df.loc[category_df['Activity'] == 'stretching', 'new'] = 'yoga'
    # category_df.loc[category_df['Activity'] == 'pilates, general', 'new'] = 'yoga'
    print('yoga: ' + str(len(category_df.loc[category_df['new'] == 'yoga', :]))) # 340

    # manmom
    # category_df.loc[category_df['Activity'] == 'video exercise workouts, TV conditioning programs', 'new'] = 'manmom'
    # category_df.loc[category_df['Activity'] == 'resistance training', 'new'] = 'resistance'
    # category_df.loc[category_df['Activity'] == 'circuit training', 'new'] = 'resistance'
    # category_df.loc[category_df['Activity'] == 'aerobic, step', 'new'] = 'aerobic'
    
    # category_df.loc[category_df['Activity'] == 'calisthenics', 'new'] = 'yoga'
    # category_df.loc[category_df['Activity'] == 'home exercise, general', 'new'] = 'home'
    # category_df.loc[category_df['Activity'] == 'slide board exercise, general', 'new'] = 'slide board'
    # category_df.loc[category_df['Activity'] == 'stretching', 'new'] = 'manmom'
    # category_df.loc[category_df['Activity'] == 'rope skipping, general', 'new'] = 'rope'
    # print('manmom: ' + str(len(category_df.loc[category_df['new'] == 'manmom', :]))) # 1024

    # rowing
    category_df.loc[category_df['Activity'] == 'rowing, stationary', 'new'] = 'rowing' 
    print('rowing: ' + str(len(category_df.loc[category_df['new'] == 'rowing', :]))) # 150


    # skiing
    # ski_list = getContainList(cate_list, 'skiing')
    # category_df.loc[((category_df['Activity'].isin(ski_list)) & 
    #                 (category_df['Category'] == 'winter activities')), 'new'] = 'skiing' 
    # print('ski: ' + str(len(category_df.loc[category_df['new'] == 'skiing', :]))) # 355

    # running
    category_df.loc[category_df['Category'] == 'running', 'new'] = 'running' 
    print('running: ' + str(len(category_df.loc[category_df['new'] == 'running', :]))) # 291

    # skateboarding
    # category_df.loc[category_df['Activity'] == 'skateboarding', 'new'] = 'skateboarding' 
    # print('skateboarding: ' + str(len(category_df.loc[category_df['new'] == 'skateboarding', :]))) # 184
    
    # baseball
    # category_df.loc[category_df['Activity'] == 'softball, general', 'new'] = 'baseball'  
    # print('baseball: ' + str(len(category_df.loc[category_df['new'] == 'baseball', :]))) # 173

    # soccer
    # category_df.loc[category_df['Activity'] == 'soccer', 'new'] = 'soccer'  
    # print('soccer: ' + str(len(category_df.loc[category_df['new'] == 'soccer', :]))) # 137

    # golf
    category_df.loc[category_df['Activity'] == 'golf', 'new'] = 'golf'
    print('golf: ' + str(len(category_df.loc[category_df['new'] == 'golf', :]))) # 138

    # basketball
    # category_df.loc[category_df['Activity'] == 'basketball', 'new'] = 'basketball'
    # category_df.loc[category_df['Activity'] == 'basketball, game (Taylor Code 490)', 'new'] = 'basketball'
    # print('basketball: ' + str(len(category_df.loc[category_df['new'] == 'basketball', :]))) # 170


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

    model = RandomForestClassifier(max_features='log2', n_estimators=200).fit(x,y)
    joblib.dump(model, 'poseModel.pkl')

    # param = {'n_estimators': [50,100,150,200],
    #         'criterion': ['gini', 'entropy'],
    #         'max_features': ['sqrt','log2']}
    # gscv =GridSearchCV(estimator=RandomForestClassifier(), param_grid=param, cv=10, n_jobs=-1)
    # gscv.fit(x,y)
    # print(gscv.best_params_)
    # print(gscv.best_score_)
    # kmeans_result = autoKmeans(x, 2, 20, 0.9)
    # print(kmeans_result)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, shuffle=True)


    # [RandomForestClassifier(), RidgeClassifier(), DecisionTreeClassifier(), MLPClassifier()]
    # for model in [RandomForestClassifier(max_features='log2', n_estimators=200)] :
    #     model.fit(X_train,y_train)
    #     score = model.score(X_test,y_test)
    #     print('{}: {}'.format(model, score))
    #     plot_confusion_matrix(model, X_test, y_test)
    #     plt.title(model)
    # plt.show()

    # dataset = pd.read_csv('Dataset/mpii_human_pose.csv')
    # dataset.sample(4000)
    # x = dataset.drop(columns=['ID', 'NAME', 'Scale','Activity', 'Category'])
    # y = dataset['Activity']

    # x = StandardScaler().fit_transform(x)
    # y = LabelEncoder().fit_transform(y)

    # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42, shuffle=True)

    # for model in [RandomForestClassifier(), RidgeClassifier(), DecisionTreeClassifier(), MLPClassifier()]:
    #     model.fit(X_train,y_train)
    #     score = model.score(X_test,y_test)
    #     print('{}: {}'.format(model, score))

