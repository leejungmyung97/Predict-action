from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import preprocessing
import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, classification_report
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from sklearn import preprocessing
import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


################
# 사용 가능 스케일러  'standard','minmax','maxabs','robust'
# 사용 가능 인코더 'onehot', ordinal'
def myPreprocess(train, test, num_process, cat_process):
    # 카테고리 와 넘버릭 분류
    X_cats = train.select_dtypes(np.object).copy()
    X_nums = train.select_dtypes(exclude=np.object).copy()

    X_cats_t = test.select_dtypes(np.object).copy()
    X_nums_t = test.select_dtypes(exclude=np.object).copy()

    if num_process == 'standard':
        scaler = preprocessing.StandardScaler()
    elif num_process == 'minmax':
        scaler = preprocessing.MinMaxScaler()
    elif num_process == 'maxabs':
        scaler = preprocessing.MaxAbsScaler()
    elif num_process == 'robust':
        scaler = preprocessing.RobustScaler()
    else:
        raise ValueError("Supported 'num_process' : 'standard','minmax','maxabs','robust'")
    if cat_process == 'onehot':
        encoder = preprocessing.OneHotEncoder(sparse=False, handle_unknown='ignore')
    elif cat_process == 'ordinal':
        encoder = preprocessing.OrdinalEncoder()
    else:
        raise ValueError("Supported 'cat_process' : 'onehot', ordinal'")

    X_nums = scaler.fit_transform(X_nums)
    X_cats = encoder.fit_transform(X_cats)

    X_nums_t = scaler.transform(X_nums_t)
    X_cats_t = encoder.transform(X_cats_t)

    train_processed = np.concatenate([X_nums, X_cats], 1)
    test_processed = np.concatenate([X_nums_t, X_cats_t], 1)

    return train_processed, test_processed


def eval_classification(name, model, pred, ytest):
    print('[' + name + ']')
    print("Accuracy (Test Set): %.4f" % accuracy_score(ytest, pred))
    print("Precision (Test Set): %.4f" % precision_score(ytest, pred))
    print("Recall (Test Set): %.4f" % recall_score(ytest, pred))
    print("F1-Score (Test Set): %.4f" % f1_score(ytest, pred))

    fpr, tpr, thresholds = roc_curve(ytest, pred, pos_label=1)  # pos_label: label yang kita anggap positive
    print("AUC: %.2f" % auc(fpr, tpr))


def show_best_hyperparameter(model, hyperparameters):
    for key, value in hyperparameters.items():
        print('Best ' + key + ':', model.get_params()[key])


def show_cmatrix(ytest, pred):
    # Creating confusion matrix
    cm = confusion_matrix(ytest, pred)

    # Putting the matrix a dataframe form
    cm_df = pd.DataFrame(cm, index=['Actually Not Canceled', 'Actually Canceled'],
                         columns=['Predicted Not Canceled', 'Predicted Canceled'])

    # visualizing the confusion matrix
    sns.set(font_scale=1.2)
    plt.figure(figsize=(10, 6))

    sns.heatmap(cm, annot=True, fmt='g', cmap="Blues", xticklabels=cm_df.columns, yticklabels=cm_df.index,
                annot_kws={"size": 20})
    plt.title("Confusion Matrix", size=20)
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.show()


def show_RocCurve(model, pred_X_test, Y_test):
    plot = plot_roc_curve(model, pred_X_test, Y_test)
    plt.show()


def initial_tuning(model, hyperparameters, pred_X_train, Y_train):
    # Find initial hyperparams using GridSearchCV
    initial_tuned_model = GridSearchCV(model, param_grid=hyperparameters, cv=3)
    initial_tuned_model.fit(pred_X_train, Y_train)

    result = pd.DataFrame(initial_tuned_model.cv_results_['params'])
    result['mean_test_score'] = initial_tuned_model.cv_results_['mean_test_score']
    result = result.sort_values(by='mean_test_score', ascending=False)

    return result


def tuning(inital_result, model_name, model, pred_X_train, Y_train):
    _1st_param = inital_result.iloc[0, 0]
    _2nd_param = inital_result.iloc[1, 0]
    _1st_score = inital_result.iloc[0, 1]
    _2nd_score = inital_result.iloc[1, 1]

    # Split to validation dataset and train
    X_train, X_val, y_train, y_val = train_test_split(pred_X_train, Y_train, test_size=0.2, random_state=100)

    best_score = _1st_score
    best_param = _1st_param
    second_score = _2nd_score
    second_param = _2nd_param

    number = 20
    if model_name == 'Decision Tree':
        number = 4
    eps = 0.0003
    k = 0
    best_model = ''

    while True:
        k = k + 1
        if _1st_param - _2nd_param < 0:
            if model_name == 'Decision Tree':
                hyperparameters = np.arange(_1st_param, _2nd_param + 1, number)
            else:
                hyperparameters = np.linspace(_1st_param, _2nd_param, num=number)

        else:
            if model_name == 'Decision Tree':
                hyperparameters = np.arange(_2nd_param, _1st_param + 1, number)
            else:
                hyperparameters = np.linspace(_2nd_param, _1st_param, num=number)

        indexes = [0, -1]
        hyperparameters = np.delete(hyperparameters, indexes)
        print("Iteration ", k)
        print('1st params:', _1st_param, ' score:', _1st_score)
        print('2nd params:', _2nd_param, ' score:', _2nd_score)
        print('hyperparameters:', hyperparameters)

        for i in range(len(hyperparameters)):
            param = hyperparameters[i]
            if model_name == 'Decision Tree':
                model.max_depth = param
            else:
                model.C = param

            model = model.fit(X_train, y_train)
            score = model.score(X_val, y_val)
            if score >= best_score:
                best_score = score
                best_param = param
                best_model = model
            elif score >= second_score:
                second_score = score
                second_param = param

        print('\nbest_score:', best_score)
        print('best_param:', best_param)
        print('second_score:', second_score)
        print('second_param:', second_param)
        print('---------------------------------------')

        if (best_score - _1st_score) / _1st_score <= eps:
            break

        _1st_param = best_param
        _1st_score = best_score
        _2nd_param = second_param
        _2nd_score = second_score
        if model_name == 'Decision Tree':
            number = number / 2

    return best_score, best_model


def autoML(models):
    hyperparameters = ''
    model = ''
    model_name = ''
    for i in range(len(models)):
        model_name = models[i]
        if model_name == 'Logistic Regression':
            # c
            hyperparameters = dict(C=[0.01, 0.1, 1, 10, 100])
            model = LogisticRegression(solver='liblinear', random_state=100)


        elif model_name == 'Decision Tree':
            # max_Depth
            hyperparameters = dict(max_depth=[2, 10, 100])
            model = DecisionTreeClassifier(random_state=100)

        elif model_name == 'SVM':
            # c
            hyperparameters = dict(C=[0.01, 0.1, 1, 10, 100])
            model = SVC(kernel='linear', random_state=100)

        result = initial_tuning(model, hyperparameters, pred_X_train, Y_train)
        print(result)

        best_score, best_model = tuning(result, model_name, model, pred_X_train, Y_train)

        y_pred = best_model.predict(pred_X_test)
        eval_classification(model_name, model, y_pred, Y_test)

        print('Train score: ' + str(best_model.score(pred_X_train, Y_train)))  # accuracy
        print('Test score:' + str(best_model.score(pred_X_test, Y_test)))  # accuracy

        show_cmatrix(Y_test, y_pred)
        show_RocCurve(best_model, pred_X_test, Y_test)


################
# 데이터셑 불러 오기
df_original = pd.read_csv('hotel_bookings.csv', encoding='utf-8')

# 결측치 확인
print("Check dirty data : \n" + str(df_original.isna().sum()))

# 오리지널 카피 for 클린 데이터
df_clean = df_original.copy()

################################ clean
# children 평균 값으로 채우기
df_clean['children'] = df_clean['children'].fillna(df_clean['children'].mean())
# agent 0 값으로 채우기
df_clean['agent'] = df_clean['agent'].fillna(0)
# company 0 값으로 채우기
df_clean['company'] = df_clean['company'].fillna(0)

# country 결측 행 제거
df_clean = df_clean.dropna(axis=0)

# 결측치 확인
print("Check dirty data : \n" + str(df_clean.isna().sum()))
################################

################################ data type setting
print(df_clean.dtypes)

# 카테고리컬 인데 넘버리컬로 된거 변환
df_clean = df_clean.astype({'agent': 'object', 'company': 'object', 'is_repeated_guest': 'object'})

print(df_clean.dtypes)
################################


################################ use preprocess()

# split First
# X에 들어갈 attribute는 사전에 준비한 그룹 별로 선택하여 사용
# 고객1
X_c1 = df_clean[['adults', 'children', 'babies', 'country', 'customer_type']]
# 고객2
X_c2 = df_clean[
    ['stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children', 'babies', 'country', 'customer_type',
     'previous_cancellations', 'previous_bookings_not_canceled']]
# 시기
X_w = df_clean[['arrival_date_month', 'stays_in_weekend_nights', 'stays_in_week_nights']]
# 서비스
X_s = df_clean[
    ['meal', 'reserved_room_type', 'assigned_room_type', 'required_car_parking_spaces', 'total_of_special_requests',
     'adr']]
# 예약정보
X_r = df_clean[
    ['meal', 'reserved_room_type', 'assigned_room_type', 'required_car_parking_spaces', 'total_of_special_requests',
     'market_segment', 'distribution_channel', 'booking_changes', 'deposit_type', 'agent', 'company', 'adr',
     'reservation_status']]

Y = df_clean['is_canceled']

################################################여기에 원하는 그룹 선택해서 넣기
X_train, X_test, Y_train, Y_test = train_test_split(X_c1, Y, test_size=0.2, shuffle=True, stratify=Y, random_state=34)

# 사용 가능 스케일러  'standard','minmax','maxabs','robust'
# 사용 가능 인코더 'onehot', ordinal'
# 테스트를 위해 고객1 그룹만 사용
pred_X_train, pred_X_test = myPreprocess(X_train, X_test, 'standard', 'onehot')

# 확인
# print(pred_X_train)
# print(pred_X_test)


models = ['Logistic Regression', 'Decision Tree', 'SVM']
autoML(models)