import pandas as pd
import os
import pickle
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.svm import SVC
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
import lime
import lime.lime_tabular



if __name__ == '__main__':
    print("reading data")

    with open('train_data_window_20.pickle', 'rb') as handle:
        df_train = pickle.load(handle)
    with open('test_data_window_20.pickle', 'rb') as handle:
        df_test = pickle.load(handle)

    # split to features and label for train and test
    X = df_train.drop(['SepsisLabel', 'patient_id'], axis=1)
    y = df_train['SepsisLabel']
    X_test = df_test.drop(['SepsisLabel', 'patient_id'], axis=1)
    y_test = df_test['SepsisLabel']
    columns = X.columns

    # fill missing data with linear regression iterative imputation
    imputer = IterativeImputer(estimator=LinearRegression(), imputation_order='ascending', max_iter=200,
                               random_state=41, n_nearest_features=20)
    imputer.fit(X)
    X = imputer.transform(X)
    X = pd.DataFrame(X, columns=columns)
    X["Unit1"] = X["Unit1"].apply(lambda x: 0 if x < 0.5 else 1)  # unit1 feature must be 0 or 1
    X_test = imputer.transform(X_test)
    X_test = pd.DataFrame(X_test, columns=columns)
    X_test["Unit1"] = X_test["Unit1"].apply(lambda x: 0 if x < 0.5 else 1)

    print("Fitting classifier")
    clf = SVC(random_state=41, class_weight='balanced', probability=True).fit(X, y)
    # clf = LogisticRegression(random_state=41, class_weight='balanced').fit(X, y)


    print("Predicting test")
    preds = clf.predict(X_test)

    print("F1 score: ", f1_score(y_test, preds))
    print("Precision score: ", precision_score(y_test, preds))
    print("Recall score: ", recall_score(y_test, preds))
    print("Accuracy score: ", accuracy_score(y_test, preds))

    print("\nPredicting train")
    preds = clf.predict(X)

    print("F1 score: ", f1_score(y, preds))
    print("Precision score: ", precision_score(y, preds))
    print("Recall score: ", recall_score(y, preds))
    print("Accuracy score: ", accuracy_score(y, preds))

    # explain some examples using Lime
    explainer = lime.lime_tabular.LimeTabularExplainer(X.values, training_labels=y.values, feature_names=X.columns, categorical_features=[113, 114],
                                                       class_names=[0, 1])
    exp = explainer.explain_instance(X_test.values[5], clf.predict_proba)
    exp.save_to_file('base01.html')
    exp = explainer.explain_instance(X_test.values[6], clf.predict_proba)
    exp.save_to_file('base11.html')

