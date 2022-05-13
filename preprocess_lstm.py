import pandas as pd
import os
import pickle
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
import torch


# save preprocessed data for lstm model
def pre_process_data_lstm(path, train=True, save=True):
    with open('mean_std_dict.pickle', 'rb') as handle:
        mean_std_dict = pickle.load(handle)
    i = 0
    df_all_list = []
    labels = []
    ids = []
    print("Reading data")
    for filename in os.listdir(path):
        if i % 500 == 0:
            print(i)
        i += 1
        f = os.path.join(path, filename)
        dataframe = pd.read_csv(f, sep='|')
        label = 0
        if 1 in dataframe['SepsisLabel'].unique():
            # all rows until first time SepsisLabel=1
            dataframe = dataframe.loc[:dataframe.loc[:, 'SepsisLabel'].gt(0).idxmax(), :]
            label = 1
        labels.append(label)
        ids.append(int(filename[8:-4]))  # use the number in the file name as id
        dataframe.drop(['SepsisLabel', 'ICULOS', 'Unit2'], axis=1, inplace=True)
        dataframe = dataframe[['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'BUN', 'Calcium', 'Creatinine',
                               'Glucose', 'Magnesium', 'Potassium', 'Hct', 'Hgb', 'WBC', 'Platelets', 'Age', 'Gender',
                               'Unit1', 'HospAdmTime']]  # use these features to predict
        dataframe = dataframe.fillna(method='bfill').fillna(method='ffill') # fill values using previous and next values

        for feature in mean_std_dict:
            if feature in dataframe.columns:
                # normalize feature
                dataframe[feature] = (dataframe[feature] - mean_std_dict[feature]["mean"]) / mean_std_dict[feature]["std"]

        dataframe['patient_id'] = int(filename[8:-4])

        df_all_list.append(dataframe)
    df_all = pd.concat(df_all_list)
    df_all = df_all.reset_index(drop=True)

    df_features = df_all.drop(['patient_id'], axis=1)
    feature_columns = df_features.columns
    df_ids = df_all[['patient_id']]

    # fill missing data with linear regression iterative imputation
    # if train data- train the imputer, otherwise read the fitted imputer
    print("Filling missing data")
    if train:
        imputer = IterativeImputer(estimator=LinearRegression(), imputation_order='ascending', max_iter=50,
                                   random_state=41, n_nearest_features=20)
        imputer.fit(df_features)
        with open('imputer.pickle', 'wb') as handle:
            pickle.dump(imputer, handle)
    else:
        with open('imputer.pickle', 'rb') as handle:
            imputer = pickle.load(handle)

    df_features = imputer.transform(df_features)
    df_features = pd.DataFrame(df_features, columns=feature_columns)
    df_features["Unit1"] = df_features["Unit1"].apply(lambda x: 0 if x < 0.5 else 1)  # # unit1 feature must be 0 or 1
    df_all = pd.concat([df_features, df_ids], axis=1)

    print("Creating tensors")
    # Creating list of torch tensors
    tensors_list = []
    for id in ids:
        df = df_all[df_all["patient_id"] == id]
        df = df.drop(['patient_id'], axis=1)
        df["Unit1"] = df["Unit1"].median()  # unit 1 is constant, so use the median for all patient's rows
        tensor = torch.tensor(df.values)
        tensors_list.append(tensor)
    target = torch.tensor(labels)

    if save:
        if train:
            with open('tensor_features_train.pickle', 'wb') as handle:
                pickle.dump(tensors_list, handle)
            with open('tensor_targets_train.pickle', 'wb') as handle:
                pickle.dump(target, handle)
        else:
            with open('tensor_features_test.pickle', 'wb') as handle:
                pickle.dump(tensors_list, handle)
            with open('tensor_targets_test.pickle', 'wb') as handle:
                pickle.dump(target, handle)
    else:
        return tensors_list, target, ids


if __name__ == '__main__':
    path = "data/train"
    pre_process_data_lstm(path)

    path = "data/test"
    pre_process_data_lstm(path, train=False)


