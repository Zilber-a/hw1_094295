import pandas as pd
import os
import pickle


# get list of features which appear for at least 80% of rows in dataset and they aren't constant (time series features).
def get_time_series_vars(path):
    ts_vars = []
    df_all_list = []
    for filename in os.listdir(path):
        f = os.path.join(path, filename)
        dataframe = pd.read_csv(f, sep='|')
        if 1 in dataframe['SepsisLabel'].unique():
            dataframe = dataframe.loc[:dataframe.loc[:, 'SepsisLabel'].gt(0).idxmax(), :]
        dataframe.drop(['ICULOS', 'Unit2'], axis=1, inplace=True)
        df_all_list.append(dataframe)
    df_all = pd.concat(df_all_list)
    df_all = df_all.reset_index(drop=True)
    df_all.drop(["Age", "Gender", 'SepsisLabel', 'HospAdmTime', 'Unit1'], axis=1, inplace=True)  # constant features
    vars_na = df_all.isna().sum() / len(df_all)
    for var in vars_na.index:
        if vars_na[var] <= 0.2:
            ts_vars.append(var)
    return ts_vars


# get mean and std of all features (except categorical features)
def get_mean_std(path):
    df_all_list = []
    for filename in os.listdir(path):
        f = os.path.join(path, filename)
        dataframe = pd.read_csv(f, sep='|')
        if 1 in dataframe['SepsisLabel'].unique():
            dataframe = dataframe.loc[:dataframe.loc[:, 'SepsisLabel'].gt(0).idxmax(), :]
        dataframe.drop(['ICULOS', 'Unit2'], axis=1, inplace=True)
        df_all_list.append(dataframe)
    df_all = pd.concat(df_all_list)
    df_all = df_all.reset_index(drop=True)
    df_all.drop(["Gender", 'Unit1'], axis=1, inplace=True)  # binary features
    mean_std_dict = df_all.describe().loc[['mean', 'std']]
    return mean_std_dict


if __name__ == '__main__':
    path = "data/train"

    print("Getting time series features")
    ts_vars = get_time_series_vars(path)

    with open('ts_vars.pickle', 'wb') as handle:
        pickle.dump(ts_vars, handle)

    print("Computing mean and std")
    mean_std_dict = get_mean_std(path)

    with open('mean_std_dict.pickle', 'wb') as handle:
        pickle.dump(mean_std_dict, handle)
