import pandas as pd
import os
import pickle


# save preprocessed data for baseline model
def pre_process_data(path, train=True):
    print("Getting time series features")
    with open('ts_vars.pickle', 'rb') as handle:
        ts_vars = pickle.load(handle)
    with open('mean_std_dict.pickle', 'rb') as handle:
        mean_std_dict = pickle.load(handle)
    i = 0
    num_rows = 20  # use the last rows for each patient
    df_all_list = []
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
        dataframe.drop(['SepsisLabel', 'ICULOS', 'Unit2'], axis=1, inplace=True)  # non used features, and label column
        dataframe = dataframe[['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'BUN', 'Calcium', 'Creatinine',
                               'Glucose', 'Magnesium', 'Potassium', 'Hct', 'Hgb', 'WBC', 'Platelets', 'Age', 'Gender',
                               'Unit1', 'HospAdmTime']]  # use these features to predict
        dataframe = dataframe.fillna(method='bfill').fillna(method='ffill')  # fill values using previous and next values
        df_ts = dataframe[ts_vars]  # df with the time series features
        df_const = dataframe.drop(ts_vars, axis=1)  # df with the constant features

        # expend df to have num_rows rows if needed
        if len(dataframe) < num_rows:
            df_ts = pd.concat([df_ts.iloc[[0]]] * (num_rows - len(dataframe)) + [df_ts], ignore_index=True)
        df_ts = df_ts.iloc[-num_rows:, :]  # use only last rows
        df_ts.reset_index(drop=True, inplace=True)
        for feature in mean_std_dict:
            if feature in df_ts.columns:
                # normalize feature
                df_ts[feature] = (df_ts[feature]-mean_std_dict[feature]["mean"])/mean_std_dict[feature]["std"]
        df_ts = df_ts.stack(dropna=False).to_frame().T  # turn df of patient to a vector
        df_ts.columns = [f'{i}_{j}' for i, j in df_ts.columns]

        df_const = df_const.median().to_frame().T  # take median of constant features
        df_const.reset_index(drop=True, inplace=True)
        for feature in mean_std_dict:
            if feature in df_const.columns:
                # normalize feature
                df_const[feature] = (df_const[feature]-mean_std_dict[feature]["mean"])/mean_std_dict[feature]["std"]
        df_const['SepsisLabel'] = label
        df_const['patient_id'] = int(filename[8:-4])  # use the number in the file name as id
        dataframe = pd.concat([df_ts, df_const], axis=1)  # vector consisting of all features

        df_all_list.append(dataframe)
    df_all = pd.concat(df_all_list)
    df_all = df_all.reset_index(drop=True)

    # save the preprocessed data
    if train:
        with open('train_data_window_'+str(num_rows)+'.pickle', 'wb') as handle:
            pickle.dump(df_all, handle)
    else:
        with open('test_data_window_'+str(num_rows)+'.pickle', 'wb') as handle:
            pickle.dump(df_all, handle)



if __name__ == '__main__':
    path = "data/train"
    pre_process_data(path)
    test_path = "data/test"
    pre_process_data(test_path, train=False)

