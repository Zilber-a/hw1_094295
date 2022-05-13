import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_selection import chi2

# EACH PATIENT REPRESENTED WITH MULTIPLE ROWS

i = 0
path = "data/train"
num_sepsis = 0
df_all_list = []  # list of each patient's preprocessed df
for filename in os.listdir(path):
    if i % 500 == 0:
        print(i)
    i += 1
    f = os.path.join(path, filename)
    dataframe = pd.read_csv(f, sep='|')
    # check if current patient is positive
    if 1 in dataframe['SepsisLabel'].unique():
        # get all rows until first time SepsisLabel=1
        dataframe = dataframe.loc[:dataframe.loc[:, 'SepsisLabel'].gt(0).idxmax(), :]
        dataframe = dataframe.assign(SepsisLabel=1)
        num_sepsis += 1
    dataframe.drop(['ICULOS', 'Unit2'], axis=1, inplace=True)
    df_all_list.append(dataframe)
df_all = pd.concat(df_all_list)
df_all = df_all.reset_index(drop=True)
print(num_sepsis, "/", i, " are labeled Sepsis")
print("NA in each column:")
print(df_all.isna().sum() / len(df_all))

# plot histogram for each feature
f, axes = plt.subplots(7, 6, figsize=(30, 30), sharex=False)
for i, feature in enumerate(df_all.columns):
    sns.histplot(data=df_all.iloc[:len(df_all) // 2], x=feature, hue='SepsisLabel',
                 ax=axes[i % 7, i // 7])  # ,hue='SepsisLabel'
plt.savefig("histograms.png")

# plot correlation heatmap of features
corr = df_all.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
plt.figure(figsize=(16, 9))
sns.heatmap(corr, mask=mask, xticklabels=corr.columns, yticklabels=corr.columns, annot=False,
            cmap=sns.diverging_palette(220, 20, as_cmap=True),
            vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.tight_layout()
plt.savefig("correlations.png")

# perform chi squared test to test dependence
pvals = {}
for col in list(df_all.columns):
    if col == 'SepsisLabel':
        continue
    df = df_all[[col, "SepsisLabel"]]
    df = df.dropna()
    if df[[col]].min()[0] < 0:
        pvals[col] = chi2(df[[col]]-df[[col]].min(), df["SepsisLabel"])[1][0]
    else:
        pvals[col] = chi2(df[[col]], df["SepsisLabel"])[1][0]
rej = [t[0] for t in pvals.items() if t[1] < 0.05/len(pvals)]
print(rej)
print(len(rej), "/", len(pvals), "hypotheses rejected (dependent features)")


# EACH PATIENT REPRESENTED WITH ONE ROW

i = 0
path = "data/train"
num_sepsis = 0
df_all_list = []  # list of each patient's preprocessed df
for filename in os.listdir(path):
    i += 1
    f = os.path.join(path, filename)
    dataframe = pd.read_csv(f, sep='|')
    # check if current patient is positive
    if 1 in dataframe['SepsisLabel'].unique():
        # all rows until first time SepsisLabel=1
        dataframe = dataframe.loc[:dataframe.loc[:, 'SepsisLabel'].gt(0).idxmax(), :]
        dataframe = dataframe.assign(SepsisLabel=1)
        num_sepsis += 1
    dataframe.drop(['SepsisLabel', 'ICULOS', 'Unit2'], axis=1, inplace=True)
    dataframe = dataframe.fillna(method='ffill').iloc[[-1]]  # fill NaN with latest values and take last row to represent current patient
    df_all_list.append(dataframe)
df_all = pd.concat(df_all_list)
df_all = df_all.reset_index(drop=True)
print(num_sepsis, "/", i, " are labeled Sepsis")
print("NA in each column")
print(df_all.isna().sum() / len(df_all))  # feature is missing for a patient row iff it is in nan in all of its rows
print("Number of patients with no missing values is ", df_all.dropna().shape[0], "/", i, )
