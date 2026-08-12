import pandas as pd

dataframe=pd.read_csv("./pandas_for_machineLearning/industrial_maintenance_with_missing_values.csv")
# print(dataframe.count())
# print(dataframe.isnull().sum())
# dataframe=dataframe.dropna()
# print(dataframe.isnull().sum())
# print(dataframe.count())

# filling nan's
mean=dataframe['operating_hours'].mean()
dataframe['operating_hours']=dataframe['operating_hours'].fillna(mean)
print(dataframe.isnull().sum())
print(dataframe.duplicated().sum())