import pandas as pd

dataframe=pd.read_csv("./pandas_for_machineLearning/industrial_maintenance_with_missing_values.csv")
print(dataframe.shape)
print(dataframe.columns)
print(dataframe.head())
print(dataframe.dtypes)
print(dataframe.info())
print(dataframe.describe())