import pandas as pd
import matplotlib.pyplot as plt
dataframe=pd.read_csv("./pandas_for_machineLearning/industrial_maintenance_with_missing_values.csv")

print(dataframe.describe())
dataframe["operating_hours"].hist()
plt.ylabel("equipment")
plt.xlabel("hours")
plt.title("equipment operating hours")
plt.show()