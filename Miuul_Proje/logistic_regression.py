import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

_df= pd.read_csv("C:/Users/makif/PycharmProjects/PyCharmMiscProject/Miuul_Proje/winequality-red.csv")
df=_df.copy()

df.head()
df["quality"].value_counts()

df['quality'] = np.where(df['quality'].isin([7, 8]), 1, 0)