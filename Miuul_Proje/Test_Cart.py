import pandas as pd
from sklearn.tree import DecisionTreeClassifier

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

#Veri Setini oku ve yedekten kopyala
_df = pd.read_csv("C:/Users/makif/PycharmProjects/PyCharmMiscProject/Miuul_Proje/winequality-red.csv")
df = _df.copy()

#Keşifçi Veri Analizi
df.head()
df.shape
df.describe(percentiles=[.05,.25,.5,.75,.95]).T
df["quality"].value_counts()

df['quality'] = df['quality'].isin([7, 8]).astype(int)

#Model için X ve y oluştur
y = df["quality"]
X = df.drop(["quality"], axis=1)

#CART Model
cart_model = DecisionTreeClassifier(random_state=1).fit(X, y)