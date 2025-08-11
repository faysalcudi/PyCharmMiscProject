import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

#Python console pandas database görünümü ayarları
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

#Veri ön işleme
X_scaled = StandardScaler().fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

#KNN modeli oluştur
knn_model = KNeighborsClassifier().fit(X, y)

#Random bir data oluştur ve test et
random_data = X.sample(1, random_state=45)
knn_model.predict(random_data)

#Confusion matrix için y_pred
y_pred = knn_model.predict(X)

#AUC için y_prob
y_prob = knn_model.predict_proba(X)[:, 1]

print(classification_report(y, y_pred))
# accuracy 0.92
# f1 0.65

#AUC
roc_auc_score(y, y_prob) #0.9560444690457295

#Cross Validation
cv_results = cross_validate(knn_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean() #0.8417848746081505
cv_results['test_f1'].mean() #0.36789631883529716
cv_results['test_roc_auc'].mean() #0.7609701589489636

#KNN parametreleri incele
knn_model.get_params()

#Hyperparameter optimization
knn_params = {"n_neighbors": range(2,50)}
knn_gs_best = GridSearchCV(knn_model, knn_params, cv=5, n_jobs=-1, verbose=1).fit(X, y)
knn_gs_best.best_params_

#Final Model
knn_final = knn_model.set_params(**knn_gs_best.best_params_).fit(X, y)

cv_results = cross_validate(knn_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean() #0.8667946708463949
cv_results['test_f1'].mean() #0.31229642617259395
cv_results['test_roc_auc'].mean() #0.830919327817678

random_user = X.sample(1)

knn_final.predict(random_user)