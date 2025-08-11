import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve

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

# Confusion matrix için y_pred:
y_pred = cart_model.predict(X)

# AUC için y_prob:
y_prob = cart_model.predict_proba(X)[:, 1]

# Confusion matrix
print(classification_report(y, y_pred))

# AUC
roc_auc_score(y, y_prob)

#####################
# Holdout Yöntemi ile Başarı Değerlendirme
#####################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=45)

cart_model = DecisionTreeClassifier(random_state=17).fit(X_train, y_train)

# Train Hatası
y_pred = cart_model.predict(X_train)
y_prob = cart_model.predict_proba(X_train)[:, 1]
print(classification_report(y_train, y_pred))
roc_auc_score(y_train, y_prob)

# Test Hatası
y_pred = cart_model.predict(X_test)
y_prob = cart_model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
roc_auc_score(y_test, y_prob)

#####################
# CV ile Başarı Değerlendirme
#####################

cart_model = DecisionTreeClassifier(random_state=17).fit(X, y)

cv_results = cross_validate(cart_model,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean() # 0.8048706896551725
cv_results['test_f1'].mean() # 0.3260318806493062
cv_results['test_roc_auc'].mean() # 0.618812102142272

################################################
# 4. Hyperparameter Optimization with GridSearchCV
################################################

cart_model.get_params()

cart_params = {'max_depth': range(1, 11),
               "min_samples_split": range(2, 20)}

cart_best_grid = GridSearchCV(cart_model, cart_params, scoring="f1", cv=5, n_jobs=-1, verbose=1).fit(X, y)

cart_best_grid.best_params_

cart_best_grid.best_score_

################################################
# 5. Final Model
################################################

cart_final = DecisionTreeClassifier(**cart_best_grid.best_params_, random_state=17).fit(X, y)
cart_final.get_params()

cart_final = cart_model.set_params(**cart_best_grid.best_params_).fit(X, y)

cv_results = cross_validate(cart_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean() # 0.8805544670846395
cv_results['test_f1'].mean() # 0.44399139048673036
cv_results['test_roc_auc'].mean() # 0.772745128588193
