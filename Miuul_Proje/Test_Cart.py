import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve
from imblearn.over_sampling import SMOTE

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

#Resampling
y_train.value_counts()
oversample = SMOTE()
X_smote, y_smote = oversample.fit_resample(X_train, y_train)
y_smote.value_counts()

cart_model = DecisionTreeClassifier(random_state=17).fit(X_train, y_train)
# cart_model = DecisionTreeClassifier(random_state=17).fit(X_smote, y_smote)

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

################################################
# 6. Feature Importance
################################################

cart_final.feature_importances_

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(cart_final, X, num=5)
################################################
# 7. Analyzing Model Complexity with Learning Curves (BONUS)
################################################


train_score, test_score = validation_curve(cart_final, X, y, param_name="max_depth",
                                           param_range=range(1, 11), scoring="roc_auc", cv=10)

mean_train_score = np.mean(train_score, axis=1)
mean_test_score = np.mean(test_score, axis=1)

plt.plot(range(1, 11), mean_train_score, label="Training Score", color='b')
plt.plot(range(1, 11), mean_test_score, label="Validation Score", color='g')
plt.title("Validation Curve for CART")
plt.xlabel("Number of max_depth")
plt.ylabel("AUC")
plt.tight_layout()
plt.legend(loc='best')
plt.show()

#fonksiyonlaştırılmışı
"""def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
    train_score, test_score = validation_curve(
        model, X=X, y=y, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv)

    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score,
             label="Training Score", color='b')

    plt.plot(param_range, mean_test_score,
             label="Validation Score", color='g')

    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show(block=True)


val_curve_params(cart_final, X, y, "max_depth", range(1, 11), scoring="f1")

cart_val_params = [["max_depth", range(1, 11)], ["min_samples_split", range(2, 20)]]

for i in range(len(cart_val_params)):
    val_curve_params(cart_model, X, y, cart_val_params[i][0], cart_val_params[i][1])"""

################################################
# 9. Extracting Decision Rules
################################################

tree_rules = export_text(cart_final, feature_names=list(X.columns))
print(tree_rules)