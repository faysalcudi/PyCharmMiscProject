import warnings
import joblib
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE

warnings.simplefilter(action='ignore', category=Warning)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

################################################
# 1. Exploratory Data Analysis
################################################

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")

def correlation_matrix(df, cols):
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(df[cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 11}, linecolor='w', cmap='RdBu')
    plt.show(block=True)

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # print(f"Observations: {dataframe.shape[0]}")
    # print(f"Variables: {dataframe.shape[1]}")
    # print(f'cat_cols: {len(cat_cols)}')
    # print(f'num_cols: {len(num_cols)}')
    # print(f'cat_but_car: {len(cat_but_car)}')
    # print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

_df = pd.read_csv("C:/Users/makif/PycharmProjects/PyCharmMiscProject/Miuul_Proje/winequality-red.csv")
df = _df.copy()


# Değişken isimleri büyütmek
# df.columns = [col.upper() for col in df.columns]
# Columnlarda boşlukları alt tire ile değiştir.
df.columns = (df.columns.str.replace(r"\s+", "_", regex=True))

check_df(df)

df['quality'] = df['quality'].isin([7, 8]).astype(int)
df["quality"].value_counts()

cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=7, car_th=20)

# for col in cat_cols:
#     cat_summary(df, col, plot=True)

df[num_cols].describe().T

# for col in num_cols:
#     num_summary(df, col, plot=True)

#correlation_matrix(df, num_cols)

for col in num_cols:
    target_summary_with_num(df, "quality", col)

################################################
# 2. Data Preprocessing & Feature Engineering
################################################

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def check_outlier(dataframe, col_name, q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df.head()

# Feature exraction denemesi (XGBoost f1 score düştü)
# df["total_acidity"] = df["fixed_acidity"] + df["volatile_acidity"]
# df['has_citric_acid'] = pd.cut(x=df['citric_acid'], bins=[-1, 0, 1], labels=[0, 1]).astype("int32")
# cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=7, car_th=20)
# for col in cat_cols:
#     target_summary_with_cat(df, "quality", col)

cat_cols = [col for col in cat_cols if "quality" not in col]

for col in num_cols:
    print(col, check_outlier(df, col, 0.05, 0.95))

replace_with_thresholds(df, "residual_sugar")
replace_with_thresholds(df, "chlorides")
replace_with_thresholds(df, "total_sulfur_dioxide")
replace_with_thresholds(df, "sulphates")

X_scaled = StandardScaler().fit_transform(df[num_cols])
df[num_cols] = pd.DataFrame(X_scaled, columns=df[num_cols].columns)

y = df["quality"]
X = df.drop("quality", axis=1)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)

#Resampling
y_train.value_counts()
oversample = SMOTE()
X_smote, y_smote = oversample.fit_resample(X_train, y_train)
y_smote.value_counts()

######################################################
# 3. Base Models
######################################################

def base_models(X, y, X_train, X_test, y_train, y_test, scoring="f1"):
    print("Base Models....")
    classifiers = [
                   ('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   ("SVC", SVC()),
                   ("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   ('Adaboost', AdaBoostClassifier()),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                   ('LightGBM', LGBMClassifier()),
                   # ('CatBoost', CatBoostClassifier(verbose=False))
                   ]

    print("Cross Validation Results:")
    scores = []
    names = []
    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=5, scoring=scoring)
        scores.append(cv_results['test_score'].mean())
        names.append(name)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")

    plt.figure(figsize=(10, 6))
    plt.bar(names, scores)
    plt.xticks(rotation=45)
    plt.ylabel(f'{scoring} Score')
    plt.title('Model Performance Comparison')
    plt.tight_layout()
    plt.show()

    print("\nTrain-Test Split Results:")
    for name, classifier in classifiers:
        # Train model
        classifier.fit(X_train, y_train)
        # Make predictions
        y_pred = classifier.predict(X_test)
        # Calculate metrics
        if scoring == "f1":
            score = f1_score(y_test, y_pred)
        elif scoring == "accuracy":
            score = accuracy_score(y_test, y_pred)
        elif scoring == "roc_auc":
            y_prob = classifier.predict_proba(X_test)[:, 1]
            score = roc_auc_score(y_test, y_prob)
        print(f"{scoring}: {round(score, 4)} ({name}) ")

base_models(X, y, X_train, X_test, y_train, y_test, scoring="f1")

######################################################
# 4. Automated Hyperparameter Optimization
######################################################

knn_params = {"n_neighbors": range(2, 50)}

cart_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2, 30)}

rf_params = {"max_depth": [8, 15, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [15, 20],
             "n_estimators": [200, 300],
             "class_weight": ["balanced"]}

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 200],
                  "scale_pos_weight": [5]}

lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [300, 500]}


classifiers = [
               ('KNN', KNeighborsClassifier(), knn_params),
               ("CART", DecisionTreeClassifier(), cart_params),
               ("RF", RandomForestClassifier(), rf_params),
               ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgboost_params),
               ('LightGBM', LGBMClassifier(), lightgbm_params)
               ]


def hyperparameter_optimization(X, y, cv=5, scoring="f1"):
    print("Hyperparameter Optimization....")
    best_models = {}
    before_scores = []
    after_scores = []
    model_names = []

    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        before_score = cv_results['test_score'].mean()
        print(f"{scoring} (Before): {round(before_score, 4)}")

        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        after_score = cv_results['test_score'].mean()
        print(f"{scoring} (After): {round(after_score, 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")

        model_names.append(name)
        before_scores.append(before_score)
        after_scores.append(after_score)
        best_models[name] = final_model

    plt.figure(figsize=(10, 6))
    x = range(len(model_names))
    width = 0.35

    plt.bar([i - width / 2 for i in x], before_scores, width, label='Before', color='skyblue')
    plt.bar([i + width / 2 for i in x], after_scores, width, label='After', color='lightgreen')

    plt.xlabel('Models')
    plt.ylabel(scoring.capitalize() + ' Score')
    plt.title('Model Performance Comparison - Before vs After Optimization')
    plt.xticks(x, model_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return best_models

best_models = hyperparameter_optimization(X, y, scoring="f1")

#Feature Importance
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num],
                palette=sns.dark_palette("red", n_colors=len(feature_imp), reverse=True))
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plotmodel = best_models["XGBoost"].fit(X, y)
plot_importance(plotmodel, X)

"""ayıraç"""
######################################################
# 5. Stacking & Ensemble Learning
######################################################

def voting_classifier(best_models, X, y):
    print("Voting Classifier...")

    voting_clf = VotingClassifier(estimators=[('XGBoost', best_models["XGBoost"]),
                                              ('RF', best_models["RF"]),
                                              ('LightGBM', best_models["LightGBM"])],
                                  voting='soft').fit(X, y)

    cv_results = cross_validate(voting_clf, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
    print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
    print(f"F1Score: {cv_results['test_f1'].mean()}")
    print(f"ROC_AUC: {cv_results['test_roc_auc'].mean()}")
    return voting_clf

voting_clf = voting_classifier(best_models, X, y)
# Accuracy: 0.8768025078369905
# F1Score: 0.39213852455787934
# ROC_AUC: 0.8561607338308086
"ayıraç"

######################################################
# 6. Prediction for a New Observation
######################################################
cv_results = cross_validate(best_models["XGBoost"].fit(X, y), X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
print(f"F1Score: {cv_results['test_f1'].mean()}")
print(f"ROC_AUC: {cv_results['test_roc_auc'].mean()}")
random_data = X.sample(1, random_state=45)
best_models["XGBoost"].fit(X, y).predict(random_data)
voting_clf.predict(random_data)


#Modeli kaydetmek için
# joblib.dump(voting_clf, "voting_clf_2.pkl")
#
# new_model = joblib.load("voting_clf_1.pkl")
# new_model.predict(random_user)