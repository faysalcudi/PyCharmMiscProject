import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
import warnings

warnings.filterwarnings("ignore")

# Veri yükleme
df = pd.read_csv(r"C:\Users\Msi-nb\PycharmProjects\PyCharmMiscProject\Miuul_Proje\winequality-red.csv")

# Kaliteyi kategorilere ayır (string label)
def label_quality(q):
    if q <= 5:
        return "Düşük"
    elif q == 6:
        return "Orta"
    else:
        return "Yüksek"

df["quality_label"] = df["quality"].apply(label_quality)

# Özellik mühendisliği (feature engineering)
df["total_acidity"] = df["fixed acidity"] + df["volatile acidity"]
df["sulfur_ratio"] = df["free sulfur dioxide"] / (df["total sulfur dioxide"] + 1e-6)  # Bölme sıfıra karşı önlem
df["alcohol_to_density"] = df["alcohol"] / df["density"]

# --- REGRESYON VERİLERİ ---
X_reg = df.drop(["quality", "quality_label"], axis=1)
y_reg = df["quality"]

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

reg_models = {
    "RandomForest": RandomForestRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42, n_estimators=200),
    "LightGBM": LGBMRegressor(random_state=42, n_estimators=200),
    "LinearRegression": LinearRegression()
}

reg_results = []

def plot_feature_importance(model, X, name):
    # Feature importance ya da coef varsa çıkar
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        if model.coef_.ndim == 1:
            importance = np.abs(model.coef_)
        else:
            importance = np.abs(model.coef_[0])
    else:
        print(f"{name} için feature importance bulunamadı.")
        return
    feat_df = pd.DataFrame({"Feature": X.columns, "Importance": importance})
    feat_df = feat_df.sort_values(by="Importance", ascending=False)
    plt.figure(figsize=(8, 4))
    plt.bar(feat_df["Feature"], feat_df["Importance"], color="darkred")
    plt.xticks(rotation=45)
    plt.title(f"{name} - Feature Importance")
    plt.tight_layout()
    plt.show()

print("\n--- REGRESYON SONUÇLARI ---")
for name, model in reg_models.items():
    model.fit(X_train_reg, y_train_reg)
    y_pred_reg = model.predict(X_test_reg)
    mse = mean_squared_error(y_test_reg, y_pred_reg)
    r2 = r2_score(y_test_reg, y_pred_reg)
    reg_results.append({"Model": name, "MSE": mse, "R2": r2})
    print(f"{name}: MSE={mse:.4f} | R²={r2:.4f}")
    plot_feature_importance(model, X_reg, name)

reg_df = pd.DataFrame(reg_results)
best_reg = reg_df.loc[reg_df["R2"].idxmax()]
print(f"\nRegresyonda en iyi model: {best_reg['Model']} (R2={best_reg['R2']:.4f})")

reg_df.plot.bar(x="Model", y="R2", legend=False, color="darkred", figsize=(8,4), title="Regresyon Modelleri R² Skorları")
plt.ylabel("R² Skoru")
plt.ylim(0,1)
plt.show()

# --- SINIFLANDIRMA VERİLERİ ---
X_clf = df.drop(["quality", "quality_label"], axis=1)
y_clf = df["quality_label"]

# XGBoost zorunluluğu nedeniyle string label'ları sayısala çeviriyoruz
label_map = {"Düşük": 0, "Orta": 1, "Yüksek": 2}
inv_label_map = {v: k for k, v in label_map.items()}  # Ters dönüşüm haritası

y_clf_num = y_clf.map(label_map)

# Train-test split, stratify ile dengesiz sınıflar için dengeli dağılım sağlanır
X_train_clf, X_test_clf, y_train_clf_num, y_test_clf_num = train_test_split(
    X_clf, y_clf_num, test_size=0.2, random_state=42, stratify=y_clf_num
)

clf_models = {
    "RandomForest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42, n_estimators=200, use_label_encoder=False, eval_metric="mlogloss"),
    "LightGBM": LGBMClassifier(random_state=42, n_estimators=200),
    "LogisticRegression": LogisticRegression(max_iter=1000)
}

clf_results = []

print("\n--- SINIFLANDIRMA SONUÇLARI ---")
for name, model in clf_models.items():
    model.fit(X_train_clf, y_train_clf_num)
    y_pred_clf_num = model.predict(X_test_clf)
    y_pred_clf = pd.Series(y_pred_clf_num, index=X_test_clf.index).map(inv_label_map)
    y_test_clf = pd.Series(y_test_clf_num, index=X_test_clf.index).map(inv_label_map)

    # Indexleri eşit, doğrudan karşılaştırabiliriz
    acc = (y_pred_clf == y_test_clf).mean()
    clf_results.append({"Model": name, "Accuracy": acc})

    print(f"\n{name}: Doğruluk Oranı={acc:.4f}")
    print(classification_report(y_test_clf, y_pred_clf))

    # Confusion Matrix gösterimi
    cm = confusion_matrix(y_test_clf, y_pred_clf, labels=["Düşük", "Orta", "Yüksek"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Düşük", "Orta", "Yüksek"])
    disp.plot(cmap="Reds")
    plt.title(f"{name} - Confusion Matrix")
    plt.show()

    plot_feature_importance(model, X_clf, name)

clf_df = pd.DataFrame(clf_results)
best_clf = clf_df.loc[clf_df["Accuracy"].idxmax()]
print(f"\nSınıflandırmada en iyi model: {best_clf['Model']} (Accuracy={best_clf['Accuracy']:.4f})")

clf_df.plot.bar(x="Model", y="Accuracy", legend=False, color="darkblue", figsize=(8,4), title="Sınıflandırma Modelleri Doğruluk Skorları")
plt.ylabel("Doğruluk")
plt.ylim(0,1)
plt.show()


#Makif

#Random forest grid search
from sklearn.model_selection import GridSearchCV

# Example for Random Forest Regressor
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
grid_search.fit(X_train_reg, y_train_reg)
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Score: {grid_search.best_score_}")

# Random forest cross validation
from sklearn.model_selection import cross_val_score

# Cross-validation for RandomForestRegressor
cv_scores = cross_val_score(RandomForestRegressor(random_state=42), X_reg, y_reg, cv=5, scoring='neg_mean_squared_error')
print(f"RandomForest Cross-validation MSE: {-cv_scores.mean()}")