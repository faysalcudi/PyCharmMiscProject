import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve,
                             ConfusionMatrixDisplay)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings

warnings.filterwarnings("ignore")

# 1. Veri Yükleme ve Ön İşleme
df = pd.read_csv("red")


# Kalite etiketlerini 3 sınıfa dönüştürme
def quality_label(q):
    if q <= 5:
        return 0  # Düşük
    elif q == 6:
        return 1  # Orta
    else:
        return 2  # Yüksek


df["quality_label"] = df["quality"].apply(quality_label)

# Özellik mühendisliği
df["total_acidity"] = df["fixed acidity"] + df["volatile acidity"]
df["sulfur_ratio"] = df["free sulfur dioxide"] / (df["total sulfur dioxide"] + 1e-6)
df["alcohol_to_density"] = df["alcohol"] / df["density"]

# 2. Görselleştirmeler
plt.figure(figsize=(15, 20))

# Kalite dağılımı
plt.subplot(4, 2, 1)
sns.countplot(x='quality_label', data=df, palette='Reds')
plt.title('Şarap Kalite Dağılımı')

# Alkol vs Kalite
plt.subplot(4, 2, 2)
sns.boxplot(x='quality_label', y='alcohol', data=df, palette='Reds')
plt.title('Alkol İçeriği ve Kalite İlişkisi')

# Uçucu Asitlik vs Kalite
plt.subplot(4, 2, 3)
sns.boxplot(x='quality_label', y='volatile acidity', data=df, palette='Reds')
plt.title('Uçucu Asitlik ve Kalite İlişkisi')

# Toplam Asitlik vs Kalite
plt.subplot(4, 2, 4)
sns.boxplot(x='quality_label', y='total_acidity', data=df, palette='Reds')
plt.title('Toplam Asitlik ve Kalite İlişkisi')

# Korelasyon Matrisi
plt.subplot(4, 2, 5)
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".1f")
plt.title('Özellikler Arası Korelasyon Matrisi')

# Özellik Önem Dereceleri (Model sonrasında güncellenecek)
plt.subplot(4, 2, 6)
importance = pd.Series(np.random.rand(len(df.columns) - 2),
                       index=df.drop(['quality', 'quality_label'], axis=1).columns)
importance.sort_values().plot(kind='barh', color='darkred')
plt.title('Özellik Önem Dereceleri (Placeholder)')

plt.tight_layout()
plt.show()

# 3. Modelleme
X = df.drop(["quality", "quality_label"], axis=1)
y = df["quality_label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)




    #################MODEL REGRESYON##############

    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from sklearn.metrics import mean_squared_error, r2_score

    # Veri yükleme
    df = pd.read_csv("winequality-red.csv")

    # Bağımsız / bağımlı değişkenler
    X = df.drop("quality", axis=1)
    y = df["quality"]

    # Eğitim - test bölme
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modellsınıf
    reg_models = {
        "RandomForest": RandomForestRegressor(random_state=42),
        "XGBoost": XGBRegressor(random_state=42, n_estimators=200),
        "LightGBM": LGBMRegressor(random_state=42, n_estimators=200),
        "LinearRegression": LinearRegression()
    }

    print("\n--- REGRESYON SONUÇLARI ---")
    for name, model in reg_models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"{name}: MSE={mse:.4f} | R²={r2:.4f}")

        # Feature importance (LinearRegression hariç)
     if hasattr(model, "feature_importances_"):
         importances = model.feature_importances_
        feat_df = pd.DataFrame({"Feature": X.columns, "Importance": importances})
      feat_df = feat_df.sort_values(by="Importance", ascending=False)

            plt.figure(figsize=(8, 4))
            plt.bar(feat_df["Feature"], feat_df["Importance"])
            plt.xticks(rotation=45)
            plt.title(f"{name} - Feature Importance (Regresyon)")
            plt.tight_layout()
            plt.show()


####################MODEL SINIFLANDIRMA###########
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.linear_model import LogisticRegression
            from xgboost import XGBClassifier
            from lightgbm import LGBMClassifier
            from sklearn.metrics import classification_report, confusion_matrix
            import matplotlib.pyplot as plt


            # Kaliteyi kategorilere ayırma
            def label_quality(q):
                if q <= 5:
                    return "düşük"
                elif q == 6:
                    return "orta"
                else:
                    return "yüksek"


            df["quality_label"] = df["quality"].apply(label_quality)

            # Bağımsız / bağımlı değişkenler
            X = df.drop(["quality", "quality_label"], axis=1)
            y = df["quality_label"]

            # Eğitim - test bölme
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Modeller
            clf_models = {
                "RandomForest": RandomForestClassifier(random_state=42),
                "XGBoost": XGBClassifier(random_state=42, n_estimators=200, use_label_encoder=False,
                                         eval_metric="mlogloss"),
                "LightGBM": LGBMClassifier(random_state=42, n_estimators=200),
                "LogisticRegression": LogisticRegression(max_iter=1000)
            }

            print("\n--- SINIFLANDIRMA SONUÇLARI ---")
            for name, model in clf_models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc = (y_pred == y_test).mean()
                print(f"\n{name}: Doğruluk Oranı={acc:.4f}")
                print(classification_report(y_test, y_pred))
                print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

                # Feature importance (LogisticRegression hariç)
                if hasattr(model, "feature_importances_"):
                    importances = model.feature_importances_
                    feat_df = pd.DataFrame({"Feature": X.columns, "Importance": importances})
                    feat_df = feat_df.sort_values(by="Importance", ascending=False)

                    plt.figure(figsize=(8, 4))
                    plt.bar(feat_df["Feature"], feat_df["Importance"])
                    plt.xticks(rotation=45)
                    plt.title(f"{name} - Feature Importance (Sınıflandırma)")
                    plt.tight_layout()
                    plt.show()