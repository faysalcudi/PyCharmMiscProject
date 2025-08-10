import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

_df = pd.read_csv("C:/Users/makif/PycharmProjects/PyCharmMiscProject/Miuul_Proje/winequality-red.csv")
df = _df.copy()


# Convert quality to categories
def label_quality(q):
    if q >= 7:
        return 0
    elif q == 6:
        return 1
    else:
        return 2


df['quality_category'] = df['quality'].apply(label_quality)

# Prepare data for modeling
X = df.drop(['quality', 'quality_category'], axis=1)
y = df['quality_category']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

# Define model parameters for grid search
param_grid = {
    'Random Forest': {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10, 20],
        'classifier__min_samples_split': [2, 5]
    },
    'XGBoost': {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [3, 5],
        'classifier__learning_rate': [0.01, 0.1]
    },
    'Logistic Regression': {
        'classifier__C': [0.1, 1.0, 10.0],
        'classifier__penalty': ['l2']
    }
}

# Define base models
base_models = {
    'Random Forest': RandomForestClassifier(random_state=1),
    'XGBoost': XGBClassifier(random_state=1),
    'Logistic Regression': LogisticRegression(random_state=1, max_iter=1000)
}

# Create pipelines with scaling
models = {}
for name, model in base_models.items():
    models[name] = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', model)
    ])

# Train and evaluate models
results = {}
for name, model in models.items():
    # Perform grid search with cross validation
    grid_search = GridSearchCV(model, param_grid[name], cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Get best model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy

    print(f"\n{name} Results:")
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Cross-validation scores: {cross_val_score(best_model, X_train, y_train, cv=5)}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"plot_ConfusionMatrix_{name}.png", dpi=100, bbox_inches="tight")
    plt.close()

# Plot accuracy comparison
plt.figure(figsize=(10, 6))
plt.bar(results.keys(), results.values())
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plot_ModelAccuracy.png", dpi=100, bbox_inches="tight")
plt.close()

cols = df.select_dtypes(include="number").columns.tolist()

# Create subplots for before replacement boxplots
plt.figure(figsize=(15, 10))
df.boxplot(column=cols)
plt.xticks(rotation=45)
plt.title('Boxplots Before Outlier Replacement')
plt.tight_layout()
plt.savefig("plot_BoxPlots_Before.png", dpi=200, bbox_inches="tight")
plt.close()