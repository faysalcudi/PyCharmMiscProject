import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Define models
models = {
    'Random Forest': RandomForestClassifier(random_state=1),
    'XGBoost': XGBClassifier(random_state=1),
    'Logistic Regression': LogisticRegression(random_state=1, max_iter=1000)
}

# Train and evaluate models
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy

    print(f"\n{name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.colorbar()
    plt.tight_layout()
    # Instead of plt.show()
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
# Instead of plt.show()
plt.savefig("plot_ModelAccuracy.png", dpi=100, bbox_inches="tight")
plt.close()

# Create new dataframe for outlier analysis
new_df = _df.copy()


# Calculate outlier thresholds
def calculate_limits(dataframe, col, q1=0.05, q3=0.95):
    quartile1 = dataframe[col].quantile(q1)
    quartile3 = dataframe[col].quantile(q3)
    iqr = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * iqr
    low_limit = quartile1 - 1.5 * iqr
    return low_limit, up_limit

cols = new_df.select_dtypes(include="number").columns.tolist()

# Create subplots for before replacement boxplots
plt.figure(figsize=(15, 10))
new_df.boxplot(column=cols)
plt.xticks(rotation=45)
plt.title('Boxplots Before Outlier Replacement')
plt.tight_layout()
# Instead of plt.show()
plt.savefig("plot_BoxPlots_Before.png", dpi=200, bbox_inches="tight")
plt.close()

# Replace outliers
for column in new_df.columns:
    lower_limit, upper_limit = calculate_limits(new_df, column)
    new_df.loc[new_df[column] > upper_limit, column] = upper_limit
    new_df.loc[new_df[column] < lower_limit, column] = lower_limit

# Create subplots for after replacement boxplots
plt.figure(figsize=(15, 10))
new_df.boxplot(column=cols)
plt.xticks(rotation=45)
plt.title('Boxplots After Outlier Replacement')
plt.tight_layout()
# Instead of plt.show()
plt.savefig("plotBoxPlots_After.png", dpi=200, bbox_inches="tight")
plt.close()

# Prepare data for modeling with outlier handled dataset
new_df['quality_category'] = new_df['quality'].apply(label_quality)
X_new = new_df.drop(['quality', 'quality_category'], axis=1)
y_new = new_df['quality_category']

# Split data
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y_new, test_size=0.2, random_state=1)

# Train and evaluate models with new data
results_new = {}
for name, model in models.items():
    model.fit(X_train_new, y_train_new)
    y_pred_new = model.predict(X_test_new)
    accuracy_new = accuracy_score(y_test_new, y_pred_new)
    results_new[name] = accuracy_new

# Compare accuracies
plt.figure(figsize=(12, 6))
x = np.arange(len(results))
width = 0.35

plt.bar(x - width / 2, results.values(), width, label='Original')
plt.bar(x + width / 2, results_new.values(), width, label='After Outlier Handling')

plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison - Before vs After Outlier Handling')
plt.xticks(x, results.keys(), rotation=45)
plt.legend()
plt.tight_layout()
# Instead of plt.show()
plt.savefig("plot_After_Outlier_Accuracy_Comparison.png", dpi=100, bbox_inches="tight")
plt.close()