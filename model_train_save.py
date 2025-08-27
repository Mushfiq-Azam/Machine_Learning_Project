# Full pipeline from data insight to model comparison
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier

import warnings
warnings.filterwarnings("ignore")

# Load data
df = pd.read_csv("2.csv")
print(df['life_satisfaction'].value_counts())


# Step 1: Data Insight
print(df.info())
print(df.describe(include='all'))

# Step 2: Numerical Histograms
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
for col in numerical_features[:5]:  # limit to first 5
    df[col].plot(kind='hist', bins=30, color='skyblue', title=col)
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

# Step 3: Categorical Bar Charts
categorical_features = df.select_dtypes(include='object').columns.tolist()
for col in categorical_features[:5]:
    df[col].value_counts().plot(kind='bar', title=col)
    plt.xticks(rotation=45)
    plt.show()

# Step 4: Bar Plot Categorical vs Categorical
pd.crosstab(df['A2'], df['life_satisfaction']).plot(kind='bar', stacked=True)
plt.title("A2 vs Life Satisfaction")
plt.ylabel("Count")
plt.show()

# Step 5: Categorical vs Numerical
sns.boxplot(data=df, x='life_satisfaction', y='age', palette='Set2')
plt.title("Age vs Life Satisfaction")
plt.show()

# Step 6: Pair Plot
numeric_df = df.select_dtypes(include=['float64', 'int64'])
sns.pairplot(numeric_df)
plt.show()

# Step 7: Missing Data Handling
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df.select_dtypes(include='object').columns

# Numeric missing values -> replace with mean
df[num_cols] = SimpleImputer(strategy='mean').fit_transform(df[num_cols])

# Categorical missing values -> replace with most frequent value
df[cat_cols] = SimpleImputer(strategy='most_frequent').fit_transform(df[cat_cols])


print("After handling missing values:")
print(df.isnull().sum())
print(df.head())

# Step 8: Define X, y

X = df.drop('life_satisfaction', axis=1)
y = df['life_satisfaction']

# ✅ Step 9: One-hot encode with get_dummies
X_encoded = pd.get_dummies(X, drop_first=True)
# View encoded data
print("🔹 After One-Hot Encoding:")
print(X_encoded.head())
print("Shape:", X_encoded.shape)

# Step 10: Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Step 11: SMOTE + Tomek Links
sm = SMOTE(random_state=42)
tl = TomekLinks()
X_resampled, y_resampled = sm.fit_resample(X_scaled, y)
X_cleaned, y_cleaned = tl.fit_resample(X_resampled, y_resampled)



# View scaled data
X_scaled_df = pd.DataFrame(X_scaled, columns=X_encoded.columns)
print("🔹 After Standardization:")
print(X_scaled_df.head())
print("Shape:", X_scaled_df.shape)

# View resampled data
X_cleaned_df = pd.DataFrame(X_cleaned, columns=X_encoded.columns)
print("🔹 After SMOTE + Tomek Links:")
print(X_cleaned_df.head())
print("Shape:", X_cleaned_df.shape)


# Step 12: Correlation Matrix
X_cleaned_df = pd.DataFrame(X_cleaned, columns=X_encoded.columns)
corr_matrix = X_cleaned_df.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
plt.title("Feature Correlation Matrix")
plt.show()

# Step 13: Outlier Removal
iso = IsolationForest(contamination=0.01, random_state=42)
outliers = iso.fit_predict(X_cleaned)
mask = outliers != -1
X_clean, y_clean = X_cleaned[mask], y_cleaned[mask]

# Step 14: Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y_clean)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y_encoded, test_size=0.3, stratify=y_encoded, random_state=42
)



# Step 15: Classification Models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
    "KNN": KNeighborsClassifier()
}

results = {
    "Accuracy": {},
    "Precision": {},
    "Recall": {},
    "F1-Score": {},
}

# Model training and evaluation
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results["Accuracy"][name] = accuracy_score(y_test, y_pred)
    results["Precision"][name] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    results["Recall"][name] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    results["F1-Score"][name] = f1_score(y_test, y_pred, average='weighted', zero_division=0)

# Step 16: Show Comparison Table
comparison_df = pd.DataFrame(results)
comparison_df = comparison_df.sort_values("Accuracy", ascending=False)
print(comparison_df.round(4))

# Step 17: Plot Evaluation
sns.set(style="whitegrid")
metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]

for metric in metrics:
    plt.figure(figsize=(10, 5))
    sns.barplot(
        x=comparison_df.index, 
        y=metric, 
        data=comparison_df, 
        palette="viridis"
    )
    plt.title(f"{metric} by Model", fontsize=16)
    plt.ylabel(metric)
    plt.xlabel("Model")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



# Save the model
import joblib

# Pick the best model
best_model_name = comparison_df.index[0]
best_model = models[best_model_name]

# Save model + preprocessing tools
joblib.dump(best_model, "life_satisfaction_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le, "label_encoder.pkl")
joblib.dump(X_encoded.columns, "X_columns.pkl")

print(f"✅ Saved best model: {best_model_name}")


# === Explainable AI (SHAP + LIME) ===
import shap
import lime
import lime.lime_tabular

# Use test set for explanations
explainer_shap = shap.Explainer(best_model, X_train)
shap_values = explainer_shap(X_test[:50])  # limit for speed

# Save SHAP explainer + values
joblib.dump(explainer_shap, "shap_explainer.pkl")
joblib.dump(shap_values, "shap_values.pkl")

# LIME explainer
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X_encoded.columns,
    class_names=le.classes_,
    mode="classification"
)

# Example explanation for first instance
lime_exp = lime_explainer.explain_instance(
    X_test[0],
    best_model.predict_proba,
    num_features=10
)

# Save LIME explainer + sample explanation
# ❌ Do NOT save the explainer object (not pickleable)
# ✅ Save only explanation output
lime_results = lime_exp.as_list()   # list of (feature, weight)
joblib.dump(lime_results, "lime_example.pkl")

