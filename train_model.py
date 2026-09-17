# 1. Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# 2. Load Dataset
data = pd.read_csv("loan_prediction.csv")

# 3. Handle Missing Values
data.fillna(data.mode().iloc[0], inplace=True)

# 4. Encode Categorical Columns
categorical_cols = ["Gender", "Married", "Dependents", "Education",
                    "Self_Employed", "Property_Area", "Loan_Status"]

encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    encoders[col] = le

# 5. Feature & Target Split
X = data.drop(["Loan_ID", "Loan_Status"], axis=1)
y = data["Loan_Status"]

# 6. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# 7. Train Multiple Models and Compare
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=0),
    "XGBoost": XGBClassifier()
}

results = {}
trained_models = {}

for name, clf in models.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    results[name] = acc
    trained_models[name] = clf
    
    print(f"\n===== {name} =====")
    print("Accuracy Score:", acc)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 8. Select Best Model
best_model_name = max(results, key=results.get)
best_model = trained_models[best_model_name]

print("\n\n========================================")
print(f"BEST MODEL: {best_model_name} with Accuracy: {results[best_model_name]:.4f}")
print("========================================")

# 9. Save Best Trained Model and Encoders
with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

print("\nModel and Encoders saved successfully!")

# 10. Optional EDA Plots
sns.histplot(data["ApplicantIncome"], kde=True)
plt.title("Applicant Income Distribution")
plt.xlabel("Income")
plt.ylabel("Count")
plt.show()

sns.countplot(x="Loan_Status", data=data)
plt.title("Loan Status Distribution")
plt.xlabel("Status (0 = Rejected, 1 = Approved)")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()