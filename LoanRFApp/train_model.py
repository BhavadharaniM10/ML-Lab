import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
data = pd.read_csv("loan_data.csv")

# Drop unnecessary column
data.drop("Loan_ID", axis=1, inplace=True)

# Fill missing values
data.fillna(method='ffill', inplace=True)

# Encode categorical variables
encoders = {}
for column in data.columns:
    if data[column].dtype == 'object':
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        encoders[column] = le

# Split features and target
X = data.drop("Loan_Status", axis=1)
y = data["Loan_Status"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Feature Importance Plot
plt.figure(figsize=(10,6))
plt.barh(X.columns, model.feature_importances_)
plt.title("Feature Importance - Random Forest")
plt.xlabel("Importance")
plt.savefig("feature_importance.png")

# Save model and encoders
pickle.dump(model, open("loan_model.pkl", "wb"))
pickle.dump(encoders, open("encoders.pkl", "wb"))

print("Random Forest Model & Encoders Saved Successfully!")
