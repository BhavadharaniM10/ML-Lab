import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import pickle

# Load dataset
data = pd.read_csv("loan_data.csv")

# Drop Loan_ID
data = data.drop("Loan_ID", axis=1)

# Fill missing values
data.fillna(method='ffill', inplace=True)

# Encode categorical columns
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

# Train Decision Tree
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# ------------------ Evaluation ------------------

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ------------------ Tree Visualization ------------------

plt.figure(figsize=(20,10))
plot_tree(model, feature_names=X.columns,
          class_names=["Rejected", "Approved"],
          filled=True)
plt.show()

# ------------------ Save Model ------------------

pickle.dump(model, open("loan_model.pkl", "wb"))
pickle.dump(encoders, open("encoders.pkl", "wb"))

print("\nModel and Encoders saved successfully!")
