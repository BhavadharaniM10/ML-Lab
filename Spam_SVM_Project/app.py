import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# ---------------------------
# Load & Train Model
# ---------------------------

@st.cache_data
def load_and_train():
    df = pd.read_csv("spam.csv", encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']

    # Convert labels to binary
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    # TF-IDF Feature Extraction
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['message'])
    y = df['label']

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train SVM Model
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return model, vectorizer, acc, cm


model, vectorizer, acc, cm = load_and_train()

# ---------------------------
# Streamlit UI
# ---------------------------

st.title("📧 Spam Mail Detection using SVM")

st.write("Enter a message below to check whether it is Spam or Not Spam.")

# User Input
message = st.text_area("Enter your message:")

if st.button("Predict"):
    if message:
        data = vectorizer.transform([message])
        prediction = model.predict(data)

        if prediction[0] == 1:
            st.error("🚨 This is SPAM!")
        else:
            st.success("✅ This is NOT Spam.")
    else:
        st.warning("Please enter a message.")

# ---------------------------
# Model Accuracy
# ---------------------------

st.subheader("📊 Model Accuracy")
st.write(f"Accuracy: {acc:.2f}")

# ---------------------------
# Confusion Matrix
# ---------------------------
st.subheader("📉 Confusion Matrix")

fig, ax = plt.subplots()

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Ham", "Spam"]
)

disp.plot(
    ax=ax,
    cmap="Blues",      # Light color theme
    colorbar=False
)

plt.title("Confusion Matrix")
st.pyplot(fig)