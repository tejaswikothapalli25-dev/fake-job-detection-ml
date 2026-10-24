import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Loading dataset...")

# Load dataset
df = pd.read_csv("fake_job_postings.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())


# Text columns
text_columns = [
    "title",
    "location",
    "department",
    "company_profile",
    "description",
    "requirements",
    "benefits"
]

# Use only columns available in dataset
available_columns = [
    col for col in text_columns
    if col in df.columns
]

print("Using columns:", available_columns)


# Fill missing values
df[available_columns] = df[available_columns].fillna("")


# Combine all text information
df["combined_text"] = df[available_columns].agg(
    " ".join,
    axis=1
)


# Input and target
X = df["combined_text"]
y = df["fraudulent"]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# TF-IDF
print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=50000,
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF completed!")


# Train model
print("\nTraining Logistic Regression model...")

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_train_tfidf, y_train)

print("Model training completed!")


# Evaluate
y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("MODEL EVALUATION")
print("==============================")

print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# Save model
joblib.dump(model, "fake_job_model.pkl")

# Save vectorizer
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("\n==============================")
print("FILES SAVED SUCCESSFULLY")
print("==============================")

print("fake_job_model.pkl")
print("tfidf_vectorizer.pkl")

print("\nYour ML model is ready!")python train_model.py