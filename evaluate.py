"""
Evaluation Module
Trains a simple classifier (Logistic Regression / Naive Bayes) to predict
job-role labels from resume text, and reports Accuracy, Precision, Recall, F1.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from preprocess import preprocess_document


def load_training_data(csv_path: str) -> pd.DataFrame:
    """
    Expects a CSV with columns: resume_text, job_role
    """
    return pd.read_csv(csv_path)


def train_and_evaluate(df: pd.DataFrame, model_type: str = "logistic"):
    df["cleaned"] = df["resume_text"].astype(str).apply(preprocess_document)

    X_train, X_test, y_train, y_test = train_test_split(
        df["cleaned"], df["job_role"], test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    if model_type == "naive_bayes":
        model = MultinomialNB()
    else:
        model = LogisticRegression(max_iter=1000)

    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
    }
    return model, vectorizer, metrics


if __name__ == "__main__":
    # Expects data/resumes_dataset.csv with columns: resume_text, job_role
    data = load_training_data("../data/resumes_dataset.csv")
    _, _, metrics = train_and_evaluate(data)
    print("Evaluation Metrics:", metrics)
