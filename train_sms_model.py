import os
import pickle
import re
import string

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "sms_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "sms_vectorizer.pkl")

TEXT_LABEL_CANDIDATES = [
    ("Msg", "Label"),
    ("message", "label"),
    ("Message", "Category"),
    ("v2", "v1"),
    ("text", "target"),
]

LIKELY_TEXT_NAMES = {
    "msg",
    "message",
    "text",
    "sms",
    "body",
    "content",
    "v2",
}

LIKELY_LABEL_NAMES = {
    "label",
    "category",
    "class",
    "target",
    "type",
    "spam",
    "v1",
}

LABEL_MAP = {
    "spam": "spam",
    "ham": "ham",
    "legitimate": "ham",
    "legit": "ham",
    "safe": "ham",
    "normal": "ham",
    "0": "ham",
    "1": "spam",
}


def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_name(name):
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def locate_dataset():
    preferred_names = [
        "spam.csv",
        "india_sms_spam.csv",
        "sms_spam.csv",
        "sms.csv",
    ]

    for filename in preferred_names:
        candidate = os.path.join(BASE_DIR, filename)
        if os.path.exists(candidate):
            return candidate

    csv_files = sorted(
        [
            os.path.join(BASE_DIR, filename)
            for filename in os.listdir(BASE_DIR)
            if filename.lower().endswith(".csv")
        ]
    )
    if csv_files:
        return csv_files[0]

    attached_fallback = "/Users/architatiwari/Desktop/spam.csv"
    if os.path.exists(attached_fallback):
        return attached_fallback

    return None


def load_dataset(dataset_path):
    read_attempts = [
        {"encoding": "utf-8"},
        {"encoding": "latin-1"},
        {"encoding": "cp1252"},
    ]

    last_error = None
    for options in read_attempts:
        try:
            return pd.read_csv(dataset_path, **options)
        except Exception as exc:
            last_error = exc

    raise last_error


def pick_schema(dataframe):
    columns = list(dataframe.columns)
    normalized_lookup = {normalize_name(column): column for column in columns}

    for text_name, label_name in TEXT_LABEL_CANDIDATES:
        text_column = normalized_lookup.get(normalize_name(text_name))
        label_column = normalized_lookup.get(normalize_name(label_name))
        if text_column and label_column:
            return text_column, label_column

    text_column = None
    label_column = None

    for column in columns:
        normalized = normalize_name(column)
        if not text_column and normalized in LIKELY_TEXT_NAMES:
            text_column = column
        if not label_column and normalized in LIKELY_LABEL_NAMES:
            label_column = column

    if text_column and label_column and text_column != label_column:
        return text_column, label_column

    cleaned = dataframe.copy()
    cleaned = cleaned.loc[:, ~cleaned.columns.astype(str).str.contains(r"^Unnamed", case=False, na=False)]

    object_columns = [
        column
        for column in cleaned.columns
        if cleaned[column].dropna().shape[0] > 0 and cleaned[column].dtype == "object"
    ]

    for candidate_label in object_columns:
        unique_values = {
            str(value).strip().lower()
            for value in cleaned[candidate_label].dropna().unique()
        }
        if unique_values and unique_values.issubset(set(LABEL_MAP.keys())):
            other_columns = [column for column in object_columns if column != candidate_label]
            if other_columns:
                best_text = max(
                    other_columns,
                    key=lambda column: cleaned[column].astype(str).str.len().mean(),
                )
                return best_text, candidate_label

    if len(object_columns) >= 2:
        ranked = sorted(
            object_columns,
            key=lambda column: cleaned[column].astype(str).str.len().mean(),
            reverse=True,
        )
        return ranked[0], ranked[-1]

    return None, None


def normalize_dataframe(dataframe, text_column, label_column):
    normalized = dataframe[[text_column, label_column]].copy()
    normalized.columns = ["message", "label"]
    normalized["message"] = normalized["message"].fillna("").astype(str)
    normalized["label"] = (
        normalized["label"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .map(LABEL_MAP)
    )
    normalized["message"] = normalized["message"].apply(clean_text)
    normalized = normalized[
        normalized["message"].ne("") & normalized["label"].isin(["spam", "ham"])
    ]
    return normalized


def main():
    dataset_path = locate_dataset()
    if not dataset_path:
        print("Dataset CSV not found")
        return

    try:
        dataframe = load_dataset(dataset_path)
    except Exception:
        print("Dataset format unsupported")
        return

    text_column, label_column = pick_schema(dataframe)
    if not text_column or not label_column:
        print("Dataset format unsupported")
        return

    normalized = normalize_dataframe(dataframe, text_column, label_column)
    if normalized.empty or normalized["label"].nunique() < 2:
        print("Dataset format unsupported")
        return

    x_train, x_test, y_train, y_test = train_test_split(
        normalized["message"],
        normalized["label"],
        test_size=0.2,
        random_state=42,
        stratify=normalized["label"],
    )

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    x_train_vectorized = vectorizer.fit_transform(x_train)
    x_test_vectorized = vectorizer.transform(x_test)

    model = MultinomialNB()
    model.fit(x_train_vectorized, y_train)

    predictions = model.predict(x_test_vectorized)
    y_test_binary = (y_test == "spam").astype(int)
    predictions_binary = (predictions == "spam").astype(int)

    print(f"Dataset: {dataset_path}")
    print(f"Detected text column: {text_column}")
    print(f"Detected label column: {label_column}")
    print(f"Accuracy: {accuracy_score(y_test_binary, predictions_binary):.4f}")
    print(f"Precision: {precision_score(y_test_binary, predictions_binary, zero_division=0):.4f}")
    print(f"Recall: {recall_score(y_test_binary, predictions_binary, zero_division=0):.4f}")
    print(f"F1: {f1_score(y_test_binary, predictions_binary, zero_division=0):.4f}")

    with open(MODEL_PATH, "wb") as model_file:
        pickle.dump(model, model_file)

    with open(VECTORIZER_PATH, "wb") as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)

    print("SMS model trained successfully")


if __name__ == "__main__":
    main()
