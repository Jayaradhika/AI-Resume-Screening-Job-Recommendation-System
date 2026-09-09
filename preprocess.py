"""
Text Preprocessing Module
Cleans and normalizes raw resume/job description text:
lowercasing, removing symbols, removing stopwords, and tokenizing.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Lowercase and strip out non-alphanumeric characters."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_and_clean(text: str) -> list:
    """Tokenize, remove stopwords, and lemmatize."""
    text = clean_text(text)
    tokens = word_tokenize(text)
    tokens = [
        LEMMATIZER.lemmatize(tok)
        for tok in tokens
        if tok not in STOP_WORDS and len(tok) > 1
    ]
    return tokens


def preprocess_document(text: str) -> str:
    """Full pipeline: returns a cleaned, space-joined string ready for TF-IDF."""
    tokens = tokenize_and_clean(text)
    return " ".join(tokens)


if __name__ == "__main__":
    sample = "Experienced Python Developer skilled in SQL, Machine Learning & Pandas!"
    print(preprocess_document(sample))
