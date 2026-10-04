import re
import html
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Ensure NLTK resources are available
try:
    _stopwords = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords', quiet=True)
    _stopwords = set(stopwords.words('english'))

stemmer = PorterStemmer()

def clean_text(text: str) -> str:
    """
    Cleans raw email text:
    1. Lowercase
    2. Unescape & strip HTML tags
    3. Remove URLs
    4. Remove punctuation, special characters, and digits
    5. Tokenize into words
    6. Remove English stopwords
    7. Stem tokens using PorterStemmer
    8. Rejoin into clean string
    """
    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Strip HTML tags & decode HTML entities
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', ' ', text)

    # 3. Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)

    # 4. Remove punctuation, symbols, and digits (keep alphabetic words)
    text = re.sub(r'[^a-z\s]', ' ', text)

    # 5. Tokenize (whitespace split)
    tokens = text.split()

    # 6 & 7. Stopword removal & Porter Stemming
    cleaned_tokens = [
        stemmer.stem(word)
        for word in tokens
        if word not in _stopwords and len(word) > 1
    ]

    # 8. Rejoin
    return " ".join(cleaned_tokens)
