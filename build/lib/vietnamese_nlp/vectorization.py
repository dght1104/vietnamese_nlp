# vietnamese_nlp/vectorization.py
from sklearn.feature_extraction.text import TfidfVectorizer

class TextVectorizer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def fit_transform(self, texts):
        return self.vectorizer.fit_transform(texts)

    def transform(self, text):
        return self.vectorizer.transform([text])
