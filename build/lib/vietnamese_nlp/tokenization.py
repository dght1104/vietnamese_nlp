# vietnamese_nlp/tokenization.py
from pyvi import ViTokenizer

class VietnameseTokenizer:
    @staticmethod
    def tokenize(text):
        return ViTokenizer.tokenize(text)
