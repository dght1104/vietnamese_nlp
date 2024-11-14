import re
from pyvi import ViUtils

import re
from pyvi import ViUtils

class TextPreprocessor:
    # Danh sách các từ nhiễu
    stop_words = {"là", "của", "và", "những", "được", "có", "trong", "một", "khi", "nếu", "thì"}

    @staticmethod
    def to_lowercase(text):
        return text.lower()

    @staticmethod
    def remove_punctuation(text):
        return re.sub(r'[^\w\s]', '', text)

    @staticmethod
    def remove_extra_whitespace(text):
        return ' '.join(text.split())

    @staticmethod
    def remove_accents(text):
        return ViUtils.remove_accents(text)

    @staticmethod
    def remove_stop_words(text):
        tokens = text.split()
        tokens = [word for word in tokens if word not in TextPreprocessor.stop_words]
        return ' '.join(tokens)
