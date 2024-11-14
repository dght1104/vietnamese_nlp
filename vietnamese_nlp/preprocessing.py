import re
from pyvi import ViUtils

class TextPreprocessor:
    # Danh sách các từ nhiễu
    stop_words = {"là", "của", "và", "những", "được", "có", "trong", "một", "khi", "nếu", "thì"}

    @staticmethod
    def to_lowercase(text):
        """Chuyển văn bản thành chữ thường."""
        return text.lower()

    @staticmethod
    def remove_punctuation(text):
        """Loại bỏ dấu câu."""
        return re.sub(r'[^\w\s]', '', text)

    @staticmethod
    def remove_extra_whitespace(text):
        """Loại bỏ khoảng trắng thừa."""
        return ' '.join(text.split())

    @staticmethod
    def remove_stop_words(text):
        """Loại bỏ các từ nhiễu khỏi văn bản."""
        tokens = text.split()
        tokens = [str(word) for word in tokens if word not in TextPreprocessor.stop_words]
        return ' '.join(tokens)

    @classmethod
    def preprocess(cls, text):
        """Thực hiện toàn bộ quy trình tiền xử lý văn bản."""
        text = cls.to_lowercase(text)
        text = cls.remove_punctuation(text)
        text = cls.remove_extra_whitespace(text)
        text = cls.remove_stop_words(text)
        return text

