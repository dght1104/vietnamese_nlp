from vietnamese_nlp.sentiment_analysis import SentimentAnalyzer
from vietnamese_nlp.preprocessing import TextPreprocessor
from vietnamese_nlp.tokenization import VietnameseTokenizer
import pickle

# Hàm dự đoán cảm xúc
def predict_sentiment(new_text):
    # Tạo các đối tượng cần thiết
    preprocessor = TextPreprocessor()
    tokenizer = VietnameseTokenizer()

    # Tiền xử lý và tách từ
    new_text_processed = preprocessor.preprocess(new_text)
    new_text_tokenized = tokenizer.tokenize(new_text_processed)

    # Load vectorizer và mô hình
    with open("models/vectorizer.pkl", "rb") as file:
        vectorizer = pickle.load(file)
    
    analyzer = SentimentAnalyzer()
    analyzer.load_model("models/sentiment_model.pkl")

    # Vector hóa văn bản mới
    new_text_vectorized = vectorizer.transform([new_text_tokenized])

    # Dự đoán cảm xúc
    prediction = analyzer.predict(new_text_vectorized)
    return prediction
