from vietnamese_nlp.vectorization import TextVectorizer
from vietnamese_nlp.sentiment_analysis import SentimentAnalyzer
from vietnamese_nlp.preprocessing import TextPreprocessor
from vietnamese_nlp.tokenization import VietnameseTokenizer

# Hàm dự đoán cảm xúc
def predict_sentiment(new_text):
    # Tạo các đối tượng cần thiết
    preprocessor = TextPreprocessor()
    tokenizer = VietnameseTokenizer()
    vectorizer = TextVectorizer()

    # Tiền xử lý và tokenization
    new_text_processed = preprocessor.preprocess(new_text)
    new_text_tokenized = tokenizer.tokenize(new_text_processed)

    # Lưu ý rằng bạn phải vectorize văn bản mới sau khi đã "fitted"
    # Đảm bảo rằng bạn đã huấn luyện vectorizer trước đó và lưu lại mô hình
    # Trước khi dự đoán, bạn cần đảm bảo vectorizer đã được fitted trước đó trong quá trình huấn luyện

    try:
        # Kiểm tra xem vectorizer đã được fitted chưa
        vectorizer.vectorizer.transform([""])  # Nếu đã fitted, không lỗi
    except ValueError:
        # Nếu chưa fitted, bạn phải sử dụng dữ liệu huấn luyện để "fit" vectorizer
        X_train = ["Tôi rất thích học lập trình", "Chương trình này thật sự khó khăn", "Tôi yêu công việc của mình"]
        y_train = [1, 0, 1]
        X_processed = [preprocessor.preprocess(text) for text in X_train]
        X_tokenized = [tokenizer.tokenize(text) for text in X_processed]
        vectorizer.fit_transform(X_tokenized)  # Fitting dữ liệu huấn luyện

    # Dự đoán cảm xúc
    new_text_vectorized = vectorizer.transform(new_text_tokenized)  # Dùng transform cho văn bản mới

    analyzer = SentimentAnalyzer()
    analyzer.load_model("sentiment_model.pkl")  # Đảm bảo đã lưu mô hình từ trước
    prediction = analyzer.predict(new_text_vectorized)

    return prediction

