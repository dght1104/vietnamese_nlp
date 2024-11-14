from vietnamese_nlp.preprocessing import TextPreprocessor
from vietnamese_nlp.tokenization import VietnameseTokenizer
from vietnamese_nlp.vectorization import TextVectorizer
from vietnamese_nlp.sentiment_analysis import SentimentAnalyzer

# Ví dụ về dữ liệu mẫu
X = ["Tôi rất thích học lập trình", "Chương trình này thật sự khó khăn", "Tôi yêu công việc của mình"]
y = [1, 0, 1]  # 1: tích cực, 0: tiêu cực

# Tiền xử lý văn bản
preprocessor = TextPreprocessor()
X_processed = [preprocessor.preprocess(text) for text in X]

# Tokenization
tokenizer = VietnameseTokenizer()
X_tokenized = [tokenizer.tokenize(text) for text in X_processed]

# Vector hóa văn bản
vectorizer = TextVectorizer()
X_vectorized = vectorizer.fit_transform(X_tokenized)

# Huấn luyện mô hình
analyzer = SentimentAnalyzer()
analyzer.train(X_vectorized, y)

# Lưu mô hình đã huấn luyện
analyzer.save_model('sentiment_model.pkl')

print("Mô hình đã được lưu thành công.")
