# Đảm bảo rằng các lớp đã được nhập đúng
from vietnamese_nlp import TextPreprocessor, VietnameseTokenizer, TextVectorizer, SentimentAnalyzer
from sklearn.model_selection import train_test_split

# Tạo dữ liệu mẫu
X = ["chào các bạn", "tôi học lập trình", "hôm nay trời đẹp", "học máy là thú vị"]
y = [1, 1, 0, 1]  # 1 là tích cực, 0 là tiêu cực

# Tiền xử lý và tokenization
preprocessor = TextPreprocessor()
tokenizer = VietnameseTokenizer()
vectorizer = TextVectorizer()
analyzer = SentimentAnalyzer()

# Tiền xử lý văn bản
X_processed = [preprocessor.preprocess(text) for text in X]

# Tokenize văn bản
X_tokenized = [tokenizer.tokenize(text) for text in X_processed]

# Chuyển các tokenized text thành dạng văn bản (chuỗi) để vectorizer có thể xử lý
X_tokenized_text = [' '.join(tokens) for tokens in X_tokenized]

# Vectorize các văn bản đã được tokenized
X_vectorized = vectorizer.fit_transform(X_tokenized_text)

# Chia dữ liệu thành bộ huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình
analyzer.train(X_train, y_train)

# Đánh giá mô hình
accuracy = analyzer.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy}")
