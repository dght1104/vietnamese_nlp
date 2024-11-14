# vietnamese_nlp/train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from vietnamese_nlp.sentiment_analysis import SentimentAnalyzer
from vietnamese_nlp.vectorization import TextVectorizer
from vietnamese_nlp.preprocessing import TextPreprocessor
from vietnamese_nlp.tokenization import VietnameseTokenizer
import pickle

# Đường dẫn tới file CSV chứa dữ liệu
data_path = 'models/shopping_rating.csv'

# Đọc dữ liệu từ file CSV
data = pd.read_csv(data_path)
X = data['comment']  # Cột chứa câu văn
y = data['sentiment']  # Cột chứa nhãn cảm xúc

# Khởi tạo các đối tượng xử lý
preprocessor = TextPreprocessor()
tokenizer = VietnameseTokenizer()
vectorizer = TextVectorizer()
analyzer = SentimentAnalyzer()

# Tiền xử lý và tokenization
X_processed = [preprocessor.preprocess(text) for text in X]
X_tokenized = [tokenizer.tokenize(text) for text in X_processed]

# Vector hóa văn bản
X_vectorized = vectorizer.fit_transform(X_tokenized)

# Huấn luyện mô hình phân tích cảm xúc
analyzer.train(X_vectorized, y)

# Lưu mô hình và vectorizer
with open('model/sentiment_model.pkl', 'wb') as model_file:
    pickle.dump(analyzer.model, model_file)

with open('model/vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer.vectorizer, vectorizer_file)

print("Huấn luyện và lưu mô hình thành công.")
