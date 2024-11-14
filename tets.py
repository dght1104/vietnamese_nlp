from my_vietnamese_nlp import TextPreprocessor, VietnameseTokenizer, TextVectorizer, SentimentAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np  # Thêm dòng này để import numpy

# Tạo dữ liệu mẫu
texts = [
    "Tôi rất thích sản phẩm này!",
    "Chất lượng của sản phẩm thật tệ.",
    "Dịch vụ khách hàng không tốt, tôi sẽ không quay lại nữa.",
    "Mua hàng rất tiện lợi, tôi sẽ giới thiệu cho bạn bè.",
    "Sản phẩm rất tốt, tôi hài lòng.",
    "Dịch vụ quá tệ, tôi không muốn mua nữa.",
    "Sản phẩm đạt yêu cầu, sẽ quay lại mua.",
    "Chất lượng sản phẩm không như quảng cáo."
]

# Tiền xử lý văn bản
clean_texts = [TextPreprocessor.to_lowercase(text) for text in texts]
clean_texts = [TextPreprocessor.remove_punctuation(text) for text in clean_texts]

# Tạo nhãn cảm xúc (0: tiêu cực, 1: trung lập, 2: tích cực)
y = np.array([2, 0, 0, 2, 2, 0, 2, 0])

# Sử dụng TF-IDF Vectorizer thay vì TextVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(clean_texts)

# Kiểm tra nhãn
print("Unique labels in y:", set(y))

# Kiểm tra xem nếu có ít nhất hai lớp
if len(set(y)) > 1:
    # Khởi tạo mô hình phân tích cảm xúc
    analyzer = SentimentAnalyzer()
    analyzer.train(X, y)
else:
    print("Dữ liệu không đủ đa dạng cho huấn luyện mô hình. Cần có ít nhất hai lớp.")
