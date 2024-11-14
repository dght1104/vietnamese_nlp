from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

class SentimentAnalyzer:
    def __init__(self):
        self.model = LogisticRegression()  # Khởi tạo mô hình Logistic Regression
    
    def train(self, X, y):
        """
        Huấn luyện mô hình phân tích cảm xúc.
        Args:
            X: Ma trận TF-IDF (đầu vào).
            y: Nhãn cảm xúc (0 - tiêu cực, 1 - trung lập, 2 - tích cực).
        """
        if X.shape[0] > 1:  # Kiểm tra nếu có nhiều hơn một mẫu
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print("Training Accuracy:", accuracy)
        else:
            # Nếu chỉ có một mẫu, dùng toàn bộ cho huấn luyện
            self.model.fit(X, y)
            print("Model trained with single sample.")

    def predict(self, vector):
        """
        Dự đoán cảm xúc từ vector TF-IDF.
        Args:
            vector: Vector TF-IDF của văn bản cần phân tích.
        Returns:
            Nhãn cảm xúc dự đoán.
        """
        if self.model is not None:
            return self.model.predict(vector)
        else:
            raise ValueError("Model has not been trained yet.")

    def save_model(self, filepath):
        """Lưu mô hình vào file."""
        with open(filepath, 'wb') as file:
            pickle.dump(self.model, file)

    def load_model(self, filepath):
        """Tải mô hình từ file."""
        with open(filepath, 'rb') as file:
            self.model = pickle.load(file)
