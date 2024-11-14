# vietnamese_nlp/sentiment_analysis.py
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

class SentimentAnalyzer:
    def __init__(self):
        self.model = LogisticRegression()

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Training Accuracy: {accuracy}")

    def predict(self, vector):
        return self.model.predict(vector)

    def save_model(self, filepath):
        with open(filepath, 'wb') as file:
            pickle.dump(self.model, file)

    def load_model(self, filepath):
        with open(filepath, 'rb') as file:
            self.model = pickle.load(file)
