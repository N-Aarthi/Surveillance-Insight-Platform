from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

class AnomalyDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.model = LogisticRegression()

    def train(self, texts, labels):
        vectors = self.vectorizer.fit_transform(texts)
        self.model.fit(vectors, labels)
        with open("model.pkl", "wb") as f:
            pickle.dump((self.vectorizer, self.model), f)
        print("✅ Logistic Regression model trained and saved!")

    def load_model(self):
        with open("model.pkl", "rb") as f:
            self.vectorizer, self.model = pickle.load(f)

    def predict(self, texts):
        vectors = self.vectorizer.transform(texts)
        predictions = self.model.predict(vectors)
        return ["Anomaly" if p == 1 else "Normal" for p in predictions]
