from models.anomaly_detector import AnomalyDetector
from utils.preprocess import clean_text

texts = []
labels = []

with open("data/labeled_data.txt", "r") as file:
    for line in file:
        if line.strip():
            parts = line.strip().rsplit(" ", 1)
            text = clean_text(parts[0])
            label = int(parts[1])
            texts.append(text)
            labels.append(label)

model = AnomalyDetector()
model.train(texts, labels)
