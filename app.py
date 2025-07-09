# from flask import Flask, request, jsonify
# from models.anomaly_detector import AnomalyDetector
# from utils.preprocess import clean_text
# import os

# app = Flask(__name__)
# model = AnomalyDetector()

# if not os.path.exists("model.pkl"):
#     raise FileNotFoundError("❌ model.pkl not found. Run train.py first.")

# model.load_model()

# @app.route('/')
# def home():
#     return "🚀 Smart Anomaly Detector is running!"

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         data = request.get_json()
#         raw_texts = data.get("texts", [])
#         if not raw_texts:
#             return jsonify({"error": "No input texts provided."}), 400

#         cleaned = [clean_text(t) for t in raw_texts]
#         results = model.predict(cleaned)
#         return jsonify({"results": results}), 200

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     app.run(debug=True)


from flask import Flask, request, jsonify, render_template
from models.anomaly_detector import AnomalyDetector
from utils.preprocess import clean_text
import os

app = Flask(__name__)
model = AnomalyDetector()

if not os.path.exists("model.pkl"):
    raise FileNotFoundError("❌ model.pkl not found. Run train.py first.")

model.load_model()

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    if request.method == 'POST':
        user_input = request.form.get('text_input')
        if user_input:
            cleaned = clean_text(user_input)
            prediction = model.predict([cleaned])[0]
            result = f"Result: {prediction}"
    return render_template('index.html', result=result)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    raw_texts = data.get("texts", [])
    if not raw_texts:
        return jsonify({"error": "No input texts provided."}), 400

    cleaned = [clean_text(t) for t in raw_texts]
    predictions = model.predict(cleaned)
    return jsonify({"results": predictions}), 200

if __name__ == "__main__":
    app.run(debug=True)
