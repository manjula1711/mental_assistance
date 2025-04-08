from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import json

app = Flask(__name__)

# Load intents
with open("intents.json", "r") as file:
    intents = json.load(file)

# Flatten patterns and responses
patterns = []
responses = []
tags = []
for intent in intents['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        responses.append(intent['responses'])
        tags.append(intent['tag'])

# Vectorize patterns
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_msg = request.json['message']
    user_vec = vectorizer.transform([user_msg])
    sims = cosine_similarity(user_vec, X)
    idx = sims.argmax()
    best_response = random.choice(responses[idx])
    return jsonify({"response": best_response})

if __name__ == "__main__":
    app.run(debug=True)
