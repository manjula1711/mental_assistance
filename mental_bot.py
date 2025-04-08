from sentence_transformers import SentenceTransformer, util
import json, random

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load intents
with open("intents.json") as f:
    intents = json.load(f)

# Precompute embeddings
pattern_texts = []
pattern_tags = []
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        pattern_texts.append(pattern)
        pattern_tags.append(intent["tag"])

pattern_embeddings = model.encode(pattern_texts)

# Chat loop
print("Chatbot: Hello! Type 'quit' to exit.")
while True:
    msg = input("You: ")
    if msg.lower() == "quit":
        break

    user_embedding = model.encode(msg)
    similarities = util.cos_sim(user_embedding, pattern_embeddings)[0]
    best_idx = similarities.argmax().item()
    confidence = similarities[best_idx].item()

    if confidence > 0.5:
        tag = pattern_tags[best_idx]
        for intent in intents["intents"]:
            if intent["tag"] == tag:
                print("Chatbot:", random.choice(intent["responses"]))
                break
    else:
        print("Chatbot: I'm not sure I understand.")