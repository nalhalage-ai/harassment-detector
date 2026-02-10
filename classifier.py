import pickle
import os
from rules import rule_score

LABELS = {
    0: "non-harassment",
    1: "verbal",
    2: "sexual",
    3: "cyber",
    4: "threat",
    5: "stalking",
    6: "workplace",
    7: "physical"
}

MODEL_PATH = "model/model.pkl"

model = None
vectorizer = None

# Try loading ML model safely
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model, vectorizer = pickle.load(f)

def classify(text: str):
    text = text.strip()
    rule_intent = rule_score(text)

    # --- ML path ---
    if model and vectorizer:
        vec = vectorizer.transform([text])
        probs = model.predict_proba(vec)[0]
        ml_label = LABELS[probs.argmax()]
        ml_conf = probs.max()
    else:
        ml_label = "non-harassment"
        ml_conf = 0.4  # conservative default

    # --- Hybrid decision ---
    if rule_intent >= 6:
        final = ml_label if ml_label != "non-harassment" else "serious harassment"
    elif rule_intent >= 3:
        final = ml_label
    else:
        final = "non-harassment"

    confidence = min(1.0, ml_conf + rule_intent * 0.08)

    return final, round(confidence, 2)
