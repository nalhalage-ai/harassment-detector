import os
import pickle
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

# Safe loading (NO crash)
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model, vectorizer = pickle.load(f)

def classify(text: str):
    text = text.strip()
    intent_score = rule_score(text)

    # ML prediction (only if model exists)
    if model and vectorizer:
        vec = vectorizer.transform([text])
        probs = model.predict_proba(vec)[0]
        ml_label = LABELS[probs.argmax()]
        ml_conf = float(probs.max())
    else:
        ml_label = "non-harassment"
        ml_conf = 0.4  # safe fallback

    # Hybrid decision logic
    if intent_score >= 6:
        final_label = ml_label if ml_label != "non-harassment" else "serious harassment"
    elif intent_score >= 3:
        final_label = ml_label
    else:
        final_label = "non-harassment"

    confidence = min(1.0, ml_conf + intent_score * 0.08)

    return final_label, round(confidence, 2)
