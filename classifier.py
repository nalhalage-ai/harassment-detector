import pickle
from rules import rule_score

with open("model/model.pkl", "rb") as f:
    model, vectorizer = pickle.load(f)

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

def classify(text):
    vec = vectorizer.transform([text])
    probs = model.predict_proba(vec)[0]
    ml_label = LABELS[probs.argmax()]
    ml_conf = probs.max()

    rule_intent = rule_score(text)

    # Hybrid decision
    if rule_intent >= 6:
        final = ml_label if ml_label != "non-harassment" else "serious harassment"
    elif rule_intent >= 3:
        final = ml_label
    else:
        final = "non-harassment"

    confidence = min(1.0, ml_conf + rule_intent * 0.05)

    return final, round(confidence, 2)
