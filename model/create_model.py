import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

texts = [
    "We had a normal disagreement",
    "A colleague disagreed politely",
    "He keeps calling me names",
    "They insulted me repeatedly",
    "Someone touched me without consent",
    "They made sexual comments",
    "I received abusive messages online",
    "They harassed me on social media",
    "They threatened to kill me",
    "I received violent threats",
    "He keeps messaging me every night",
    "Someone follows me regularly",
    "My manager humiliates me in meetings",
    "I face harassment at my workplace",
    "I was pushed aggressively",
    "Someone hit me intentionally"
]

labels = [
    0, 0,  # non-harassment
    1, 1,  # verbal
    2, 2,  # sexual
    3, 3,  # cyber
    4, 4,  # threat
    5, 5,  # stalking
    6, 6,  # workplace
    7, 7   # physical
]

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(texts)

model = LogisticRegression(max_iter=1000)
model.fit(X, labels)

with open("model.pkl", "wb") as f:
    pickle.dump((model, vectorizer), f)

print("model.pkl created successfully")
