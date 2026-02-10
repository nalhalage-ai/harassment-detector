import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

df = pd.read_csv("harassment_1000.csv")

label_map = {
    "non-harassment": 0,
    "verbal": 1,
    "sexual": 2,
    "cyber": 3,
    "threat": 4,
    "stalking": 5,
    "workplace": 6,
    "physical": 7
}

df["label_id"] = df["label"].map(label_map)

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label_id"], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=8000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

preds = model.predict(X_test_vec)

print(classification_report(y_test, preds))

with open("model.pkl", "wb") as f:
    pickle.dump((model, vectorizer), f)

print("✅ Model trained and saved as model.pkl")
