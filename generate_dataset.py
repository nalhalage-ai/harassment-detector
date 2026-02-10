import random
import pandas as pd

categories = {
    "non-harassment": [
        "We had a normal disagreement",
        "A colleague disagreed with my opinion",
        "Someone spoke loudly but apologized later",
        "A misunderstanding occurred at work",
        "We resolved the issue through discussion"
    ],

    "verbal": [
        "He keeps calling me names",
        "They insulted me repeatedly",
        "I was mocked in front of others",
        "Someone shouted abusive words at me",
        "They humiliated me verbally"
    ],

    "sexual": [
        "Someone touched me without consent",
        "They made sexual comments about my body",
        "I was forced into physical contact",
        "They sent explicit sexual messages",
        "Unwanted sexual advances were made"
    ],

    "cyber": [
        "I received abusive messages online",
        "Someone posted offensive comments about me",
        "They harassed me through social media",
        "I was threatened via email",
        "Abusive messages were sent repeatedly"
    ],

    "stalking": [
        "He keeps messaging me every night",
        "Someone follows me regularly",
        "They monitor my movements constantly",
        "I am being watched repeatedly",
        "Unwanted contact happens daily"
    ],

    "threat": [
        "They threatened to kill me",
        "Someone said they would hurt me",
        "I received violent threats",
        "They warned me of serious harm",
        "Threatening messages were sent"
    ],

    "workplace": [
        "My manager humiliates me in meetings",
        "I face harassment at my workplace",
        "Colleagues make offensive remarks",
        "Unfair treatment at work continues",
        "I am targeted by seniors at work"
    ],

    "physical": [
        "I was pushed aggressively",
        "Someone hit me intentionally",
        "Physical force was used against me",
        "I was attacked without provocation",
        "They grabbed me violently"
    ]
}

def expand(sentence):
    templates = [
        sentence,
        sentence + " repeatedly",
        sentence + " despite my objections",
        "It makes me feel unsafe because " + sentence.lower(),
        "I feel distressed as " + sentence.lower()
    ]
    return random.choice(templates)

data = []

TARGET = 1000
labels = list(categories.keys())

while len(data) < TARGET:
    label = random.choice(labels)
    base = random.choice(categories[label])
    text = expand(base)
    data.append({"text": text, "label": label})

df = pd.DataFrame(data)
df.to_csv("harassment_1000.csv", index=False)

print("✅ Dataset generated: harassment_1000.csv")
