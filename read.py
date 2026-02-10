BAD_WORDS = ["slut", "bitch", "whore", "fuck"]
THREATS = ["kill", "hurt", "beat", "rape"]
SEXUAL = ["touch", "kiss", "forced", "naked"]
REPEAT = ["again", "daily", "always", "every day"]

def rule_score(text):
    text = text.lower()
    score = 0
    for w in BAD_WORDS:
        if w in text: score += 2
    for w in THREATS:
        if w in text: score += 3
    for w in SEXUAL:
        if w in text: score += 3
    for w in REPEAT:
        if w in text: score += 1
    return score
