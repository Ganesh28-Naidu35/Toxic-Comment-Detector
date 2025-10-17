# Install first (only once):
# pip install detoxify

from detoxify import Detoxify

print("Loading toxicity model...")
model = Detoxify('original')
print("Model loaded!")

def predict_toxicity(text, threshold=0.5):
    results = model.predict(text)

    toxic_match = any(score >= threshold for score in results.values())
    matched_labels = [label for label, score in results.items() if score >= threshold]

    return {
        "text": text,
        "toxic_match": toxic_match,
        "categories": matched_labels,
        "scores": results
    }

# ---- Test ----
examples = [
    "You are sexy!",
    "You are a stupid idiot!",
    "Good morning, have a nice day!",
    "I will kill you!",
    "Let's hang out today",
    "what the fuck are you",
    "i will fuck u",
    "u are a bitch"
]

for ex in examples:
    out = predict_toxicity(ex)
    print("\n--- Result ---")
    print(f"Text: {out['text']}")
    print(f"Toxic Match: {out['toxic_match']}")
    print(f"Categories: {out['categories']}")
    for k, v in out['scores'].items():
        print(f"  {k}: {v:.2f}")
