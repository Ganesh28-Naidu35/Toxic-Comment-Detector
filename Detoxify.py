import nltk
from detoxify import Detoxify
from nltk.tokenize import sent_tokenize

# Load model
model = Detoxify('original')

def analyze_paragraph(paragraph, threshold=0.5):
    """
    Returns toxicity analysis per sentence:
    - sentence
    - toxic categories
    - toxicity score
    - whether toxic or safe
    """
    sentences = sent_tokenize(paragraph)
    analysis_results = []

    for sent in sentences:
        scores = model.predict(sent)
        toxic_categories = [cat for cat, score in scores.items() if score >= threshold]
        toxicity_score = scores['toxicity']
        toxic_match = toxicity_score >= threshold

        analysis_results.append({
            'sentence': sent,
            'toxicity_score': round(toxicity_score, 2),
            'toxic_match': toxic_match,
            'categories': toxic_categories
        })

    return analysis_results

# --- Test Paragraph ---
paragraph = """
You are such a stupid idiot and I will kill you if you come here.
Have a nice day!
I hate your guts and you are worthless.
Let's hang out today.
"""

results = analyze_paragraph(paragraph, threshold=0.5)

# --- Display ---
for r in results:
    print("\n--- Sentence Analysis ---")
    print(f"Sentence: {r['sentence']}")
    print(f"Toxic Match: {r['toxic_match']}")
    print(f"Toxicity Score: {r['toxicity_score']}")
    print(f"Categories: {r['categories']}")
