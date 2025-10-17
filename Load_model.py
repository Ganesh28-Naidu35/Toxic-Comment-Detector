import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd

class TextToxicityAnalyzer:
    def __init__(self, threshold=0.7):
        self.threshold = threshold
        self.model = None
        self.training_data = pd.DataFrame(columns=["text", "toxic"])

    def load_model(self):
        """Load a toxicity model from TF Hub (or local)."""
        print("Loading model...")
        # Example: using a universal sentence encoder-based model
        # You can replace this with a specific toxicity classifier hub URL
        self.model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        print("Model loaded!")

    def load_training_data(self, csv_path):
        """Load custom CSV data: text,toxic(1/0)."""
        self.training_data = pd.read_csv(csv_path, header=None, names=["text", "toxic"])
        print(f"Training data loaded: {len(self.training_data)} records")

    def classify_text(self, text):
        if self.model is None:
            raise Exception("Model not loaded yet!")

        # In a real setup, you'd call a toxicity classifier model
        # Here we'll simulate probabilities for demo
        embeddings = self.model([text])  # Get text embeddings (feature vector)

        # Fake classification logic for illustration
        import numpy as np
        toxicity_score = float(abs(hash(text)) % 100) / 100  # mock score (0.0-1.0)

        # Adjust score based on custom training data
        if not self.training_data.empty:
            similar_examples = self.training_data[
                self.training_data["text"].str.contains(text, case=False, na=False)
            ]
            if not similar_examples.empty:
                toxic_ratio = similar_examples["toxic"].mean()
                toxicity_score = (toxicity_score + toxic_ratio) / 2

        return {
            "text": text,
            "toxicity_score": toxicity_score,
            "match": toxicity_score > self.threshold
        }


# ---------------------- Example Usage ----------------------

analyzer = TextToxicityAnalyzer(threshold=0.7)
analyzer.load_model()

# Optional: load CSV training data
# analyzer.load_training_data("training_data.csv")

result = analyzer.classify_text("I will kill you!")
print(result)
