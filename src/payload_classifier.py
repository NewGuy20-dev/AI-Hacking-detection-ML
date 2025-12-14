"""Payload Classifier - Detects malicious payloads using security wordlists."""
import os
import re
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

BASE = Path(__file__).parent.parent / "datasets" / "security_payloads"

def load_payloads(folder, label, max_per_file=1000):
    """Load payloads from text files."""
    data = []
    for f in Path(folder).rglob("*"):
        if f.is_file() and f.suffix in ('', '.txt', '.lst', '.list', '.md'):
            try:
                lines = f.read_text(errors='ignore').splitlines()[:max_per_file]
                data.extend([(line.strip(), label) for line in lines if line.strip()])
            except: pass
    return data

def load_training_data():
    """Load all payload categories."""
    data = []
    # Malicious payloads
    data += load_payloads(BASE / "injection", "malicious")
    data += load_payloads(BASE / "fuzzing" / "fuzzdb" / "attack", "malicious", 500)
    mal_count = len(data)
    
    # Benign samples - common words, paths, normal text
    benign = []
    # Normal words from wordlists
    for wl in ["english", "common", "words"]:
        for f in (BASE / "wordlists").rglob(f"*{wl}*"):
            if f.is_file():
                try:
                    benign += [(l.strip(), "benign") for l in f.read_text(errors='ignore').splitlines()[:2000] if l.strip() and len(l) < 50]
                except: pass
    # Add synthetic benign
    benign += [("hello world", "benign"), ("user123", "benign"), ("test@email.com", "benign"),
               ("normal text here", "benign"), ("john doe", "benign"), ("2024-01-01", "benign")] * 1000
    
    data += benign[:mal_count]  # Balance
    return data

def train_classifier():
    """Train payload classifier."""
    print("Loading payloads...")
    data = load_training_data()
    if len(data) < 100:
        print(f"Not enough data ({len(data)} samples)")
        return None
    
    texts, labels = zip(*data)
    print(f"Loaded {len(texts)} samples")
    
    # Vectorize
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,3), analyzer='char_wb')
    X = vectorizer.fit_transform(texts)
    
    # Binary: malicious vs benign
    y = [0 if l == "benign" else 1 for l in labels]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training...")
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    clf.fit(X_train, y_train)
    
    acc = clf.score(X_test, y_test)
    print(f"Accuracy: {acc:.2%}")
    
    # Save
    model_path = Path(__file__).parent.parent / "models" / "payload_classifier.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump({'vectorizer': vectorizer, 'classifier': clf}, f)
    print(f"Saved to {model_path}")
    return clf, vectorizer

class PayloadClassifier:
    def __init__(self):
        model_path = Path(__file__).parent.parent / "models" / "payload_classifier.pkl"
        if model_path.exists():
            with open(model_path, 'rb') as f:
                m = pickle.load(f)
                self.vectorizer = m['vectorizer']
                self.classifier = m['classifier']
        else:
            self.classifier = None
    
    def predict(self, text):
        """Predict if text is malicious payload."""
        if not self.classifier:
            return {"error": "Model not trained"}
        X = self.vectorizer.transform([text])
        proba = self.classifier.predict_proba(X)[0]
        prob = proba[1] if len(proba) > 1 else proba[0]
        return {
            "text": text[:100],
            "malicious": prob > 0.5,
            "confidence": float(prob),
            "risk": "high" if prob > 0.8 else "medium" if prob > 0.5 else "low"
        }

if __name__ == "__main__":
    train_classifier()
    
    # Test
    clf = PayloadClassifier()
    tests = [
        "<script>alert('xss')</script>",
        "' OR 1=1--",
        "%0d%0aSet-Cookie:evil",
        "hello world",
        "SELECT * FROM users",
    ]
    print("\nTest predictions:")
    for t in tests:
        print(clf.predict(t))
