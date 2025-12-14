"""Prediction script for AI Hacking Detection system."""
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from data_loader import preprocess_network_data, NSL_KDD_COLS
from feature_engineering import extract_url_features_batch, ContentFeatureExtractor
from ensemble import EnsembleDetector


class HackingDetector:
    """Unified interface for hacking detection."""
    
    def __init__(self, models_dir: str):
        self.models_dir = Path(models_dir)
        self.ensemble = None
        self.network_scaler = None
        self.content_vectorizer = None
        self._load_models()
    
    def _load_models(self):
        """Load all models and preprocessors."""
        # Load ensemble
        ensemble_path = self.models_dir / 'ensemble.pkl'
        if ensemble_path.exists():
            self.ensemble = EnsembleDetector.load(str(ensemble_path))
        else:
            self.ensemble = EnsembleDetector(str(self.models_dir))
            self.ensemble.load_models()
        
        # Load preprocessors
        scaler_path = self.models_dir / 'network_scaler.pkl'
        if scaler_path.exists():
            self.network_scaler = joblib.load(scaler_path)
        
        vectorizer_path = self.models_dir / 'content_vectorizer.pkl'
        if vectorizer_path.exists():
            self.content_vectorizer = joblib.load(vectorizer_path)
    
    def predict_network(self, data: pd.DataFrame) -> dict:
        """Predict on network traffic data."""
        if self.ensemble.network_model is None:
            return {'error': 'Network model not loaded'}
        
        # Preprocess
        X, _, _, _ = preprocess_network_data(data)
        if self.network_scaler:
            X = self.network_scaler.transform(X)
        
        probs = self.ensemble.predict_proba_network(X)
        preds = (probs >= 0.5).astype(int)
        
        return {
            'predictions': preds.tolist(),
            'probabilities': probs.tolist(),
            'detector': 'network'
        }
    
    def predict_url(self, urls: list) -> dict:
        """Predict on URLs."""
        if self.ensemble.url_model is None:
            return {'error': 'URL model not loaded'}
        
        X = extract_url_features_batch(urls)
        probs = self.ensemble.predict_proba_url(X)
        preds = (probs >= 0.5).astype(int)
        
        return {
            'predictions': preds.tolist(),
            'probabilities': probs.tolist(),
            'urls': urls,
            'detector': 'url'
        }
    
    def predict_content(self, texts: list) -> dict:
        """Predict on text content."""
        if self.ensemble.content_model is None:
            return {'error': 'Content model not loaded'}
        
        if self.content_vectorizer:
            X = self.content_vectorizer.transform(texts)
        else:
            return {'error': 'Content vectorizer not loaded'}
        
        probs = self.ensemble.predict_proba_content(X)
        preds = (probs >= 0.5).astype(int)
        
        return {
            'predictions': preds.tolist(),
            'probabilities': probs.tolist(),
            'detector': 'content'
        }
    
    def predict_combined(self, network_data: pd.DataFrame = None, 
                        urls: list = None, texts: list = None,
                        threshold: float = 0.5) -> dict:
        """Combined prediction using ensemble."""
        network_probs = url_probs = content_probs = None
        
        if network_data is not None and self.ensemble.network_model:
            X, _, _, _ = preprocess_network_data(network_data)
            if self.network_scaler:
                X = self.network_scaler.transform(X)
            network_probs = self.ensemble.predict_proba_network(X)
        
        if urls and self.ensemble.url_model:
            X = extract_url_features_batch(urls)
            url_probs = self.ensemble.predict_proba_url(X)
        
        if texts and self.ensemble.content_model and self.content_vectorizer:
            X = self.content_vectorizer.transform(texts)
            content_probs = self.ensemble.predict_proba_content(X)
        
        result = self.ensemble.predict(network_probs, url_probs, content_probs, threshold)
        
        return {
            'is_attack': result['is_attack'].tolist(),
            'confidence': result['confidence'].tolist(),
            'threshold': threshold
        }


def main():
    parser = argparse.ArgumentParser(description='AI Hacking Detection Prediction')
    parser.add_argument('--input', '-i', required=True, help='Input file (CSV for network, TXT for URLs)')
    parser.add_argument('--type', '-t', choices=['network', 'url', 'content', 'auto'], 
                       default='auto', help='Input type')
    parser.add_argument('--models', '-m', default='models', help='Models directory')
    parser.add_argument('--output', '-o', help='Output file (JSON)')
    parser.add_argument('--threshold', default=0.5, type=float, help='Detection threshold')
    args = parser.parse_args()
    
    base = Path('/workspaces/AI-Hacking-detection-ML')
    models_dir = base / args.models
    
    detector = HackingDetector(str(models_dir))
    input_path = Path(args.input)
    
    # Auto-detect input type
    input_type = args.type
    if input_type == 'auto':
        if input_path.suffix == '.csv':
            input_type = 'network'
        elif input_path.suffix == '.txt':
            with open(input_path) as f:
                first_line = f.readline().strip()
            input_type = 'url' if first_line.startswith('http') else 'content'
    
    # Run prediction
    if input_type == 'network':
        df = pd.read_csv(input_path, names=NSL_KDD_COLS)
        result = detector.predict_network(df)
    elif input_type == 'url':
        with open(input_path) as f:
            urls = [line.strip() for line in f if line.strip()]
        result = detector.predict_url(urls)
    else:
        with open(input_path) as f:
            texts = [line.strip() for line in f if line.strip()]
        result = detector.predict_content(texts)
    
    # Output
    output = json.dumps(result, indent=2)
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"Results saved to {args.output}")
    else:
        print(output)
    
    # Summary
    if 'predictions' in result:
        preds = result['predictions']
        print(f"\nSummary: {sum(preds)}/{len(preds)} detected as malicious")


if __name__ == '__main__':
    main()
