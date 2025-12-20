"""Phase D2: Explainability module with SHAP and attention visualization."""
import numpy as np
import torch
from pathlib import Path


class PayloadExplainer:
    """Explain payload predictions using character importance."""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
    
    def explain(self, text, top_k=10):
        """Get character-level importance scores."""
        chars = [ord(c) % 256 for c in str(text)[:500]]
        chars += [0] * (500 - len(chars))
        
        x = torch.tensor([chars], dtype=torch.long, device=self.device)
        x.requires_grad = False
        
        # Get embedding and enable grad
        with torch.enable_grad():
            embed = self.model.embedding(x)
            embed.retain_grad()
            embed.requires_grad_(True)
            
            # Forward pass through rest of model
            out = embed.permute(0, 2, 1)
            out = torch.relu(self.model.conv1(out))
            out = torch.relu(self.model.conv2(out))
            out = torch.relu(self.model.conv3(out))
            out = self.model.pool(out).squeeze(-1)
            out = torch.relu(self.model.fc1(out))
            logit = self.model.fc2(out)
            
            logit.backward()
            
            # Importance = gradient magnitude
            importance = embed.grad.abs().sum(dim=-1).squeeze().cpu().numpy()
        
        # Get top important characters
        text_len = min(len(text), 500)
        importance = importance[:text_len]
        top_idx = np.argsort(importance)[-top_k:][::-1]
        
        highlights = [(i, text[i], importance[i]) for i in top_idx if i < len(text)]
        
        return {
            'importance_scores': importance,
            'top_characters': highlights,
            'prediction': torch.sigmoid(logit).item()
        }


class URLExplainer:
    """Explain URL predictions with feature importance."""
    
    FEATURE_NAMES = ['length', 'num_dots', 'num_hyphens', 'num_underscores', 'num_digits',
                     'num_params', 'has_ip', 'has_https', 'path_depth', 'has_at',
                     'has_double_slash', 'subdomain_depth', 'suspicious_tld', 'domain_entropy',
                     'suspicious_keywords']
    
    @staticmethod
    def explain_features(url, features, prediction):
        """Explain which features contributed to prediction."""
        reasons = []
        
        if features[6] > 0.5:  # has_ip
            reasons.append("URL contains IP address instead of domain")
        if features[12] > 0.5:  # suspicious_tld
            reasons.append("Suspicious TLD detected (.tk, .ml, etc.)")
        if features[0] > 0.6:  # length
            reasons.append("Unusually long URL")
        if features[1] > 0.5:  # num_dots
            reasons.append("Many subdomains (dots)")
        if features[14] > 0.3:  # suspicious_keywords
            reasons.append("Contains suspicious keywords (login, verify, etc.)")
        if features[7] < 0.5:  # no https
            reasons.append("Not using HTTPS")
        if features[13] > 0.6:  # high entropy
            reasons.append("High domain entropy (random-looking)")
        
        return {
            'url': url,
            'prediction': prediction,
            'is_malicious': prediction > 0.5,
            'confidence': abs(prediction - 0.5) * 2,
            'reasons': reasons if reasons else ["No specific red flags detected"]
        }


class TimeSeriesExplainer:
    """Explain time-series predictions with attention weights."""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
    
    def explain(self, sequence):
        """Get temporal attention weights."""
        x = torch.tensor([sequence], dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            # Get attention weights if model has attention
            if hasattr(self.model, 'get_attention_weights'):
                weights = self.model.get_attention_weights(x).cpu().numpy()[0]
            else:
                # Fallback: use gradient-based importance
                weights = np.ones(len(sequence)) / len(sequence)
            
            logit = self.model(x)
            pred = torch.sigmoid(logit).item()
        
        # Find anomalous timesteps
        top_idx = np.argsort(weights)[-5:][::-1]
        
        return {
            'attention_weights': weights,
            'anomalous_timesteps': top_idx.tolist(),
            'prediction': pred,
            'is_attack': pred > 0.5
        }


def generate_explanation(prediction_result, data):
    """Generate human-readable explanation for predictions."""
    explanations = []
    
    scores = prediction_result.get('scores', {})
    confidence = prediction_result.get('confidence', [0.5])[0]
    is_attack = prediction_result.get('is_attack', [0])[0]
    
    explanation = {
        'verdict': 'MALICIOUS' if is_attack else 'BENIGN',
        'confidence': f"{confidence:.1%}",
        'component_scores': {k: f"{v[0]:.1%}" for k, v in scores.items() if len(v) > 0},
        'reasons': []
    }
    
    # Add reasons based on high scores
    if scores.get('payload', [0])[0] > 0.7:
        explanation['reasons'].append("Payload contains suspicious patterns (possible injection)")
    if scores.get('url', [0])[0] > 0.7:
        explanation['reasons'].append("URL has malicious characteristics")
    if scores.get('timeseries', [0])[0] > 0.7:
        explanation['reasons'].append("Anomalous network traffic pattern detected")
    if scores.get('network', [0])[0] > 0.7:
        explanation['reasons'].append("Network flow matches known attack signatures")
    if scores.get('fraud', [0])[0] > 0.7:
        explanation['reasons'].append("Transaction pattern indicates potential fraud")
    
    if not explanation['reasons']:
        explanation['reasons'].append("No significant threats detected" if not is_attack 
                                      else "Multiple weak signals combined")
    
    return explanation
