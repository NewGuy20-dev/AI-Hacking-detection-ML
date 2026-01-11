"""Host-based threat detection agent."""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class DetectionResult:
    score: float
    is_threat: bool
    attack_type: Optional[str]
    indicators: List[str]


class ProcessRuleEngine:
    """Rule-based process execution analyzer."""
    
    SUSPICIOUS_PATTERNS = [
        {'parent': 'explorer.exe', 'child': 'powershell.exe', 'score': 0.7, 'type': 'suspicious_spawn'},
        {'parent': 'winword.exe', 'child': 'cmd.exe', 'score': 0.9, 'type': 'macro_execution'},
        {'parent': 'excel.exe', 'child': 'powershell.exe', 'score': 0.9, 'type': 'macro_execution'},
        {'cmdline_contains': '-enc', 'score': 0.8, 'type': 'encoded_command'},
        {'cmdline_contains': '-nop', 'score': 0.6, 'type': 'bypass_attempt'},
        {'cmdline_contains': 'bypass', 'score': 0.7, 'type': 'bypass_attempt'},
        {'process': 'certutil.exe', 'cmdline_contains': '-urlcache', 'score': 0.9, 'type': 'download_cradle'},
        {'process': 'mshta.exe', 'score': 0.8, 'type': 'script_execution'},
        {'process': 'regsvr32.exe', 'cmdline_contains': '/s /n', 'score': 0.9, 'type': 'applocker_bypass'},
    ]
    
    def analyze(self, events: List[Dict]) -> DetectionResult:
        indicators = []
        max_score = 0.0
        attack_type = None
        
        for event in events:
            for pattern in self.SUSPICIOUS_PATTERNS:
                if self._matches(event, pattern):
                    indicators.append(f"{pattern.get('type', 'unknown')}: {event.get('process', 'unknown')}")
                    if pattern['score'] > max_score:
                        max_score = pattern['score']
                        attack_type = pattern.get('type')
        
        return DetectionResult(
            score=max_score,
            is_threat=max_score > 0.5,
            attack_type=attack_type,
            indicators=indicators
        )
    
    def _matches(self, event: Dict, pattern: Dict) -> bool:
        for key, value in pattern.items():
            if key in ['score', 'type']:
                continue
            if key == 'cmdline_contains':
                if value.lower() not in event.get('cmdline', '').lower():
                    return False
            elif key == 'parent':
                if value.lower() not in event.get('parent', '').lower():
                    return False
            elif key == 'child':
                if value.lower() not in event.get('process', '').lower():
                    return False
            elif key == 'process':
                if value.lower() not in event.get('process', '').lower():
                    return False
        return True


class FileAnomalyDetector:
    """File system activity anomaly detector."""
    
    SUSPICIOUS_PATHS = [
        '/etc/passwd', '/etc/shadow', '/root/.ssh',
        'C:\\Windows\\System32', 'C:\\Windows\\SysWOW64',
    ]
    
    SUSPICIOUS_EXTENSIONS = ['.exe', '.dll', '.ps1', '.bat', '.vbs', '.js', '.hta']
    
    def analyze(self, events: List[Dict]) -> DetectionResult:
        indicators = []
        max_score = 0.0
        
        for event in events:
            path = event.get('path', '').lower()
            op = event.get('operation', '')
            
            # Check suspicious paths
            for sus_path in self.SUSPICIOUS_PATHS:
                if sus_path.lower() in path:
                    indicators.append(f"suspicious_path: {path}")
                    max_score = max(max_score, 0.7)
            
            # Check suspicious writes
            if op in ['write', 'create', 'modify']:
                for ext in self.SUSPICIOUS_EXTENSIONS:
                    if path.endswith(ext):
                        indicators.append(f"suspicious_write: {path}")
                        max_score = max(max_score, 0.6)
        
        return DetectionResult(
            score=max_score,
            is_threat=max_score > 0.5,
            attack_type='file_anomaly' if max_score > 0.5 else None,
            indicators=indicators
        )


class HostBehaviorDetector:
    """
    Host-based threat detection using syscalls, memory, processes, and files.
    
    Weights: syscall (0.3), memory (0.3), process (0.2), file (0.2)
    """
    
    def __init__(self, models_dir: str = 'models', device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.models_dir = Path(models_dir)
        
        # ML models (loaded on demand)
        self.syscall_lstm = None
        self.memory_cnn = None
        
        # Rule engines
        self.process_rules = ProcessRuleEngine()
        self.file_anomaly = FileAnomalyDetector()
        
        # Ensemble weights
        self.weights = {'syscall': 0.3, 'memory': 0.3, 'process': 0.2, 'file': 0.2}
    
    def load_models(self):
        """Load ML models if available."""
        syscall_path = self.models_dir / 'syscall_lstm.pt'
        if syscall_path.exists():
            self.syscall_lstm = torch.jit.load(str(syscall_path), map_location=self.device)
            self.syscall_lstm.eval()
            print("Loaded syscall LSTM")
        
        memory_path = self.models_dir / 'memory_cnn.pt'
        if memory_path.exists():
            self.memory_cnn = torch.jit.load(str(memory_path), map_location=self.device)
            self.memory_cnn.eval()
            print("Loaded memory CNN")
        
        return self
    
    @torch.no_grad()
    def detect_from_syscalls(self, sequence: List[int], max_len: int = 1000) -> DetectionResult:
        """Detect anomalies from syscall sequences."""
        if self.syscall_lstm is None:
            return DetectionResult(0.5, False, None, ["syscall_model_not_loaded"])
        
        # Pad/truncate
        seq = sequence[:max_len]
        seq = seq + [0] * (max_len - len(seq))
        
        x = torch.tensor([seq], dtype=torch.long, device=self.device)
        logits = self.syscall_lstm(x)
        score = torch.sigmoid(logits).item()
        
        attack_type = self._classify_syscall_attack(sequence) if score > 0.5 else None
        
        return DetectionResult(
            score=score,
            is_threat=score > 0.5,
            attack_type=attack_type,
            indicators=[f"syscall_anomaly_score: {score:.2f}"] if score > 0.5 else []
        )
    
    @torch.no_grad()
    def detect_from_memory(self, features: np.ndarray) -> DetectionResult:
        """Detect malware from memory dump features."""
        if self.memory_cnn is None:
            return DetectionResult(0.5, False, None, ["memory_model_not_loaded"])
        
        x = torch.tensor(features, dtype=torch.float32, device=self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        logits = self.memory_cnn(x)
        score = torch.sigmoid(logits).item()
        
        return DetectionResult(
            score=score,
            is_threat=score > 0.5,
            attack_type='memory_malware' if score > 0.5 else None,
            indicators=[f"memory_anomaly_score: {score:.2f}"] if score > 0.5 else []
        )
    
    def detect_from_processes(self, events: List[Dict]) -> DetectionResult:
        """Detect suspicious process execution."""
        return self.process_rules.analyze(events)
    
    def detect_from_files(self, events: List[Dict]) -> DetectionResult:
        """Detect suspicious file activity."""
        return self.file_anomaly.analyze(events)
    
    def ensemble_predict(self, syscalls: List[int] = None, memory: np.ndarray = None,
                        processes: List[Dict] = None, files: List[Dict] = None) -> Dict:
        """Combined prediction from all available sources."""
        scores = {}
        all_indicators = []
        attack_types = []
        
        if syscalls:
            result = self.detect_from_syscalls(syscalls)
            scores['syscall'] = result.score
            all_indicators.extend(result.indicators)
            if result.attack_type:
                attack_types.append(result.attack_type)
        
        if memory is not None:
            result = self.detect_from_memory(memory)
            scores['memory'] = result.score
            all_indicators.extend(result.indicators)
            if result.attack_type:
                attack_types.append(result.attack_type)
        
        if processes:
            result = self.detect_from_processes(processes)
            scores['process'] = result.score
            all_indicators.extend(result.indicators)
            if result.attack_type:
                attack_types.append(result.attack_type)
        
        if files:
            result = self.detect_from_files(files)
            scores['file'] = result.score
            all_indicators.extend(result.indicators)
            if result.attack_type:
                attack_types.append(result.attack_type)
        
        if not scores:
            return {'score': 0.5, 'is_threat': False, 'confidence': 0.0}
        
        # Weighted average
        total_weight = sum(self.weights[k] for k in scores)
        weighted_score = sum(scores[k] * self.weights[k] for k in scores) / total_weight
        
        return {
            'score': weighted_score,
            'is_threat': weighted_score > 0.5,
            'confidence': abs(weighted_score - 0.5) * 2,
            'attack_type': attack_types[0] if attack_types else None,
            'component_scores': scores,
            'indicators': all_indicators
        }
    
    def _classify_syscall_attack(self, sequence: List[int]) -> Optional[str]:
        """Classify attack type from syscall patterns."""
        seq_set = set(sequence)
        
        # Common syscall numbers (Linux x86_64)
        if 101 in seq_set:  # ptrace
            return 'process_injection'
        if 59 in seq_set and 57 in seq_set:  # execve + fork
            return 'privilege_escalation'
        if 41 in seq_set and 42 in seq_set and 59 in seq_set:  # socket + connect + execve
            return 'reverse_shell'
        
        return 'syscall_anomaly'
