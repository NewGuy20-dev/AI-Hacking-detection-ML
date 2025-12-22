"""Unified model registry with versioning for sklearn and PyTorch models."""
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import torch


@dataclass
class ModelVersion:
    """Represents a specific version of a model."""
    version: str
    model_type: str
    filename: str
    created_at: str
    metrics: Dict[str, float]
    metadata: Dict[str, Any]
    
    def is_better_than(self, other: 'ModelVersion', metric: str = 'f1_score') -> bool:
        return self.metrics.get(metric, 0) > other.metrics.get(metric, 0)


class ModelRegistry:
    """Unified registry for managing ML models with versioning."""
    
    MANIFEST_FILE = 'manifest.json'
    
    def __init__(self, models_dir: str = 'models'):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, Any] = {}
        self._manifest = self._load_manifest()
    
    def _load_manifest(self) -> dict:
        path = self.models_dir / self.MANIFEST_FILE
        if path.exists():
            try:
                return json.loads(path.read_text())
            except json.JSONDecodeError:
                return {'models': {}, 'active': {}}
        return {'models': {}, 'active': {}}
    
    def _save_manifest(self):
        path = self.models_dir / self.MANIFEST_FILE
        path.write_text(json.dumps(self._manifest, indent=2))
    
    def _detect_model_type(self, model) -> str:
        return 'pytorch' if hasattr(model, 'state_dict') else 'sklearn'
    
    def _generate_version(self, name: str) -> str:
        versions = self.list_versions(name)
        if not versions:
            return '1.0.0'
        latest = max(versions, key=lambda v: [int(x) for x in v.split('.')])
        parts = [int(x) for x in latest.split('.')]
        parts[-1] += 1
        return '.'.join(map(str, parts))
    
    def save(self, name: str, model, metrics: Dict[str, float] = None,
             version: str = None, metadata: Dict[str, Any] = None) -> ModelVersion:
        """Save a model with versioning."""
        model_type = self._detect_model_type(model)
        version = version or self._generate_version(name)
        ext = '.pt' if model_type == 'pytorch' else '.pkl'
        filename = f"{name}_v{version}{ext}"
        filepath = self.models_dir / filename
        
        if model_type == 'pytorch':
            torch.save({'state_dict': model.state_dict(), 'metadata': metadata or {}}, filepath)
        else:
            joblib.dump(model, filepath)
        
        ver = ModelVersion(
            version=version, model_type=model_type, filename=filename,
            created_at=datetime.now().isoformat(),
            metrics=metrics or {}, metadata=metadata or {}
        )
        
        if name not in self._manifest['models']:
            self._manifest['models'][name] = {}
        self._manifest['models'][name][version] = asdict(ver)
        self._manifest['active'][name] = version
        self._save_manifest()
        return ver
    
    def load(self, name: str, version: str = None, model_class=None) -> Any:
        """Load a model by name and optional version."""
        version = version or self._manifest['active'].get(name)
        if not version:
            return self._load_legacy(name, model_class)
        
        cache_key = f"{name}:{version}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        ver_data = self._manifest['models'].get(name, {}).get(version)
        if not ver_data:
            raise ValueError(f"Model {name} version {version} not found")
        
        filepath = self.models_dir / ver_data['filename']
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        if ver_data['model_type'] == 'pytorch':
            if model_class is None:
                raise ValueError("model_class required for PyTorch models")
            checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
            model = model_class()
            model.load_state_dict(checkpoint['state_dict'])
            model.eval()
            result = model
        else:
            result = joblib.load(filepath)
        
        self._cache[cache_key] = result
        return result
    
    def _load_legacy(self, name: str, model_class=None) -> Any:
        """Load legacy model files without manifest entry."""
        for pattern in [f"{name}.pkl", f"{name}_model.pkl", f"{name}.pt", f"{name}.pth"]:
            filepath = self.models_dir / pattern
            if filepath.exists():
                if filepath.suffix == '.pkl':
                    return joblib.load(filepath)
                checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
                if model_class:
                    model = model_class()
                    state = checkpoint.get('state_dict', checkpoint)
                    model.load_state_dict(state)
                    model.eval()
                    return model
                return checkpoint
        raise FileNotFoundError(f"No model found for: {name}")
    
    def list_models(self) -> List[str]:
        return list(self._manifest['models'].keys())
    
    def list_versions(self, name: str) -> List[str]:
        return list(self._manifest['models'].get(name, {}).keys())
    
    def get_active_version(self, name: str) -> Optional[str]:
        return self._manifest['active'].get(name)
    
    def set_active(self, name: str, version: str):
        """Set the active version (rollback)."""
        if version not in self._manifest['models'].get(name, {}):
            raise ValueError(f"Version {version} not found for {name}")
        self._manifest['active'][name] = version
        self._save_manifest()
        self._cache.pop(f"{name}:{version}", None)
    
    def get_metrics(self, name: str, version: str = None) -> Dict[str, float]:
        version = version or self._manifest['active'].get(name)
        return self._manifest['models'].get(name, {}).get(version, {}).get('metrics', {})
    
    def delete_version(self, name: str, version: str):
        if name not in self._manifest['models']:
            return
        ver_data = self._manifest['models'][name].pop(version, None)
        if ver_data:
            (self.models_dir / ver_data['filename']).unlink(missing_ok=True)
            if self._manifest['active'].get(name) == version:
                versions = self.list_versions(name)
                self._manifest['active'][name] = versions[-1] if versions else None
            self._save_manifest()
            self._cache.pop(f"{name}:{version}", None)
    
    def clear_cache(self):
        self._cache.clear()
