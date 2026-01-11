"""Scenario dataclasses and registry for V1.4 stress test suite."""
from dataclasses import dataclass
from typing import Any, Optional, List, Dict
from datetime import datetime
from pathlib import Path
import yaml


@dataclass
class Scenario:
    """Test scenario definition."""
    id: str
    model: str
    category: str
    subcategory: str
    input_data: Any
    expected_label: int
    difficulty: str
    description: str
    source: str


@dataclass
class ScenarioResult:
    """Result from running a scenario."""
    scenario: Scenario
    prediction: int
    confidence: float
    passed: bool
    latency_ms: float
    timestamp: str
    error: Optional[str] = None


class ScenarioRegistry:
    """Registry for loading and managing scenarios."""
    
    def __init__(self, scenarios_dir: Path):
        self.scenarios_dir = Path(scenarios_dir)
        
    def load_static(self, model_name: str) -> List[Scenario]:
        """Load static scenarios from YAML for a model."""
        yaml_path = self.scenarios_dir / f"{model_name}.yaml"
        
        if not yaml_path.exists():
            return []
        
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        
        scenarios = []
        for item in data.get('scenarios', []):
            scenarios.append(Scenario(
                id=item['id'],
                model=model_name,
                category=item['category'],
                subcategory=item['subcategory'],
                input_data=item['input'],
                expected_label=item['expected'],
                difficulty=item['difficulty'],
                description=item['description'],
                source='static'
            ))
        
        return scenarios
