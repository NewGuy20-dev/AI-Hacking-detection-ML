"""AI-Hacking-Detection-ML: Infrastructure components."""

try:
    from .model_registry import ModelRegistry, ModelVersion
except ModuleNotFoundError:  # pragma: no cover - optional runtime deps
    ModelRegistry = None
    ModelVersion = None

try:
    from .input_validator import InputValidator, ValidationError
except ModuleNotFoundError:  # pragma: no cover - optional runtime deps
    InputValidator = None
    ValidationError = None

try:
    from .streaming_dataset import StreamingDataset, LabeledStreamingDataset, create_dataloader
except ModuleNotFoundError:  # pragma: no cover - optional runtime deps
    StreamingDataset = None
    LabeledStreamingDataset = None
    create_dataloader = None

try:
    from .model_monitor import ModelMonitor, DriftDetector
except ModuleNotFoundError:  # pragma: no cover - optional runtime deps
    ModelMonitor = None
    DriftDetector = None

try:
    from .batch_predictor import BatchHybridPredictor, create_batch_predictor
except ModuleNotFoundError:  # pragma: no cover - optional runtime deps
    BatchHybridPredictor = None
    create_batch_predictor = None

__all__ = [
    # Model Management
    "ModelRegistry",
    "ModelVersion",
    # Validation
    "InputValidator",
    "ValidationError",
    # Data Loading
    "StreamingDataset",
    "LabeledStreamingDataset",
    "create_dataloader",
    # Monitoring
    "ModelMonitor",
    "DriftDetector",
    # Prediction
    "BatchHybridPredictor",
    "create_batch_predictor",
]
