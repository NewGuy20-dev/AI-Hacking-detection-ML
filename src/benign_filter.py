"""Backward-compatible shim for the benign prefilter."""
try:
    from src.prefilters.benign_pre_filter import BenignPreFilter, get_filter
except Exception:  # pragma: no cover - path layout dependent
    from prefilters.benign_pre_filter import BenignPreFilter, get_filter

__all__ = ["BenignPreFilter", "get_filter"]
