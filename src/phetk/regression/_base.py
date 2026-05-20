from abc import ABC, abstractmethod
import numpy as np
import polars as pl


class RegressionBackend(ABC):
    """Abstract base for PheWAS regression backends."""
    requires_time_to_event: bool = False

    @abstractmethod
    def fit(
        self,
        regressors: pl.DataFrame,
        y: np.ndarray,
        analysis_var_cols: list[str],
        independent_variable_of_interest: str,
        **kwargs,
    ) -> dict[str, float | str] | None:
        """Fit regression and return result dict, or None on failure."""
        ...


_REGISTRY: dict[str, type[RegressionBackend]] = {}


def register(method_name: str):
    """Decorator to register a backend under a method name."""
    def decorator(cls):
        _REGISTRY[method_name] = cls
        return cls
    return decorator


def get_backend(method: str) -> RegressionBackend:
    """Instantiate the registered backend for a method name."""
    if method not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"Unknown method '{method}'. Available: {available}")
    return _REGISTRY[method]()


def available_methods() -> list[str]:
    """Return sorted list of registered method names."""
    return sorted(_REGISTRY.keys())
