from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import polars as pl


@dataclass
class PlotContext:
    """Shared state computed once in Plot.__init__(), passed to every backend."""
    phewas_result: pl.DataFrame
    bonferroni: float
    nominal_significance: float
    phecode_version: str
    phecode_categories: list[str]
    color_dict: dict[str, str]
    color_palette: tuple[str, ...]
    inf_proxy: float | None
    direction_col: str | None


class PlotBackend(ABC):
    """Abstract base for plot backends."""

    @abstractmethod
    def render(self, ctx: PlotContext, **kwargs) -> None:
        """Render the plot using the given context and keyword arguments."""
        ...


_REGISTRY: dict[str, type[PlotBackend]] = {}


def register(plot_type: str):
    """Decorator to register a backend under a plot type name."""
    def decorator(cls):
        _REGISTRY[plot_type] = cls
        return cls
    return decorator


def get_plot_backend(plot_type: str) -> PlotBackend:
    """Instantiate the registered backend for a plot type."""
    if plot_type not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"Unknown plot type '{plot_type}'. Available: {available}")
    return _REGISTRY[plot_type]()


def available_plot_types() -> list[str]:
    """Return sorted list of registered plot type names."""
    return sorted(_REGISTRY.keys())
