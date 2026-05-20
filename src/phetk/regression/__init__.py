"""PheWAS regression backends. Importing registers all built-in backends."""
from phetk.regression._base import RegressionBackend, available_methods, get_backend
from phetk.regression import _logit, _cox, _firth_logit, _firth_cox  # noqa: F401

__all__ = ["RegressionBackend", "available_methods", "get_backend"]
