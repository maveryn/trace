"""Public local review application.

Server imports are lazy so artifact indexing remains usable without installing
the optional web dependencies.
"""

from typing import Any

from .security import DEFAULT_TOKEN_ENV, is_loopback_host, validate_review_bind


def __getattr__(name: str) -> Any:
    if name in {"ReviewAppState", "create_review_app", "serve_review_app"}:
        from . import server

        return getattr(server, name)
    raise AttributeError(name)


__all__ = [
    "DEFAULT_TOKEN_ENV",
    "ReviewAppState",
    "create_review_app",
    "is_loopback_host",
    "serve_review_app",
    "validate_review_bind",
]
