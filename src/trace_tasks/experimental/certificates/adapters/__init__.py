"""Certificate adapters. Each module ``<name>.py`` exposes ``create_adapter() -> Adapter``;
``registry.py`` imports them lazily. ``base.py`` holds the abstract contract."""

from .base import Adapter, PublicContext

__all__ = ["Adapter", "PublicContext"]
