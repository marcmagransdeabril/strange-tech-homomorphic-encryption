"""Configuración compartida de pytest para todos los tests."""

import sys

import pytest


@pytest.fixture(autouse=True)
def _clean_quick_start_cache():
    """Evita que quick_start quede cacheado entre directorios distintos."""
    sys.modules.pop("quick_start", None)
    yield
    sys.modules.pop("quick_start", None)
