"""Helpers for reading environment configuration."""

import os


def get_env_value(name: str) -> str | None:
    """Return an environment value without accidental surrounding whitespace."""
    value = os.getenv(name)
    return value.strip() if value is not None else None
