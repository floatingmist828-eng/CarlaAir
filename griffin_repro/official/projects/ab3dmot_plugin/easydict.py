"""Minimal EasyDict replacement for the vendored AB3DMOT plugin."""

from __future__ import annotations


class EasyDict(dict):
    def __init__(self, mapping=None, **kwargs):
        super().__init__()
        mapping = {} if mapping is None else dict(mapping)
        mapping.update(kwargs)
        for key, value in mapping.items():
            self[key] = self._convert(value)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = self._convert(value)

    @classmethod
    def _convert(cls, value):
        if isinstance(value, dict) and not isinstance(value, EasyDict):
            return cls(value)
        if isinstance(value, list):
            return [cls._convert(item) for item in value]
        return value
