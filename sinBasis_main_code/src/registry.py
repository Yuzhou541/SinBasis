from typing import Dict, Callable

class Registry:
    def __init__(self):
        self._table: Dict[str, Callable] = {}

    def register(self, name: str):
        def deco(fn):
            if name in self._table:
                raise KeyError(f"Duplicate registry key: {name}")
            self._table[name] = fn
            return fn
        return deco

    def get(self, name: str):
        if name not in self._table:
            raise KeyError(f"Unknown key: {name}. Available: {list(self._table.keys())}")
        return self._table[name]

MODEL_REGISTRY = Registry()
DATA_REGISTRY = Registry()
