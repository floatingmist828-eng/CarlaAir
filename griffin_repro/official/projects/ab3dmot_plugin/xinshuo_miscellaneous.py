"""Small compatibility subset of Xinshuo_PyToolbox used by AB3DMOT."""

from __future__ import annotations

import time
from typing import Iterable, TypeVar


T = TypeVar("T")


def merge_listoflist(items: Iterable[Iterable[T]], unique: bool = False) -> list[T]:
    merged: list[T] = []
    seen: set[T] = set()
    for group in items:
        for value in group:
            if unique:
                if value in seen:
                    continue
                seen.add(value)
            merged.append(value)
    return merged


def print_log(message, log=None, display: bool = True) -> None:
    text = str(message)
    if display:
        print(text)
    if log is not None:
        log.write(text + "\n")
        if hasattr(log, "flush"):
            log.flush()


def get_timestring() -> str:
    return time.strftime("%Y%m%d_%H%M%S")
