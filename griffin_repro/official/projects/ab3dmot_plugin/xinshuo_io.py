"""Small compatibility subset of Xinshuo_PyToolbox used by AB3DMOT.

The Griffin release vendors AB3DMOT but not the toolbox package it imports.
Only these file helpers are used by the included tracking pipeline.
"""

from __future__ import annotations

import os
from pathlib import Path


def mkdir_if_missing(path: str | os.PathLike[str]) -> None:
    path = os.fspath(path)
    if Path(path).suffix:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    else:
        Path(path).mkdir(parents=True, exist_ok=True)


def is_path_exists(path: str | os.PathLike[str]) -> bool:
    return Path(path).exists()


def fileparts(path: str | os.PathLike[str]) -> tuple[str, str, str]:
    path_obj = Path(path)
    return str(path_obj.parent), path_obj.stem, path_obj.suffix


def load_txt_file(path: str | os.PathLike[str]) -> tuple[list[str], int]:
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        lines = [line.rstrip("\n") for line in handle]
    return lines, len(lines)


def save_txt_file(data: list[str], path: str | os.PathLike[str]) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("w", encoding="utf-8", newline="\n") as handle:
        for line in data:
            handle.write(str(line).rstrip("\n") + "\n")


def load_list_from_folder(folder: str | os.PathLike[str]) -> tuple[list[str], int]:
    files = sorted(str(path) for path in Path(folder).iterdir() if path.is_file())
    return files, len(files)
