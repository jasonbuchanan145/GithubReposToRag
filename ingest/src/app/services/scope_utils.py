from __future__ import annotations
from collections import defaultdict
from pathlib import PurePosixPath
from typing import Dict, Iterable, List, Tuple, Any
from llama_index.core.schema import BaseNode


def top_directory(path: str, depth: int = 1) -> str:
    p = PurePosixPath(path or "")
    parts = [x for x in p.parts if x not in (".", "")]
    return "/".join(parts[:depth]) if parts else ""


def group_nodes_by_file(nodes: Iterable[BaseNode]) -> Dict[str, List[BaseNode]]:
    by_file: Dict[str, List[BaseNode]] = defaultdict(list)
    for n in nodes:
        fp = (n.metadata.get("file_path") or n.metadata.get("path") or "").strip()
        by_file[fp].append(n)
    return by_file


def group_files_by_module(file_paths: Iterable[str], depth: int = 1) -> Dict[str, List[str]]:
    by_mod: Dict[str, List[str]] = defaultdict(list)
    for fp in file_paths:
        mod = top_directory(fp, depth=depth)
        by_mod[mod].append(fp)
    return by_mod
