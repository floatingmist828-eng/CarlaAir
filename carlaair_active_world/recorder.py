from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class EpisodeRecorder:
    output_path: Path
    meta: Dict[str, Any] = field(default_factory=dict)
    steps: List[Dict[str, Any]] = field(default_factory=list)

    def append(self, step: Dict[str, Any]) -> None:
        self.steps.append(step)

    def save(self) -> Path:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": self.meta,
            "steps": self.steps,
        }
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return self.output_path

