# NOTE: All comments are in English.
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any

@dataclass
class UnifiedQAItem:
    dataset: str
    id: str
    question: str
    contexts: List[str]
    answer: str
    label: Optional[str] = None
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["metadata"] = d.get("metadata") or {}
        return d
