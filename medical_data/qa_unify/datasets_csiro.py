# -*- coding: utf-8 -*-
# All code comments are in English.

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import json
from pathlib import Path

from .schema import UnifiedQAItem  # dataset, id, question, contexts, answer, label, metadata

def _read_json_items(path: str) -> List[Dict[str, Any]]:
    """Read a JSON file that contains a top-level list of dicts."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected a list at top-level: {path}")
        return data

def _norm_one(entry: Dict[str, Any], idx: int, split: str) -> UnifiedQAItem:
    """
    Each entry example (per your sample):
    {
      "question": "...",
      "document": "...",   # passage/article text to embed
      "response": "..."    # reference/answer text
    }
    """
    q = str(entry.get("question", "")).strip()
    doc = str(entry.get("document", "")).strip()
    ans = str(entry.get("response", "")).strip()

    # Use 'document' as a single context string. You can add more later if needed.
    contexts = [doc] if doc else []

    meta: Dict[str, Any] = {
        "source": "CSIRO-MedRedQA+PubMed",
        "raw_keys": list(entry.keys()),
        "split": split,
    }

    return UnifiedQAItem(
        dataset="csiro/medredqa_pubmed",
        id=str(idx),                   # file-local id; replace with stable id if available
        question=q,
        contexts=contexts,
        answer=ans,
        label=None,
        metadata=meta
    )

def load_csiro_medredqa_pubmed(split: str, json_path: str, limit: Optional[int] = None) -> List[UnifiedQAItem]:
    """
    Load one split from the CSIRO dataset you downloaded.
    Args:
      split: "train" | "val" | "test"
      json_path: path to 'medredqa+pubmed_{split}.json'
    """
    entries = _read_json_items(json_path)
    items: List[UnifiedQAItem] = []
    for i, ex in enumerate(entries):
        item = _norm_one(ex, i, split)
        items.append(item)
        if limit is not None and len(items) >= limit:
            break
    return items
