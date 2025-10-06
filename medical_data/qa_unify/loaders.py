# NOTE: All comments are in English.
from __future__ import annotations
import json
from typing import Iterable, List, Optional, Dict, Any
from .schema import UnifiedQAItem

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None
from dataclasses import dataclass, asdict
from datasets import load_dataset, get_dataset_split_names


@dataclass
class QAItem:
    id: str
    dataset: str
    split: str
    question: str
    reference_answer: str
    source_urls: list

    def to_dict(self):
        return asdict(self)

def _resolve_split_or_fallback(dataset_name: str, requested: str) -> str:
    names = get_dataset_split_names(dataset_name)
    if requested in names:
        return requested
    for cand in ("train", "validation", "dev", "test"):
        if cand in names:
            return cand
    return names[0]

def _safe_contexts(value) -> List[str]:
    """Normalize various context field shapes into a list of strings."""
    if value is None:
        return []
    if isinstance(value, list):
        out = []
        for v in value:
            if isinstance(v, str):
                out.append(v)
            elif isinstance(v, dict) and "contexts" in v and isinstance(v["contexts"], list):
                out.extend([str(x) for x in v["contexts"]])
            else:
                out.append(str(v))
        return out
    if isinstance(value, dict) and "contexts" in value:
        v = value["contexts"]
        if isinstance(v, list):
            return [str(x) for x in v]
        return [str(v)]
    if isinstance(value, str):
        try:
            obj = json.loads(value)
            return _safe_contexts(obj)
        except Exception:
            return [value]
    return [str(value)]

def normalize_pubmedqa_example(ex: Dict[str, Any], subset: str) -> UnifiedQAItem:
    q = ex.get("question") or ex.get("QUESTION") or ""
    ctx = _safe_contexts(ex.get("context"))
    ans = ex.get("long_answer") or ex.get("answer") or ""
    lab = ex.get("final_decision") or ex.get("label")
    uid = str(ex.get("pubid") or ex.get("id") or ex.get("pub_id") or "")
    meta = {"subset": subset, "raw_keys": list(ex.keys())}
    return UnifiedQAItem(dataset="qiaojin/PubMedQA", id=uid, question=q, contexts=ctx, answer=ans, label=lab, metadata=meta)

def load_pubmedqa(subset: str = "pqa_artificial", split: str = "train", limit: Optional[int] = None) -> List[UnifiedQAItem]:
    if load_dataset is None:
        raise RuntimeError("Please install `datasets`: pip install datasets")
    ds = load_dataset("qiaojin/PubMedQA", subset, split=split)
    items: List[UnifiedQAItem] = []
    for i, ex in enumerate(ds):
        items.append(normalize_pubmedqa_example(ex, subset))
        if limit is not None and len(items) >= limit:
            break
    return items

def normalize_medquad_example(ex: Dict[str, Any]) -> UnifiedQAItem:
    uid = str(ex.get("question_id") or ex.get("id") or "")
    q = ex.get("question") or ""
    ans = ex.get("answer") or ""
    ctx = []
    meta = {
        "umls_cui": ex.get("umls_cui"),
        "umls_semantic_group": ex.get("umls_semantic_group"),
        "umls_semantic_types": ex.get("umls_semantic_types"),
        "question_focus": ex.get("question_focus"),
        "question_type": ex.get("question_type"),
        "category": ex.get("category"),
        "synonyms": ex.get("synonyms"),
        "source": ex.get("source") or ex.get("link"),
    }
    return UnifiedQAItem(dataset="lavita/MedQuAD", id=uid, question=q, contexts=ctx, answer=ans, label=None, metadata=meta)

def load_medquad(split: str = "train", limit: Optional[int] = None) -> List[UnifiedQAItem]:
    if load_dataset is None:
        raise RuntimeError("Please install `datasets`: pip install datasets")
    ds = load_dataset("lavita/MedQuAD", split=split)
    items: List[UnifiedQAItem] = []
    for i, ex in enumerate(ds):
        items.append(normalize_medquad_example(ex))
        if limit is not None and len(items) >= limit:
            break
    return items

def normalize_liveqa_example(ex: Dict[str, Any]) -> UnifiedQAItem:
    uid = str(ex.get("question_id") or ex.get("qid") or ex.get("id") or "")
    q = ex.get("question") or ex.get("question_body") or ex.get("query") or ""
    ans = ex.get("best_answer") or ex.get("final_answer") or ex.get("answer") or ""
    if not ans:
        answers = ex.get("answers") or ex.get("answer_list")
        if isinstance(answers, list) and answers:
            first = answers[0]
            if isinstance(first, dict):
                ans = first.get("text") or first.get("answer") or str(first)
            else:
                ans = str(first)
    ctx = []
    meta = {k: v for k, v in ex.items() if k not in {"question","answer","best_answer","final_answer","answers","question_body","query"}}
    return UnifiedQAItem(dataset="hyesunyun/liveqa_medical_trec2017", id=uid, question=q, contexts=ctx, answer=ans, label=None, metadata=meta)
# NOTE: All comments are in English.
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import json

from datasets import load_dataset
from .schema import UnifiedQAItem  # expects: dataset, id, question, contexts, answer, label, metadata

def _choose_question_text(ex: Dict[str, Any]) -> str:
    """Prefer NIST_PARAPHRASE; otherwise join subject + message."""
    q = (ex.get("NIST_PARAPHRASE") or "").strip()
    if q:
        return q
    subj = (ex.get("ORIGINAL_QUESTION_SUBJECT") or "").strip()
    msg  = (ex.get("ORIGINAL_QUESTION_MESSAGE") or "").strip()
    text = " ".join([t for t in (subj, msg) if t])
    return text or (ex.get("QUESTION_ID") or "")

def _extract_ref_answers_and_urls(ex: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """Unpack REFERENCE_ANSWERS list of dicts into plain texts and URLs."""
    answers, urls = [], []
    ra = ex.get("REFERENCE_ANSWERS") or []
    if isinstance(ra, list):
        for item in ra:
            if isinstance(item, dict):
                ans = item.get("ANSWER")
                url = item.get("AnswerURL")
                if ans:
                    answers.append(str(ans))
                if url:
                    urls.append(str(url))
            else:
                # Fallback for unexpected shapes
                answers.append(str(item))
    return answers, urls

def _build_contexts(ex: Dict[str, Any]) -> List[str]:
    """Optional contexts: NLM summary and raw subject/message for retrieval prompts."""
    ctx = []
    if ex.get("NLM_SUMMARY"):
        ctx.append(str(ex["NLM_SUMMARY"]))
    # Keep the original subject/message lightly as auxiliary context
    subj = ex.get("ORIGINAL_QUESTION_SUBJECT")
    msg  = ex.get("ORIGINAL_QUESTION_MESSAGE")
    if subj:
        ctx.append(f"[SUBJECT] {subj}")
    if msg:
        ctx.append(f"[MESSAGE] {msg}")
    return ctx

def _maybe_fetch_docs(urls: List[str], timeout: float = 8.0) -> Dict[str, str]:
    """Optionally fetch page text for GT docs. Safe to skip if no Internet."""
    try:
        import requests
        from bs4 import BeautifulSoup
    except Exception:
        return {}
    texts = {}
    for u in urls[:5]:  # cap to avoid huge downloads
        try:
            r = requests.get(u, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            # Simple readability-lite: drop scripts/styles, get visible text
            for tag in soup(["script", "style", "noscript"]):
                tag.extract()
            text = " ".join(soup.get_text(separator=" ").split())
            if text:
                texts[u] = text[:40000]  # soft limit
        except Exception:
            continue
    return texts

def normalize_liveqa_example(ex: Dict[str, Any],
                             join_all_answers: bool = False,
                             include_gold_docs: bool = True,
                             fetch_docs: bool = False) -> UnifiedQAItem:
    """Map one HF row to UnifiedQAItem."""
    qid = str(ex.get("QUESTION_ID") or ex.get("id") or "")
    q   = _choose_question_text(ex)
    answers, urls = _extract_ref_answers_and_urls(ex)
    # Choose answer: either first, or concat all
    if join_all_answers and answers:
        ans_text = "\n\n".join(answers)
    else:
        ans_text = answers[0] if answers else ""
    ctx = _build_contexts(ex)
    meta: Dict[str, Any] = {
        "urls": urls,
        "file": ex.get("ORIGINAL_QUESTION_FILE"),
        "annotations": {
            "focus": ex.get("ANNOTATIONS_FOCUS"),
            "type": ex.get("ANNOTATIONS_TYPE"),
            "keyword": ex.get("ANNOTATIONS_KEYWORD"),
        }
    }
    if include_gold_docs and fetch_docs and urls:
        meta["gold_doc_text"] = _maybe_fetch_docs(urls)
    return UnifiedQAItem(
        dataset="hyesunyun/liveqa_medical_trec2017",
        id=qid,
        question=q,
        contexts=ctx,
        answer=ans_text,
        label=None,
        metadata=meta
    )

def load_liveqa(split: str = "test",
                limit: Optional[int] = None,
                join_all_answers: bool = False,
                include_gold_docs: bool = True,
                fetch_docs: bool = False) -> List[UnifiedQAItem]:
    """
    Load TREC LiveQA Medical 2017 (HF id: hyesunyun/liveqa_medical_trec2017).
    Note: Only 'test' split is published on HF for this dataset.
    """
    ds_name = "hyesunyun/liveqa_medical_trec2017"
    ds = load_dataset(ds_name, split="test")  # the dataset exposes test-only
    items: List[UnifiedQAItem] = []
    for i, ex in enumerate(ds):
        item = normalize_liveqa_example(ex,
                                        join_all_answers=join_all_answers,
                                        include_gold_docs=include_gold_docs,
                                        fetch_docs=fetch_docs)
        items.append(item)
        if limit is not None and len(items) >= limit:
            break
    return items

def to_jsonl(items: List[UnifiedQAItem], path: str) -> None:
    import json
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it.to_dict(), ensure_ascii=False) + "\n")
