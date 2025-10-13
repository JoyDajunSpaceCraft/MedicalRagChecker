# python make_rag_inputs_v3.py \
#   --datasets csiro medquad pubmedqa \
#   --split train \
#   --limit 50 \
#   --out-dir tests/_min_input \
#   --csiro-dir /ocean/projects/med230010p/yji3/MedicalRagChecker/medical_data/MedRadQA \
#   --csiro-mode both \
#   --csiro-index-dir /ocean/projects/med230010p/yji3/MedicalRagChecker/medical_data/MedRadQA/csiro_corpus \
#   --csiro-model "pritamdeka/S-PubMedBert-MS-MARCO" \
#   --csiro-rrf-k 60 \
#   --csiro-weights "gold=1.0,csiro_faiss=1.2" \
#   --retriever MedCPT \
#   --corpus PubMed \
#   --k 8 \
#   --use-medquad \
#   --medquad-dir /ocean/projects/med230010p/yji3/MedicalRagChecker/medical_data/MedQuAD/medquad_corpus \
#   --medquad-hybrid \
#   --medquad-alpha 0.65

import os
import json
import argparse
from typing import List, Dict, Any

from qa_unify.loaders import load_pubmedqa, load_medquad, load_liveqa
from qa_unify.datasets_csiro import load_csiro_medredqa_pubmed

# Retrievers
from retriever.csiro_faiss import CsiroFaissRetriever
from retriever.medrag_retriever import MedRAGRetriever  # NOTE: folder is "retrievers", plural.
from MedQuAD.medquad_retriever import MedQuADRetriever  
# ---------------------- Mappers ----------------------

def _map_csiro_gold(ex: Dict[str, Any]) -> Dict[str, Any]:
    """
    CSIRO MedRedQA+PubMed items contain a single long 'document' we normalized
    into 'contexts' (list[str]). Use them directly as retrieved_context.
    """
    qid = str(ex.get("id") or "csiro-na")
    q   = ex.get("question", "")
    ans = ex.get("answer", "") or ""
    ctxs = ex.get("contexts", []) or []
    retrieved_context = [{"doc_id": f"csiro-gold-{i+1}", "text": c} for i, c in enumerate(ctxs)]
    return {
        "query_id": qid,
        "query": q,
        "gt_answer": ans,
        "rag_response": "",
        "retrieved_context": retrieved_context,
        "retrieved_images": []
    }



def _map_common_with_hits(ex: Dict[str, Any], hits: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a unified row from arbitrary hit list (id/text/title/score/source)."""
    qid = str(ex.get("id") or ex.get("qid") or ex.get("dataset_id") or "na")
    q   = ex.get("question", "")
    ans = ex.get("answer") or ex.get("final_answer") or ex.get("long_answer") or ""
    return {
        "query_id": qid,
        "query": q,
        "gt_answer": ans,
        "rag_response": "",
        "retrieved_context": [
            {
                "doc_id": h.get("doc_id") or h.get("id") or "",
                "text": h.get("text", ""),
                "title": h.get("title"),
                "score": h.get("score"),
                "source": h.get("source"),
            } for h in hits or []
        ],
        "retrieved_images": []
    }

def _map_liveqa(ex: Dict[str, Any], retriever, k: int) -> Dict[str, Any]:
    hits = retriever.retrieve(ex.get("question",""), k=k)
    return _map_common_with_hits(ex, hits)

def _map_pubmedqa(ex: Dict[str, Any], retriever, k: int) -> Dict[str, Any]:
    hits = retriever.retrieve(ex.get("question",""), k=k)
    return _map_common_with_hits(ex, hits)

def _map_medquad(ex: Dict[str, Any], retriever, k: int) -> Dict[str, Any]:
    hits = retriever.retrieve(ex.get("question",""), k=k)
    return _map_common_with_hits(ex, hits)

MAPPERS = {
    "liveqa": _map_liveqa,
    "pubmedqa": _map_pubmedqa,
    "medquad": _map_medquad,
}

# ---------------------- RRF fusion ----------------------

def rrf_fuse(runs: Dict[str, List[Dict[str, Any]]], K: int, k_rrf: int = 60, weights: Dict[str, float] | None = None) -> List[Dict[str, Any]]:
    """Reciprocal Rank Fusion over multiple hit lists."""
    weights = weights or {}
    pool, canon = {}, {}

    def key(h):  # stable key
        return h.get("doc_id") or h.get("id") or h.get("text")

    for name, hits in runs.items():
        w = float(weights.get(name, 1.0))
        for rank, h in enumerate(hits):
            k = key(h)
            canon.setdefault(k, h)
            pool[k] = pool.get(k, 0.0) + w * (1.0 / (k_rrf + rank + 1))

    order = sorted(pool.items(), key=lambda kv: -kv[1])[:K]
    return [canon[k] for k, _ in order]

# ---------------------- CLI & main ----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["liveqa"],
                    choices=["liveqa", "pubmedqa", "medquad", "csiro"],
                    help="Which datasets to export.")

    # CSIRO json paths
    ap.add_argument("--csiro-dir", default=None,
                    help="Directory containing medredqa+pubmed_{train,val,test}.json")
    ap.add_argument("--csiro-train", default=None,
                    help="Path to medredqa+pubmed_train.json (overrides --csiro-dir)")
    ap.add_argument("--csiro-val", default=None,
                    help="Path to medredqa+pubmed_val.json (overrides --csiro-dir)")
    ap.add_argument("--csiro-test", default=None,
                    help="Path to medredqa+pubmed_test.json (overrides --csiro-dir)")

    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--out-dir", default="medical_data/normalized_out")

    # MedRAG knobs (for non-CSIRO datasets)
    ap.add_argument("--k", type=int, default=8, help="Top-k snippets per query.")
    ap.add_argument("--retriever", default="MedCPT", choices=["BM25", "Contriever", "SPECTER", "MedCPT"])
    ap.add_argument("--corpus", default="MedCorp", choices=["PubMed", "StatPearls", "Textbooks", "Wikipedia", "MedCorp"])
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--hnsw", action="store_true")

    # CSIRO retrieval mode
    ap.add_argument("--csiro-mode", default="gold", choices=["gold", "csiro_faiss", "both"],
                    help="'gold' uses item.contexts; 'csiro_faiss' queries your FAISS index; 'both' does RRF fusion.")

    # CSIRO FAISS knobs
    ap.add_argument("--csiro-index-dir", default=None,
                    help="Directory with faiss.index + texts.jsonl + meta.jsonl (built by your builder).")
    ap.add_argument("--csiro-model", default="pritamdeka/S-PubMedBert-MS-MARCO",
                    help="Must equal the model used at index-build time.")


    # ---- MedQuAD fusion knobs ----
    ap.add_argument("--use-medquad", action="store_true",
                    help="If set, also retrieve from MedQuAD FAISS/BM25 and fuse.")
    ap.add_argument("--medquad-dir", default="medquad_corpus",
                    help="Directory containing faiss.index + texts.jsonl + meta.jsonl for MedQuAD.")
    ap.add_argument("--medquad-model", default="pritamdeka/S-PubMedBert-MS-MARCO",
                    help="Model name used when MedQuAD index was built.")
    ap.add_argument("--medquad-hybrid", action="store_true",
                    help="Enable hybrid (BM25+dense) inside MedQuADRetriever.")
    ap.add_argument("--medquad-alpha", type=float, default=0.65,
                    help="Dense weight for MedQuAD hybrid fusion (0..1).")
    # Fusion
    ap.add_argument("--csiro-rrf-k", type=int, default=60)
    ap.add_argument("--csiro-weights", default="gold=1.0,csiro_faiss=1.0",
                    help="Comma list like 'gold=1.0,csiro_faiss=1.2' for 'both' mode.")
    ap.add_argument("--debug", action="store_true",
                help="Print and dump retrieval diagnostics before fusion.")
    ap.add_argument("--pubmedqa-subset", default="pqa_artificial",
                choices=["pqa_artificial", "pqa_labeled", "pqa_unlabeled"])

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.limit is not None and args.limit < 0:
        args.limit = None
    # Build a single MedRAG retriever (for non-CSIRO datasets)
    # medrag = MedRAGRetriever(
    #     retriever_name=args.retriever,
    #     corpus_name=args.corpus,
    #     corpus_cache=args.cache,
    #     hnsw=args.hnsw,
    # )
    # All comments in English.
    # needs_medrag = any(ds in ["liveqa", "pubmedqa"] for ds in args.datasets)
    needs_medrag = any(ds in ["liveqa", "pubmedqa", "medquad"] for ds in args.datasets)

    medrag = None
    if needs_medrag:
        medrag = MedRAGRetriever(
            retriever_name=args.retriever,
            corpus_name=args.corpus,
            corpus_cache=args.cache,
            hnsw=args.hnsw,
        )
    medquad = None
    if args.use_medquad:
        print(f"[DBG] MedQuAD dir = {args.medquad_dir}")
        print(f"[DBG] exists(faiss.index) = {os.path.exists(os.path.join(args.medquad_dir, 'faiss.index'))}")
    
        medquad = MedQuADRetriever(
            corpus_dir=args.medquad_dir,
            model_name=args.medquad_model,
            use_hybrid=args.medquad_hybrid,
            alpha_dense=args.medquad_alpha,
        )
    
    # Optional CSIRO FAISS retriever
    csiro_faiss = None
    if args.csiro_mode in ("csiro_faiss", "both"):
        if not args.csiro_index_dir:
            raise ValueError("--csiro-index-dir is required for csiro_faiss/both")
        csiro_faiss = CsiroFaissRetriever(
            corpus_dir=args.csiro_index_dir,
            model_name=args.csiro_model,
            k=args.k
        )

    # Parse fusion weights for CSIRO 'both' mode
    csiro_w = {}
    if args.csiro_mode == "both" and args.csiro_weights:
        for kv in args.csiro_weights.split(","):
            if not kv.strip(): continue
            n, v = kv.split("=")
            csiro_w[n.strip()] = float(v)

    # Helper to resolve CSIRO split file
    def _pick(default_file: str, override: str | None) -> str:
        return override if override else (os.path.join(args.csiro_dir, default_file) if args.csiro_dir else "")
    print("args.csiro_dir", args.csiro_dir)
    # for ds in args.datasets:
    #     print(f"\n=== Building {ds} ===")
    #     # 1) Load items
    #     if ds == "liveqa":
    #         items = load_liveqa(split=args.split, limit=args.limit)
    #     elif ds == "pubmedqa":
    #         items = load_pubmedqa(split=args.split, limit=args.limit)
    #     # elif ds == "medquad":
    #     #     items = load_medquad(split=args.split, limit=args.limit)
    #     elif ds == "medquad":
    #         items = load_medquad(split=args.split, limit=args.limit)
    #         for it in items:
    #             ex = it.to_dict() if hasattr(it, "to_dict") else dict(it)
    #             q = ex.get("question","")
        
    #             hits_medquad = medquad.search(q, top_k=max(args.k, 10)) if medquad else []
    #             hits_medrag  = medrag.retrieve(q, k=max(args.k, 10)) if medrag else []
        
    #             if hits_medquad and hits_medrag:
    #                 fused = rrf_fuse({"medquad": hits_medquad, "medrag": hits_medrag},
    #                                  K=args.k, k_rrf=60, weights={"medquad":1.0, "medrag":1.0})
    #                 rows.append(_map_common_with_hits(ex, fused))
    #             else:
    #                 rows.append(_map_common_with_hits(ex, hits_medquad or hits_medrag))

    #     elif ds == "csiro":
    #         split_lower = args.split.lower()
    #         if split_lower.startswith("train"):
    #             fpath = _pick("medredqa+pubmed_train.json", args.csiro_train)
    #             split_name = "train"
    #         elif split_lower.startswith(("val", "dev")):
    #             fpath = _pick("medredqa+pubmed_val.json", args.csiro_val)
    #             split_name = "val"
    #         else:
    #             fpath = _pick("medredqa+pubmed_test.json", args.csiro_test)
    #             split_name = "test"
    #         if not fpath or not os.path.exists(fpath):
    #             raise FileNotFoundError("CSIRO file not found. Provide --csiro-dir or explicit --csiro-<split> paths.")
    #         items = load_csiro_medredqa_pubmed(split=split_name, json_path=fpath, limit=args.limit)
    #     else:
    #         raise ValueError(f"Unknown dataset {ds}")
    for ds in args.datasets:
        if ds == "liveqa":
            items = load_liveqa(split=args.split, limit=args.limit)
        elif ds == "pubmedqa":
            # items = load_pubmedqa(split=args.split, limit=args.limit)
            items = load_pubmedqa(subset=args.pubmedqa_subset, split="train", limit=args.limit)

        elif ds == "medquad":
            items = load_medquad(split=args.split, limit=args.limit)
        elif ds == "csiro":
            split_lower = args.split.lower()
            def _pick(default_file: str, override: str | None) -> str:
                return override if override else (os.path.join(args.csiro_dir, default_file) if args.csiro_dir else "")
            if split_lower.startswith("train"):
                fpath = _pick("medredqa+pubmed_train.json", args.csiro_train); split_name = "train"
            elif split_lower.startswith(("val","dev")):
                fpath = _pick("medredqa+pubmed_val.json", args.csiro_val);     split_name = "val"
            else:
                fpath = _pick("medredqa+pubmed_test.json", args.csiro_test);   split_name = "test"
            if not fpath or not os.path.exists(fpath):
                raise FileNotFoundError("CSIRO file not found. Provide --csiro-dir or explicit --csiro-<split> paths.")
            items = load_csiro_medredqa_pubmed(split=split_name, json_path=fpath, limit=args.limit)
        else:
            raise ValueError(f"Unknown dataset {ds}")

        rows: List[Dict[str, Any]] = []
        if ds == "csiro":
            for it in items:
                ex = it.to_dict() if hasattr(it, "to_dict") else dict(it)
                if args.csiro_mode == "gold":
                    rows.append(_map_csiro_gold(ex))
                elif args.csiro_mode == "csiro_faiss":
                    q = ex.get("question","")
                    hits = csiro_faiss.search(q, top_k=args.k)
                    rows.append(_map_common_with_hits(ex, hits))
                else:  # both
                    q = ex.get("question","")
                    gold_hits = [{"doc_id": f"csiro-gold-{i+1}", "text": c, "source":"gold", "score":1.0}
                                 for i, c in enumerate(ex.get("contexts", []) or [])]
                    faiss_hits = csiro_faiss.search(q, top_k=max(args.k, 10))
                    fused = rrf_fuse({"gold": gold_hits, "csiro_faiss": faiss_hits},
                                     K=args.k, k_rrf=args.csiro_rrf_k, weights=csiro_w)
                    rows.append(_map_common_with_hits(ex, fused))
    
        elif ds == "medquad":
            for it in items:
                ex = it.to_dict() if hasattr(it, "to_dict") else dict(it)
                q  = ex.get("question","")
                hits_medquad = medquad.search(q, top_k=max(args.k, 10)) if medquad else []
                hits_medrag  = medrag.retrieve(q, k=max(args.k, 10)) if medrag else []
                if hits_medquad and hits_medrag:
                    fused = rrf_fuse({"medquad": hits_medquad, "medrag": hits_medrag},
                                     K=args.k, k_rrf=60, weights={"medquad":1.0,"medrag":1.0})
                    rows.append(_map_common_with_hits(ex, fused))
                else:
                    rows.append(_map_common_with_hits(ex, hits_medquad or hits_medrag))
    
        else:  # liveqa / pubmedqa 走统一 mapper
            mapper = MAPPERS[ds]
            for it in items:
                ex = it.to_dict() if hasattr(it, "to_dict") else dict(it)
                rows.append(mapper(ex, medrag, args.k))
    
        # ------------------ 3) Write out ------------------
        # out_path = os.path.join(args.out_dir, f"rag_generation_outputs_{ds}.jsonl")
        out_path = os.path.join(
            args.out_dir,
            f"rag_generation_outputs_{ds}_{args.split}.jsonl"
        )
        with open(out_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"✅ Wrote {len(rows)} rows → {out_path}")
        # # 2) Build rows
        # if ds == "csiro":
        #     for it in items:
        #         ex = it.to_dict() if hasattr(it, "to_dict") else dict(it)
        #         if args.csiro_mode == "gold":
        #             rows.append(_map_csiro_gold(ex))
        #         elif args.csiro_mode == "csiro_faiss":
        #             q = ex.get("question", "")
        #             hits = csiro_faiss.search(q, top_k=args.k)
        #             rows.append(_map_common_with_hits(ex, hits))
        #         else:  # both -> RRF(gold, csiro_faiss)
        #             q = ex.get("question", "")
        #             gold_hits = [
        #                 {"doc_id": f"csiro-gold-{i+1}", "text": c, "source": "gold", "score": 1.0}
        #                 for i, c in enumerate(ex.get("contexts", []) or [])
        #             ]
        #             faiss_hits = csiro_faiss.search(q, top_k=max(args.k, 10))
        #             fused = rrf_fuse({"gold": gold_hits, "csiro_faiss": faiss_hits}, K=args.k, k_rrf=args.csiro_rrf_k, weights=csiro_w)
        #             rows.append(_map_common_with_hits(ex, fused))
        # else:
        #     # Non-CSIRO datasets: use MedRAG
        #     if ds in ["liveqa", "pubmedqa"]:
        #         assert medrag is not None, "MedRAG must be initialized for non-CSIRO datasets."
        #         mapper = MAPPERS[ds]
        #         for it in items:
        #             ex = it.to_dict() if hasattr(it, "to_dict") else dict(it)
        #             rows.append(mapper(ex, medrag, args.k))

        # # 3) Write out
        # out_path = os.path.join(args.out_dir, f"rag_generation_outputs_{ds}.jsonl")
        # with open(out_path, "w", encoding="utf-8") as f:
        #     for r in rows:
        #         f.write(json.dumps(r, ensure_ascii=False) + "\n")
        # print(f"✅ Wrote {len(rows)} rows → {out_path}")
# python make_rag_inputs_v3.py --datasets csiro medquad pubmedqa   --split train --limit 50 --out-dir tests/_min_input    --csiro-mode both --csiro-index-dir ./MedRadQA/csiro_corpus  --csiro-model "pritamdeka/S-PubMedBert-MS-MARCO" --csiro-rrf-k 60   --csiro-weights "gold=1.0,csiro_faiss=1.2"  --retriever MedCPT  --corpus PubMed --k 8   --use-medquad --medquad-dir medquad_corpus --medquad-hybrid --medquad-alpha 0.65

if __name__ == "__main__":
    main()
# python make_rag_inputs_v3.py   --datasets csiro   --split train   --limit 50   --csiro-dir ./MedRadQA   --out-dir tests/_min_input   --csiro-mode csiro_faiss   --csiro-index-dir ./MedRadQA/csiro_corpus   --csiro-model "pritamdeka/S-PubMedBert-MS-MARCO"   --k 8