#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid KG Scorer - Combines DRKG and BioPortal for MedRAGChecker

This module provides a unified interface that can use:
1. DRKG only (original approach)
2. BioPortal only (new ontology-based approach)
3. Both combined (ensemble scoring)

The hybrid approach is recommended for:
- Better coverage (BioPortal has more entities)
- Better precision (DRKG TransE provides probabilistic scores)
- Smoother migration from DRKG to BioPortal

Usage:
    # BioPortal only
    python hybrid_kg_scorer.py \
        --claims ./dataset/results_text.json \
        --mode bioportal \
        --bioportal_key YOUR_API_KEY \
        --outfile ./scores.jsonl

    # DRKG only (original)
    python hybrid_kg_scorer.py \
        --claims ./dataset/results_text.json \
        --mode drkg \
        --embed_dir ./KG/DRKG/embed \
        --name_map ./aux/name_map.csv \
        --outfile ./scores.jsonl

    # Hybrid (both)
    python hybrid_kg_scorer.py \
        --claims ./dataset/results_text.json \
        --mode hybrid \
        --embed_dir ./KG/DRKG/embed \
        --name_map ./aux/name_map.csv \
        --bioportal_key YOUR_API_KEY \
        --outfile ./scores.jsonl
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import numpy as np
from tqdm import tqdm

# Import DRKG components (from your existing code)
try:
    from end2end_drkg_softscore import (
        load_id_maps, load_embeddings, build_alias_index,
        build_relation_index, topk_entities, topk_entities_any,
        topm_relations, transe_score, norm_txt,
        REL_TYPE_FILTER, REL_CANON
    )
    HAS_DRKG = True
except ImportError:
    HAS_DRKG = False
    print("[INFO] DRKG module not found. DRKG mode will be unavailable.")

# Import BioPortal components
try:
    from bioportal_kg_scorer import (
        BioPortalKGScorer, BioPortalClient, BioPortalEntityLinker,
        OntologyScorer, ClaimScore, load_claims
    )
    HAS_BIOPORTAL = True
except ImportError:
    HAS_BIOPORTAL = False
    print("[INFO] BioPortal module not found. BioPortal mode will be unavailable.")


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class HybridScore:
    """Combined score from multiple KG sources"""
    claim_text: str
    subject: str
    relation: str
    object: str
    
    # DRKG scores
    drkg_head_id: Optional[str] = None
    drkg_tail_id: Optional[str] = None
    drkg_rel: Optional[str] = None
    drkg_text_sim: float = 0.0
    drkg_kge_score: Optional[float] = None
    drkg_transe_dist: Optional[float] = None
    
    # BioPortal scores
    bioportal_head_id: Optional[str] = None
    bioportal_head_name: Optional[str] = None
    bioportal_tail_id: Optional[str] = None
    bioportal_tail_name: Optional[str] = None
    bioportal_ontology_score: float = 0.0
    bioportal_text_sim: float = 0.0
    bioportal_evidence: List[str] = field(default_factory=list)
    
    # Fused scores
    drkg_final: float = 0.0
    bioportal_final: float = 0.0
    hybrid_final: float = 0.0
    
    # Status
    status: str = "pending"
    source: str = "none"  # drkg, bioportal, hybrid


# ============================================================================
# DRKG Scorer Wrapper
# ============================================================================

class DRKGScorerWrapper:
    """Wrapper around existing DRKG scoring logic"""
    
    def __init__(self, embed_dir: Path, name_map_path: Path,
                 k: int = 5, m: int = 5, alpha: float = 0.5):
        if not HAS_DRKG:
            raise ImportError("DRKG module not available")
        
        self.k = k
        self.m = m
        self.alpha = alpha
        
        # Load DRKG data
        self.ent2idx, self.rel2idx = load_id_maps(embed_dir)
        self.E, self.R = load_embeddings(embed_dir)
        
        # Build indices
        (self.indices, self.owner_lists, 
         self.alias_texts, self.ent_enc) = build_alias_index(name_map_path)
        self.rel_enc, self.rel_idx, self.rel_keys = build_relation_index()
    
    def score(self, subject: str, relation: str, obj: str,
              type_pairs: List[Tuple[str, str]] = None) -> Dict:
        """Score a claim using DRKG"""
        
        if type_pairs is None:
            type_pairs = [
                ("Compound", "Disease"), ("Compound", "Side Effect"),
                ("Compound", "Gene"), ("Gene", "Disease"),
                ("Compound", "Compound")
            ]
        
        best = None
        best_parts = None
        
        for (ht, tt) in type_pairs:
            # Entity candidates
            H = topk_entities(self.ent_enc, self.indices, self.owner_lists,
                             subject, ht, k=self.k)
            T = topk_entities(self.ent_enc, self.indices, self.owner_lists,
                             obj, tt, k=self.k)
            
            # Fallback
            if not H:
                H = topk_entities_any(self.ent_enc, self.indices,
                                      self.owner_lists, subject, k=self.k)
            if not T:
                T = topk_entities_any(self.ent_enc, self.indices,
                                      self.owner_lists, obj, k=self.k)
            
            if not H or not T:
                continue
            
            # Relation candidates
            rset = REL_TYPE_FILTER.get((ht, tt), [])
            RC = topm_relations(self.rel_enc, self.rel_idx, self.rel_keys,
                               relation, m=self.m,
                               type_filter=rset if rset else None)
            
            if not RC:
                RC = [(rk, 0.0) for rk in (rset if rset else self.rel_keys)]
            
            # Score all combinations
            for (h_id, sim_h) in H:
                for (r_key, sim_r) in RC:
                    for (t_id, sim_t) in T:
                        text_sim = float(min(sim_h, sim_t)) * (0.5 + 0.5 * sim_r)
                        kge = transe_score(self.E, self.R, self.ent2idx,
                                          self.rel2idx, h_id, r_key, t_id)
                        
                        if kge is None:
                            final_score = text_sim
                            candidate = (h_id, r_key, t_id, text_sim, None, None)
                        else:
                            score, p_kge, dist = kge
                            p_final = (1 - self.alpha) * p_kge + self.alpha * text_sim
                            final_score = p_final
                            candidate = (h_id, r_key, t_id, text_sim, p_kge, dist)
                        
                        if best is None or final_score > best:
                            best = final_score
                            best_parts = candidate
        
        if best_parts:
            h_id, r_key, t_id, text_sim, p_kge, dist = best_parts
            return {
                "status": "ok",
                "head_id": h_id,
                "rel": r_key,
                "tail_id": t_id,
                "text_sim": text_sim,
                "p_kge": p_kge,
                "transe_dist": dist,
                "final": best
            }
        
        return {"status": "no_candidate", "final": 0.0}


# ============================================================================
# Hybrid Scorer
# ============================================================================

class HybridKGScorer:
    """
    Hybrid scorer combining DRKG and BioPortal
    
    Fusion strategies:
    1. max: Take the max score from either source
    2. weighted: Weighted average based on confidence
    3. ensemble: Train weights on validation data
    """
    
    def __init__(self, 
                 mode: str = "hybrid",
                 # DRKG params
                 embed_dir: Optional[Path] = None,
                 name_map_path: Optional[Path] = None,
                 drkg_k: int = 5,
                 drkg_m: int = 5,
                 drkg_alpha: float = 0.5,
                 # BioPortal params
                 bioportal_key: Optional[str] = None,
                 bioportal_ontologies: List[str] = None,
                 bioportal_cache_dir: Optional[Path] = None,
                 bioportal_k: int = 5,
                 bioportal_alpha: float = 0.5,
                 # Fusion params
                 fusion_strategy: str = "max",
                 drkg_weight: float = 0.5):
        """
        Args:
            mode: 'drkg', 'bioportal', or 'hybrid'
            fusion_strategy: 'max', 'weighted', or 'ensemble'
            drkg_weight: Weight for DRKG in weighted fusion (0-1)
        """
        self.mode = mode
        self.fusion_strategy = fusion_strategy
        self.drkg_weight = drkg_weight
        
        # Initialize DRKG scorer
        self.drkg_scorer = None
        if mode in ("drkg", "hybrid") and embed_dir and name_map_path:
            if HAS_DRKG:
                try:
                    self.drkg_scorer = DRKGScorerWrapper(
                        embed_dir, name_map_path,
                        k=drkg_k, m=drkg_m, alpha=drkg_alpha
                    )
                except Exception as e:
                    print(f"[WARN] Failed to initialize DRKG: {e}")
            else:
                print("[WARN] DRKG requested but module not available")
        
        # Initialize BioPortal scorer
        self.bioportal_scorer = None
        if mode in ("bioportal", "hybrid") and bioportal_key:
            if HAS_BIOPORTAL:
                try:
                    self.bioportal_scorer = BioPortalKGScorer(
                        api_key=bioportal_key,
                        ontologies=bioportal_ontologies or ["SNOMEDCT", "MESH", "RXNORM", "DOID"],
                        cache_dir=bioportal_cache_dir,
                        alpha=bioportal_alpha
                    )
                except Exception as e:
                    print(f"[WARN] Failed to initialize BioPortal: {e}")
            else:
                print("[WARN] BioPortal requested but module not available")
        
        # Validate configuration
        if mode == "drkg" and not self.drkg_scorer:
            raise ValueError("DRKG mode requested but scorer not initialized")
        if mode == "bioportal" and not self.bioportal_scorer:
            raise ValueError("BioPortal mode requested but scorer not initialized")
        if mode == "hybrid" and not (self.drkg_scorer or self.bioportal_scorer):
            raise ValueError("Hybrid mode requires at least one scorer")
    
    def score_claim(self, subject: str, relation: str, obj: str) -> HybridScore:
        """Score a single claim"""
        
        result = HybridScore(
            claim_text=f"{subject} | {relation} | {obj}",
            subject=subject,
            relation=relation,
            object=obj
        )
        
        # Get DRKG score
        if self.drkg_scorer:
            drkg_result = self.drkg_scorer.score(subject, relation, obj)
            result.drkg_head_id = drkg_result.get("head_id")
            result.drkg_tail_id = drkg_result.get("tail_id")
            result.drkg_rel = drkg_result.get("rel")
            result.drkg_text_sim = drkg_result.get("text_sim", 0.0)
            result.drkg_kge_score = drkg_result.get("p_kge")
            result.drkg_transe_dist = drkg_result.get("transe_dist")
            result.drkg_final = drkg_result.get("final", 0.0)
        
        # Get BioPortal score
        if self.bioportal_scorer:
            bp_result = self.bioportal_scorer.score_claim(subject, relation, obj)
            if bp_result.subject_entity:
                result.bioportal_head_id = bp_result.subject_entity.full_id
                result.bioportal_head_name = bp_result.subject_entity.name
            if bp_result.object_entity:
                result.bioportal_tail_id = bp_result.object_entity.full_id
                result.bioportal_tail_name = bp_result.object_entity.name
            result.bioportal_ontology_score = bp_result.ontology_support_score
            result.bioportal_text_sim = bp_result.semantic_similarity
            result.bioportal_evidence = bp_result.evidence
            result.bioportal_final = bp_result.final_score
        
        # Fuse scores
        result.hybrid_final = self._fuse_scores(
            result.drkg_final, result.bioportal_final
        )
        
        # Determine source and status
        if result.drkg_final > 0 and result.bioportal_final > 0:
            result.source = "hybrid"
            result.status = "ok"
        elif result.drkg_final > 0:
            result.source = "drkg"
            result.status = "ok"
        elif result.bioportal_final > 0:
            result.source = "bioportal"
            result.status = "ok"
        else:
            result.source = "none"
            result.status = "no_support"
        
        return result
    
    def _fuse_scores(self, drkg_score: float, bioportal_score: float) -> float:
        """Fuse scores from different sources"""
        
        if self.fusion_strategy == "max":
            return max(drkg_score, bioportal_score)
        
        elif self.fusion_strategy == "weighted":
            if drkg_score > 0 and bioportal_score > 0:
                return (self.drkg_weight * drkg_score + 
                       (1 - self.drkg_weight) * bioportal_score)
            elif drkg_score > 0:
                return drkg_score
            else:
                return bioportal_score
        
        elif self.fusion_strategy == "ensemble":
            # For ensemble, we would need learned weights
            # For now, use geometric mean
            if drkg_score > 0 and bioportal_score > 0:
                return math.sqrt(drkg_score * bioportal_score)
            else:
                return max(drkg_score, bioportal_score)
        
        return max(drkg_score, bioportal_score)
    
    def score_claims_batch(self, claims: List[Dict]) -> List[HybridScore]:
        """Score a batch of claims"""
        results = []
        
        for claim in tqdm(claims, desc=f"Scoring ({self.mode})"):
            s = claim.get("s", "")
            r = claim.get("r", "")
            o = claim.get("o", "")
            
            result = self.score_claim(s, r, o)
            results.append(result)
        
        return results


# ============================================================================
# Output Formatters
# ============================================================================

def format_drkg_compatible(claim: Dict, score: HybridScore) -> Dict:
    """Format output to be compatible with existing DRKG format"""
    return {
        **claim,
        "status": score.status,
        "chosen_head": score.drkg_head_id or score.bioportal_head_id,
        "chosen_rel": score.drkg_rel,
        "chosen_tail": score.drkg_tail_id or score.bioportal_tail_id,
        "text_sim": round(max(score.drkg_text_sim, score.bioportal_text_sim), 6),
        "p_kge": score.drkg_kge_score,
        "transe_dist": score.drkg_transe_dist,
        "p_final": round(score.hybrid_final, 6),
        # Additional BioPortal fields
        "bioportal_head_name": score.bioportal_head_name,
        "bioportal_tail_name": score.bioportal_tail_name,
        "bioportal_evidence": score.bioportal_evidence,
        "source": score.source
    }


def format_detailed(claim: Dict, score: HybridScore) -> Dict:
    """Format output with all details"""
    return {
        **claim,
        "status": score.status,
        "source": score.source,
        # DRKG
        "drkg": {
            "head_id": score.drkg_head_id,
            "tail_id": score.drkg_tail_id,
            "rel": score.drkg_rel,
            "text_sim": round(score.drkg_text_sim, 6) if score.drkg_text_sim else None,
            "kge_score": round(score.drkg_kge_score, 6) if score.drkg_kge_score else None,
            "transe_dist": round(score.drkg_transe_dist, 6) if score.drkg_transe_dist else None,
            "final": round(score.drkg_final, 6)
        },
        # BioPortal
        "bioportal": {
            "head_id": score.bioportal_head_id,
            "head_name": score.bioportal_head_name,
            "tail_id": score.bioportal_tail_id,
            "tail_name": score.bioportal_tail_name,
            "ontology_score": round(score.bioportal_ontology_score, 6),
            "text_sim": round(score.bioportal_text_sim, 6),
            "evidence": score.bioportal_evidence,
            "final": round(score.bioportal_final, 6)
        },
        # Fused
        "hybrid_final": round(score.hybrid_final, 6)
    }


# ============================================================================
# Claims Loader
# ============================================================================

def load_claims_from_json(path: Path) -> List[Dict]:
    """Load claims from results_text.json"""
    data = json.load(open(path, "r", encoding="utf-8"))
    claims = []
    
    for item in data.get("results", []):
        qid = item.get("query_id")
        for tpl in item.get("response_claims", []):
            if isinstance(tpl, (list, tuple)) and len(tpl) >= 3:
                s, r, o = tpl[0], tpl[1], tpl[2]
                claims.append(dict(query_id=qid, s=s, r=r, o=o))
    
    return claims


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Hybrid KG scorer combining DRKG and BioPortal"
    )
    
    # Mode selection
    ap.add_argument("--mode", choices=["drkg", "bioportal", "hybrid"],
                    default="hybrid", help="Scoring mode")
    ap.add_argument("--claims", required=True, help="Path to results_text.json")
    ap.add_argument("--outfile", required=True, help="Output JSONL path")
    
    # DRKG params
    ap.add_argument("--embed_dir", type=str, help="DRKG embeddings directory")
    ap.add_argument("--name_map", type=str, help="DRKG name_map.csv path")
    ap.add_argument("--drkg_k", type=int, default=5, help="DRKG entity candidates")
    ap.add_argument("--drkg_m", type=int, default=5, help="DRKG relation candidates")
    ap.add_argument("--drkg_alpha", type=float, default=0.5, help="DRKG fusion weight")
    
    # BioPortal params
    ap.add_argument("--bioportal_key", type=str, help="BioPortal API key")
    ap.add_argument("--bioportal_ontologies", type=str, 
                    default="SNOMEDCT,MESH,RXNORM,DOID",
                    help="Comma-separated ontology list")
    ap.add_argument("--bioportal_cache", type=str, default="./bioportal_cache",
                    help="BioPortal cache directory")
    ap.add_argument("--bioportal_k", type=int, default=5, 
                    help="BioPortal entity candidates")
    ap.add_argument("--bioportal_alpha", type=float, default=0.5,
                    help="BioPortal fusion weight")
    
    # Fusion params
    ap.add_argument("--fusion", choices=["max", "weighted", "ensemble"],
                    default="max", help="Fusion strategy")
    ap.add_argument("--drkg_weight", type=float, default=0.5,
                    help="DRKG weight in weighted fusion")
    
    # Output format
    ap.add_argument("--format", choices=["drkg_compatible", "detailed"],
                    default="drkg_compatible", help="Output format")
    
    args = ap.parse_args()
    
    # Parse ontologies
    ontologies = [x.strip() for x in args.bioportal_ontologies.split(",")]
    
    # Initialize scorer
    scorer = HybridKGScorer(
        mode=args.mode,
        # DRKG
        embed_dir=Path(args.embed_dir) if args.embed_dir else None,
        name_map_path=Path(args.name_map) if args.name_map else None,
        drkg_k=args.drkg_k,
        drkg_m=args.drkg_m,
        drkg_alpha=args.drkg_alpha,
        # BioPortal
        bioportal_key=args.bioportal_key,
        bioportal_ontologies=ontologies,
        bioportal_cache_dir=Path(args.bioportal_cache) if args.bioportal_cache else None,
        bioportal_k=args.bioportal_k,
        bioportal_alpha=args.bioportal_alpha,
        # Fusion
        fusion_strategy=args.fusion,
        drkg_weight=args.drkg_weight
    )
    
    # Load and score claims
    claims = load_claims_from_json(Path(args.claims))
    print(f"Loaded {len(claims)} claims")
    
    results = scorer.score_claims_batch(claims)
    
    # Write output
    Path(args.outfile).parent.mkdir(parents=True, exist_ok=True)
    
    formatter = format_drkg_compatible if args.format == "drkg_compatible" else format_detailed
    
    with open(args.outfile, "w", encoding="utf-8") as f:
        for claim, score in zip(claims, results):
            output = formatter(claim, score)
            f.write(json.dumps(output, ensure_ascii=False) + "\n")
    
    # Print summary
    ok_count = sum(1 for s in results if s.status == "ok")
    drkg_only = sum(1 for s in results if s.source == "drkg")
    bp_only = sum(1 for s in results if s.source == "bioportal")
    hybrid = sum(1 for s in results if s.source == "hybrid")
    
    print(f"\n[Done] Wrote {args.outfile}")
    print(f"  - Total claims: {len(results)}")
    print(f"  - Supported: {ok_count} ({100*ok_count/len(results):.1f}%)")
    print(f"  - Source breakdown:")
    print(f"    - DRKG only: {drkg_only}")
    print(f"    - BioPortal only: {bp_only}")
    print(f"    - Both (hybrid): {hybrid}")


if __name__ == "__main__":
    main()
