# comments in English only
# python end2end_drkg_softscore.py   --claims ./dataset/results_text.json   --embed_dir ./KG/DRKG/embed   --name_map ./aux/name_map.csv   --outfile ./work/drkg_ke/soft_transe_scores.jsonl   --rel_filter "(Compound,Disease)|(Compound,Side Effect)|(Gene,Disease)|(Compound,Gene)"   --k 5 --m 5 --alpha 0.5
# awk -F, 'NR>1{split($2,a,"::"); t=a[1]; c[t]++} END{for(k in c) print k,c[k]}' ./aux/name_map.csv | sort

import argparse, json, re, csv, math
from pathlib import Path
from typing import Dict, List, Tuple, Set
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss

# ---------- text & relation helpers ----------
def norm_txt(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\(.*?\)|\[.*?\]", " ", s)
    s = re.sub(r"\b(tablets?|capsules?|injection|solution|suspension|cream|gel|patch|drops?|spray|mg|ml|mcg|units?)\b", " ", s)
    s = re.sub(r"[^a-z0-9+/\s-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

REL_CANON = {
    # canonical relation key -> list of paraphrases (for text similarity over relations)
    "CtD": ["treats", "used to treat", "therapy for", "first-line treatment", "indicated for"],
    "CpD": ["palliates", "relieves", "alleviates", "improves symptoms"],
    "CcSE": ["causes side effect", "induces adverse effect", "side effects include"],
    "GaD": ["gene associated with disease", "association with disease"],
    "CbG": ["binds gene", "targets protein", "inhibits protein", "antagonist of"],
}

# restrict plausible relation set by (head_type, tail_type)
REL_TYPE_FILTER = {
    ("Compound","Disease"): ["CtD","CpD","CcSE"],
    ("Compound","Side Effect"): ["CcSE"],
    ("Gene","Disease"): ["GaD"],
    ("Compound","Gene"): ["CbG"],
    ("Compound","Compound"): [],
}
# comments in English only
def topk_entities_any(enc, indices, alias_to_entity, text, k=5):
    q = enc.encode([norm_txt(text)], normalize_embeddings=True, convert_to_numpy=True)
    agg = {}
    for t, idx in indices.items():
        D, I = idx.search(q, min(k*5, len(alias_to_entity[t])))
        for score, ii in zip(D[0], I[0]):
            rid = alias_to_entity[t][ii]
            agg[rid] = max(agg.get(rid, 0.0), float(score))
    return sorted(agg.items(), key=lambda x: -x[1])[:k]

# ---------- DRKG loading ----------
def load_id_maps(embed_dir: Path):
    ents = pd.read_csv(embed_dir/"entities.tsv", sep="\t", header=None, names=["raw_id","idx"])
    rels = pd.read_csv(embed_dir/"relations.tsv", sep="\t", header=None, names=["raw_id","idx"])
    ent2idx = dict(zip(ents["raw_id"], ents["idx"]))
    rel2idx = dict(zip(rels["raw_id"], rels["idx"]))
    return ent2idx, rel2idx

def load_embeddings(embed_dir: Path):
    E = np.load(embed_dir/"DRKG_TransE_l2_entity.npy")
    R = np.load(embed_dir/"DRKG_TransE_l2_relation.npy")
    return E, R

def id2type(raw_id: str) -> str:
    # DRKG raw ids look like "Compound::DB00316", "Disease::DOID:4166", "Gene::2157", "Side Effect::C0032584"
    return raw_id.split("::", 1)[0] if "::" in raw_id else "Unknown"

# ---------- alias index (FAISS over SapBERT) ----------
def build_alias_index(name_map_csv: Path):
    by_type: Dict[str, Dict[str, List[str]]] = {}  # type -> {raw_id: [aliases]}
    with open(name_map_csv, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            name = row.get("name","").strip()
            rid  = row.get("raw_id","").strip()
            if not name or not rid: 
                continue
            t = id2type(rid)
            by_type.setdefault(t, {}).setdefault(rid, []).append(name)

    # prepare FAISS per type
    enc = SentenceTransformer("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")  # SapBERT for biomedical synonymy
    indices: Dict[str, faiss.IndexFlatIP] = {}
    alias_to_entity: Dict[str, List[str]] = {}  # type -> alias-level entity id list parallel to index
    alias_texts: Dict[str, List[str]] = {}

    for t, d in by_type.items():
        aliases, owners = [], []
        for rid, names in d.items():
            for nm in names:
                aliases.append(norm_txt(nm))
                owners.append(rid)
        if not aliases:
            continue
        emb = enc.encode(aliases, normalize_embeddings=True, convert_to_numpy=True)
        idx = faiss.IndexFlatIP(emb.shape[1])  # dot on normalized vectors == cosine
        idx.add(emb)
        indices[t] = idx
        alias_to_entity[t] = owners
        alias_texts[t] = aliases

    return indices, alias_to_entity, alias_texts, enc

def topk_entities(enc, indices, alias_to_entity, text: str, t: str, k=5):
    if t not in indices:
        return []
    q = enc.encode([norm_txt(text)], normalize_embeddings=True, convert_to_numpy=True)
    D, I = indices[t].search(q, min(k*10, len(alias_to_entity[t])))  # search more aliases first
    agg: Dict[str, float] = {}
    for score, idx in zip(D[0], I[0]):
        rid = alias_to_entity[t][idx]
        agg[rid] = max(agg.get(rid, 0.0), float(score))  # aggregate by max-alias score
    return sorted(agg.items(), key=lambda x: -x[1])[:k]  # list[(raw_id, sim)]

# ---------- relation soft mapping ----------
def build_relation_index():
    # build small relation embedding bank
    enc = SentenceTransformer("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    keys, vecs = [], []
    for r, phrases in REL_CANON.items():
        v = enc.encode(phrases, normalize_embeddings=True, convert_to_numpy=True).mean(axis=0)
        vecs.append(v)
        keys.append(r)
    mat = np.vstack(vecs)
    idx = faiss.IndexFlatIP(mat.shape[1])
    idx.add(mat)
    return enc, idx, keys

def topm_relations(rel_enc, rel_idx, rel_keys, text: str, m=5, type_filter: List[str]=None):
    q = rel_enc.encode([norm_txt(text)], normalize_embeddings=True, convert_to_numpy=True)
    D, I = rel_idx.search(q, m + 5)
    cands = [(rel_keys[j], float(D[0][i])) for i, j in enumerate(I[0])]
    if type_filter is not None and len(type_filter) > 0:
        cands = [(r,s) for (r,s) in cands if r in type_filter]
    return cands[:m]

# ---------- TransE scoring ----------
def transe_score(E, R, ent2idx, rel2idx, h_id: str, r_key: str, t_id: str):
    # In DRKG, relation raw ids are like "Compound::treats::Disease"; we use canonical keys (CtD/CpD/...) to map.
    # Minimal mapping from canonical to DRKG raw ids (extend as needed).
    CANON2RAW = {
        "CtD": "Compound::treats::Disease",
        "CpD": "Compound::palliates::Disease",
        "CcSE": "Compound::causes::Side Effect",
        "GaD": "Gene::associates::Disease",
        "CbG": "Compound::binds::Gene",
    }
    r_raw = CANON2RAW.get(r_key)
    if h_id not in ent2idx or t_id not in ent2idx or r_raw not in rel2idx:
        return None
    h = E[ent2idx[h_id]]
    t = E[ent2idx[t_id]]
    r = R[rel2idx[r_raw]]
    diff = h + r - t
    dist = np.linalg.norm(diff, ord=2)
    score = -float(dist)
    prob = 1.0 / (1.0 + math.exp(-(-dist)))  # sigmoid(-L2)
    return score, prob, dist

# ---------- claims loader ----------
def load_claims(results_json_path: Path):
    data = json.load(open(results_json_path, "r", encoding="utf-8"))
    claims = []
    for item in data.get("results", []):
        qid = item.get("query_id")
        for tpl in item.get("response_claims", []):
            if isinstance(tpl, (list, tuple)) and len(tpl) >= 3:
                s, r, o = tpl[0], tpl[1], tpl[2]
                claims.append(dict(query_id=qid, s=s, r=r, o=o))
    return claims

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", required=True, help="results_text.json")
    ap.add_argument("--embed_dir", required=True, help="DRKG embed dir with npy/tsv")
    ap.add_argument("--name_map", help="name_map.csv (name,raw_id). Strongly recommended")
    ap.add_argument("--outfile", required=True, help="output JSONL")
    ap.add_argument("--k", type=int, default=5, help="top-k entity candidates per side")
    ap.add_argument("--m", type=int, default=5, help="top-m relation candidates")
    ap.add_argument("--alpha", type=float, default=0.5, help="fusion weight: final=(1-alpha)*KGE + alpha*text")
    ap.add_argument("--rel_filter", default="", help="pipe of allowed (H,T) types, e.g. (Compound,Disease)|(Compound,Side Effect)")
    args = ap.parse_args()

    embed_dir = Path(args.embed_dir)
    ent2idx, rel2idx = load_id_maps(embed_dir)
    E, R = load_embeddings(embed_dir)

    # build alias indices
    if not args.name_map:
        print("[warn] name_map.csv not provided; entity recall will be low.")
        return
    indices, owner_lists, alias_texts, ent_enc = build_alias_index(Path(args.name_map))

    # relation index
    rel_enc, rel_idx, rel_keys = build_relation_index()

    # parse relation filter spec
    filt_pairs: Set[Tuple[str,str]] = set()
    spec = args.rel_filter.strip()
    if spec:
        for seg in spec.split("|"):
            mt = re.match(r"\(([^,]+),([^,\)]+)\)", seg.strip())
            if mt:
                filt_pairs.add((mt.group(1), mt.group(2)))

    claims = load_claims(Path(args.claims))
    out = open(args.outfile, "w", encoding="utf-8")
    hit_e, total = 0, 0

    for c in tqdm(claims, desc="Scoring"):
        total += 1
        s_txt, r_txt, o_txt = c["s"], c["r"], c["o"]

        # try all type pairs; if filter is set, only those
        type_pairs = list(filt_pairs) if len(filt_pairs)>0 else [
            ("Compound","Disease"), ("Compound","Side Effect"), ("Compound","Gene"),
            ("Gene","Disease"), ("Compound","Compound")
        ]

        best = None
        best_parts = None
        best_textsim = 0.0

        for (ht, tt) in type_pairs:
            # entity candidates
            H = topk_entities(ent_enc, indices, owner_lists, s_txt, ht, k=args.k)
            T = topk_entities(ent_enc, indices, owner_lists, o_txt, tt, k=args.k)
            if not H or not T:
                continue
            # relation candidates
            rset = REL_TYPE_FILTER.get((ht, tt), [])
            RC = topm_relations(rel_enc, rel_idx, rel_keys, r_txt, m=args.m, type_filter=rset if rset else None)

            # text similarity aggregate
            for (h_id, sim_h) in H:
                for (r_key, sim_r) in RC:
                    for (t_id, sim_t) in T:
                        text_sim = float(min(sim_h, sim_t)) * (0.5 + 0.5*sim_r)  # scale relation sim to [0.5,1]
                        kge = transe_score(E, R, ent2idx, rel2idx, h_id, r_key, t_id)
                        if kge is None:
                            # fall back to text-only
                            final_score = text_sim
                            candidate = (h_id, r_key, t_id, text_sim, None, None)
                        else:
                            score, p_kge, dist = kge
                            p_final = (1-args.alpha)*p_kge + args.alpha*text_sim
                            final_score = p_final
                            candidate = (h_id, r_key, t_id, text_sim, p_kge, dist)

                        if (best is None) or (final_score > best):
                            best = final_score
                            best_parts = candidate

        status = "ok" if best is not None else "no_candidate"
        if best_parts:
            h_id, r_key, t_id, text_sim, p_kge, dist = best_parts
            if p_kge is not None:
                hit_e += 1
            out.write(json.dumps({
                **c,
                "status": status,
                "chosen_head": h_id,
                "chosen_rel": r_key,
                "chosen_tail": t_id,
                "text_sim": round(float(text_sim), 6),
                "p_kge": None if p_kge is None else round(float(p_kge), 6),
                "transe_dist": None if dist is None else round(float(dist), 6),
                "p_final": None if best is None else round(float(best), 6)
            }) + "\n")
        else:
            out.write(json.dumps({**c, "status": status, "p_final": 0.0}) + "\n")

    out.close()
    print(f"[done] wrote {args.outfile}; kge_hit={hit_e}/{total}")
    
if __name__ == "__main__":
    main()
