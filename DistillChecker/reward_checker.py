# reward_checker.py
# English-only comments: exact-match reward for checker labels.

from typing import Dict

LABELS = ["entailed", "contradicted", "neutral"]
ALIASES = {
    "entailed": {"entailed","entail","entails","supported","yes"},
    "contradicted": {"contradicted","contradict","refuted","no"},
    "neutral": {"neutral","unknown","insufficient","not enough info","not enough information","uncertain"},
}

def _norm_label(text: str):
    t = (text or "").lower().strip()
    t = t.replace(".", "").replace("label:", "").strip()
    for lab, al in ALIASES.items():
        for a in al:
            if t == a or t.startswith(a):
                return lab
    if "contrad" in t or "refut" in t:
        return "contradicted"
    if "neutral" in t or "insufficient" in t or "not enough" in t or "uncertain" in t:
        return "neutral"
    if "entail" in t or "support" in t or t in {"yes","y"}:
        return "entailed"
    return None

class CheckerExactMatchReward:
    """
    A minimal reward function for VERL: takes model output string and sample meta,
    returns a float reward and optional info.
    """
    def __call__(self, model_output: str, sample_meta: Dict) -> Dict:
        gold = sample_meta.get("label", "")
        pred = _norm_label(model_output)
        reward = 1.0 if (pred is not None and pred == gold) else 0.0
        return {
            "reward": reward,
            "metrics": {
                "pred": pred or "none",
                "gold": gold,
                "is_correct": float(reward),
            }
        }
