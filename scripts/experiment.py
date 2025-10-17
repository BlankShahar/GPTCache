"""
Experiment: Similarity-search + Embedding cache behavior under different eviction policies.

This script:
  - Uses GPTCache (FAISS vector base + sqlite cache base) with Ollama via OpenAI-compatible API.
  - Runs the SAME workload across multiple eviction policies (e.g., LFU/LRU/YourPolicy).
  - Measures cache hit rate, LLM latency, total latency, and stability for semantically similar prompts.
  - Writes per-step logs and a compact summary to CSV for later analysis.

Author notes:
  - Your custom eviction policy name can be added to POLICIES_TO_TEST.
  - If your GPTCache build recognizes that name, it will be used as-is.
  - Adjust MAX_SIZE and CLEAN_SIZE to pressure the cache and provoke evictions.
"""

from __future__ import annotations
import os
import time
import hashlib
from dataclasses import dataclass, asdict
import inspect
import argparse
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

import openai
from gptcache import cache
from gptcache.adapter import openai as cached_openai
from gptcache.manager import get_data_manager, CacheBase, VectorBase
from gptcache.similarity_evaluation import SearchDistanceEvaluation
from gptcache.config import Config
# Note: We provide our own chat pre-processor for compatibility across GPTCache versions

# ----------------------------
# Ollama/OpenAI-compatible setup
# ----------------------------
openai.api_base = "http://localhost:11434/v1"
openai.api_key = "ollama"  # ignored by Ollama; must be non-empty

EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.1:8b"

os.environ.setdefault("EMBED_DEBUG", "0")

# Lazily discovered at runtime from the actual embedder path
_INDEX_DIM: Optional[int] = None

# ----------------------------
# Embedding function
# ----------------------------
def ollama_embed(text_or_list, **kwargs):
    """
    Defensive embedder:
    - Accepts str, list/tuple of strs, or any other type (coerces to str).
    - Handles None gracefully.
    Returns:
      - single vector (list[float]) for single string
      - list of vectors for list inputs
    """
    if os.environ.get("EMBED_DEBUG") == "1":
        try:
            print("EMBED INPUT TYPE:", type(text_or_list), "VALUE:", repr(text_or_list))
        except Exception:
            pass

    def _embed_batch(inputs: List[str]):
        global _INDEX_DIM
        resp = openai.Embedding.create(model=EMBED_MODEL, input=inputs)
        vecs = [d["embedding"] for d in resp["data"]]
        if _INDEX_DIM is None:
            _INDEX_DIM = len(vecs[0])
            if os.environ.get("EMBED_DEBUG") == "1":
                try:
                    print("DISCOVERED EMBED DIM:", _INDEX_DIM)
                except Exception:
                    pass
        for i, v in enumerate(vecs):
            if len(v) != _INDEX_DIM:
                raise RuntimeError(
                    f"Embedding dim mismatch in batch at {i}: {len(v)} != {_INDEX_DIM}"
                )
        return vecs

    def _embed_one(s: str):
        s = s if isinstance(s, str) and s.strip() else "<EMPTY>"
        vecs = _embed_batch([s])
        return vecs[0]

    # Case A: single string
    if isinstance(text_or_list, str):
        return _embed_one(text_or_list)

    # Case B: list/tuple -> filter Nones, coerce to str
    if isinstance(text_or_list, (list, tuple)):
        inputs = [v if isinstance(v, str) and v.strip() else "<EMPTY>" for v in text_or_list]
        if len(inputs) == 1:
            return _embed_one(inputs[0])
        return _embed_batch(inputs)

    # Case C: None or any other type -> coerce to str
    s = "" if text_or_list is None else str(text_or_list)
    return _embed_one(s)


def extract_assistant_text(resp):
    """
    Robust post-process function for GPTCache.
    Extracts assistant text from an OpenAI ChatCompletion-like response.
    Falls back to the original object if the shape is unexpected.

    Parameters
    ----------
    resp : Any
        The raw response object passed through the GPTCache adapter.

    Returns
    -------
    str | Any
        Assistant message content if available; otherwise the original object.
    """
    try:
        return resp["choices"][0]["message"]["content"]
    except Exception:
        return resp

def ensure_embed_ready() -> int:
    """
    Probe embeddings at runtime and return the embedding dimension.
    Raises a clear error if the embedding backend is unavailable.
    """
    try:
        v = openai.Embedding.create(model=EMBED_MODEL, input=["probe"])
        return len(v["data"][0]["embedding"])
    except Exception as e:
        raise RuntimeError(
            f"Embeddings unavailable. Ensure Ollama is running and model '{EMBED_MODEL}' is pulled.\n{e}"
        )


def chat_pre_func(data, **kwargs) -> str:
    """
    Extract a stable string to embed from OpenAI ChatCompletion payloads.
    Always returns a string (possibly empty), never None.
    """
    try:
        if isinstance(data, dict) and "messages" in data:
            msgs = data["messages"]
        elif isinstance(data, list) and all(isinstance(m, dict) for m in data):
            msgs = data
        else:
            s = "" if data is None else str(data)
            return s if s.strip() else "<EMPTY>"

        for m in reversed(msgs):
            s = m.get("content") or ""
            if m.get("role") == "user":
                return s if s.strip() else "<EMPTY>"

        s = (msgs[-1].get("content") or "") if msgs else ""
        return s if s.strip() else "<EMPTY>"
    except Exception:
        return "<EMPTY>"


# ----------------------------
# Workload: clusters of near-duplicates
# Each inner list is a "semantic cluster"
# ----------------------------
WORKLOAD_CLUSTERS: List[List[str]] = [
    # Cluster 1: Simple Paraphrasing (should be high similarity)
    [
        "What is the capital of France?",
        "France's capital city is what?",
        "Tell me the capital of France.",
    ],
    # Cluster 2: Subtle Negation (should be low similarity)
    [
        "What are the advantages of using a semantic cache for LLMs?",
        "What are the disadvantages of using a semantic cache for LLMs?",
    ],
    # Cluster 3: Topic Shift (should be low similarity)
    [
        "Write a python function to calculate a factorial.",
        "Write a javascript function to calculate a factorial.",
    ],
    # Cluster 4: Real-world near-duplicates (should be high similarity)
    [
        "Summarize the plot of the movie 'The Matrix'",
        "Give me a short summary of what happens in 'The Matrix'",
        "Can you explain the story of 'The Matrix'?",
    ]
]

WORKLOAD: List[Tuple[int, str]] = [
    (cluster_idx, prompt) for cluster_idx, cluster in enumerate(WORKLOAD_CLUSTERS)
    for prompt in cluster
]

# Optional: repeat workload to see “warm-cache” behavior
REPEATS = 3


@dataclass
class RunConfig:
    """
    RunConfig controls cache pressure and behavior for each policy run.

    Attributes
    ----------
    eviction : str
        Eviction policy name recognized by GPTCache (e.g., "LFU", "LRU", or your custom name).
    max_size : int
        Maximum cache size (#items) before evictions are considered.
    clean_size : int
        Number of items to remove during a clean/eviction cycle.
    """
    eviction: str
    max_size: int = 32
    clean_size: int = 8
    similarity_threshold: float = 0.8


def make_data_manager(eviction: str, max_size: int, clean_size: int):
    """
    Constructs a new GPTCache data manager with sqlite + FAISS and specified eviction policy.

    Notes
    -----
    - This creates a fresh in-memory FAISS index each time we re-init the cache (per policy run).
    - If your custom policy name is available in your GPTCache build, passing `eviction` will activate it.
    """
    # Discover embedding dimension from the same path GPTCache will use
    _ = ollama_embed("faiss_dim_probe")
    if _INDEX_DIM is None:
        raise RuntimeError("Failed to discover embedding dimension.")
    if os.environ.get("EMBED_DEBUG") == "1":
        try:
            print("FAISS index dim =", _INDEX_DIM)
        except Exception:
            pass

    db_dir = os.path.join("results", "db")
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, f"cache_{eviction}_m{max_size}_c{clean_size}.db")
    index_path = os.path.join(db_dir, f"faiss_{eviction}_m{max_size}_c{clean_size}.index")
    sql_url = f"sqlite:///{os.path.abspath(db_path)}"
    return get_data_manager(
        CacheBase("sqlite", sql_url=sql_url),                   # persistent key/value store for cache entries
        VectorBase("faiss", dimension=_INDEX_DIM, index_path=index_path),              # vector index for similarity search
        eviction=eviction,
        max_size=max_size,
        clean_size=clean_size,
    )


def init_cache_for_policy(cfg: RunConfig):
    """
    Initialize GPTCache with a specific eviction policy and similarity evaluator.

    This re-initializes the global cache singleton for each policy run.
    """
    dm = make_data_manager(cfg.eviction, cfg.max_size, cfg.clean_size)

    def make_similarity_eval(threshold: float = 0.35):
        sig = inspect.signature(SearchDistanceEvaluation.__init__)
        if "distance_threshold" in sig.parameters:
            return SearchDistanceEvaluation(distance_threshold=threshold)
        return SearchDistanceEvaluation()

    init_sig = inspect.signature(cache.init)
    supported = set(init_sig.parameters.keys())

    init_kwargs = {
        "embedding_func": ollama_embed,
        "data_manager": dm,
        "similarity_evaluation": make_similarity_eval(0.35),
    }
    if "pre_embedding_func" in supported:
        init_kwargs["pre_embedding_func"] = chat_pre_func

    # Use configured similarity threshold
    if "config" in supported:
        init_kwargs["config"] = Config(similarity_threshold=cfg.similarity_threshold)

    cache.init(**init_kwargs)


def ask_llm(prompt: str) -> Dict[str, Any]:
    """
    Calls LLM via GPTCache adapter and returns the raw result dict.
    We rely on GPTCache's adapter to attach 'gptcache_meta' with hit/miss + timings.
    """
    result = cached_openai.ChatCompletion.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return result


def run_once(policy_cfg: RunConfig, model_name: str, repeats: int = 1) -> pd.DataFrame:
    """
    Run the workload `repeats` times for a given eviction policy, recording per-step metrics.

    Returns
    -------
    DataFrame with columns:
        policy, step, cluster, prompt, hit (bool), llm_time_s (float|None), total_time_s (float|None)
        miss_reason (str|None), cache_size (int|None)
    """
    init_cache_for_policy(policy_cfg)
    # Sanity: exact repeat should hit on the second call
    def _is_cache_hit(resp: Any) -> bool:
        try:
            if isinstance(resp, dict):
                meta = resp.get("gptcache_meta", {}) or {}
                return bool(resp.get("gptcache")) or (meta.get("hit") is True)
        except Exception:
            pass
        return False

    sanity_a = cached_openai.ChatCompletion.create(
        model=CHAT_MODEL, messages=[{"role":"user","content":"probe"}], temperature=0
    )
    sanity_b = cached_openai.ChatCompletion.create(
        model=CHAT_MODEL, messages=[{"role":"user","content":"probe"}], temperature=0
    )
    try:
        print(
            "Sanity check:",
            {"a_meta": getattr(sanity_a, "get", lambda *_: None)("gptcache_meta"),
             "b_meta": getattr(sanity_b, "get", lambda *_: None)("gptcache_meta"),
             "b_gptcache": getattr(sanity_b, "get", lambda *_: None)("gptcache")},
        )
    except Exception:
        pass
    assert _is_cache_hit(sanity_b), \
        "GPTCache did not return a detectable hit on identical second call; check pre/post/adapter/threshold."

    rows: List[Dict[str, Any]] = []
    step = 0
    for r in range(repeats):
        for cluster_idx, prompt in WORKLOAD:
            step += 1
            t0 = time.time()
            raw = ask_llm(prompt)
            t1 = time.time()

            meta = raw.get("gptcache_meta", {}) or {}
            # Use the same robust check as the sanity test to determine a hit
            is_hit = bool(raw.get("gptcache")) or (meta.get("hit") is True)

            rows.append({
                "embedding_model": model_name,
                "threshold": policy_cfg.similarity_threshold,
                "policy": policy_cfg.eviction,
                "max_size": policy_cfg.max_size,
                "clean_size": policy_cfg.clean_size,
                "repeat": r,
                "step": step,
                "cluster": cluster_idx,
                "prompt": prompt,
                "hit": is_hit, # Use our reliable is_hit variable
                # If it's a hit, there is no LLM time.
                "llm_time_s": meta.get("llm_time_s") if not is_hit else 0.0,
                "total_time_s": meta.get("total_time_s", (t1 - t0)),
                "miss_reason": meta.get("miss_reason"),
                "cache_size": meta.get("cache_size"),
                "similarity": meta.get("similarity") or meta.get("score") or meta.get("sim"),
                "distance": meta.get("distance"),
            })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a compact summary by policy.

    Metrics
    -------
    - hit_rate: (#hits / total)
    - avg_llm_time_s: mean of llm_time_s over all rows (ignores None)
    - avg_total_time_s: mean of total_time_s
    - first_hit_step: first step index where hit==True (smaller is better warm-up)
    - cluster_hit_rate_mean: average of per-cluster hit rates
    """
    def _first_hit(g: pd.DataFrame) -> Optional[int]:
        hits = g.loc[g["hit"] == True, "step"]
        return int(hits.min()) if not hits.empty else None

    # per-experiment aggregates (by model, threshold, and policy)
    summary_rows = []
    group_cols = [
        c for c in ["embedding_model", "threshold", "policy", "max_size", "clean_size"]
        if c in df.columns
    ]
    for keys, g in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        total = len(g)
        hit_rate = float(g["hit"].fillna(False).mean())

        avg_llm = float(g["llm_time_s"].dropna().mean()) if g["llm_time_s"].notna().any() else None
        avg_total = float(g["total_time_s"].dropna().mean()) if g["total_time_s"].notna().any() else None
        first_hit = _first_hit(g)

        # Per-cluster hit rate, then average across clusters (measures semantic stability)
        cluster_rates = g.assign(hit=g["hit"].fillna(False)).groupby("cluster")["hit"].mean()
        cluster_hit_rate_mean = float(cluster_rates.mean()) if not cluster_rates.empty else None

        row = {
            "rows": total,
            "hit_rate": hit_rate,
            "avg_llm_time_s": avg_llm,
            "avg_total_time_s": avg_total,
            "first_hit_step": first_hit,
            "cluster_hit_rate_mean": cluster_hit_rate_mean,
        }

        for col, val in zip(group_cols, keys):
            row[col] = val
        summary_rows.append(row)

    sort_cols = [c for c in ["embedding_model", "threshold", "policy", "hit_rate", "cluster_hit_rate_mean"] if c in group_cols or c in ["hit_rate", "cluster_hit_rate_mean"]]
    return pd.DataFrame(summary_rows).sort_values(sort_cols, ascending=[True, True, True, False, False][:len(sort_cols)])


def main():
    """
    Run a single experiment configuration, controlled by command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Run a single GPTCache experiment.")
    parser.add_argument('--embedding-model', type=str, required=True, help='Name of the embedding model to test.')
    parser.add_argument('--similarity-threshold', type=float, required=True, help='Similarity threshold for cache hits.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--output-dir', type=str, default="results", help='Directory to save experiment results.')
    args = parser.parse_args()

    run_output_dir = os.path.join(args.output_dir, args.embedding_model, str(args.similarity_threshold), f"seed_{args.seed}")
    os.makedirs(run_output_dir, exist_ok=True)

    global EMBED_MODEL
    EMBED_MODEL = args.embedding_model
    global _INDEX_DIM
    _INDEX_DIM = None

    policy = "LFU"
    max_size, clean_size = 16, 4

    cfg = RunConfig(
        eviction=policy,
        max_size=max_size,
        clean_size=clean_size,
        similarity_threshold=args.similarity_threshold,
    )

    print(f"\n=== Running Config ===")
    print(f"  Model: {args.embedding_model}")
    print(f"  Threshold: {args.similarity_threshold}")
    print(f"  Seed: {args.seed}")
    print(f"======================")

    try:
        run_df = run_once(cfg, model_name=args.embedding_model, repeats=REPEATS)

        if run_df.empty:
            print("WARNING: Experiment produced no data.")
            return

        run_df.to_csv(os.path.join(run_output_dir, "experiment_run.csv"), index=False)

        summary = summarize(run_df)
        summary.to_csv(os.path.join(run_output_dir, "summary.csv"), index=False)

        print(f"\n\u2705 Success! Results saved to: {run_output_dir}")
        print("\nSummary for this run:")
        print(summary.to_string(index=False))

    except Exception as e:
        print(f"\n\u274c FAILED to run experiment for model '{args.embedding_model}' with threshold {args.similarity_threshold}.")
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
