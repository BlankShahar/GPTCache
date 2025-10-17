"""
scripts/eval_cache_bank77_hf.py
-------------------------------------

Evaluate GPTCache on BANKING77 with intent–paraphrase queries.
Replaces custom vector cache with GPTCache's built-in API:

  • Cache init: embedding_func, FAISS vector store, eviction policy & capacity.
  • Similarity decision: GPTCache similarity_evaluation (+ global threshold).
  • LLM calls: via gptcache.adapter.openai → works with Ollama's OAI-compatible API.

Metrics reported (per policy × threshold):
  - Should-Hit: queries whose intent has been seen earlier in the stream.
  - Hits: GPTCache-reported cache hits.
  - Hit-Recall: hits / should_hit.
  - False-Hit (lower bound): hits that occur before the first time that intent appeared.
  - Avg LLM time (on GPT-backed responses only, from gptcache_meta.llm_time_s).
  - Avg cache time (lookup time per query, from total_time_s - llm_time_s).
  - Hit Rate, LLM call count.

Notes
  * Policies supported by GPTCache today: LRU, FIFO. (LFU is not provided.) [docs]
  * Threshold is configured via Config(similarity_threshold=...).
  * For "cosine"-like behavior, we map to NumpyNormEvaluation(enable_normal=True),
    which normalizes embeddings so L2-distance is monotonic with cosine similarity.

References:
  - GPTCache Quick Start & Manager config (capacity/eviction): see docs. 
  - Evaluators: ExactMatch, NumpyNorm, SearchDistance, SBERT, ONNX, Cohere.
  - OpenAI adapter wrapper and gptcache_meta fields (hit, llm_time_s, total_time_s).

"""
from __future__ import annotations

import argparse
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import csv
import sys
from datetime import datetime
import traceback
# --- Hugging Face datasets & helpers (unchanged) ---
from datasets import load_dataset

# --- GPTCache core pieces ---
from gptcache import cache
from gptcache.config import Config
from gptcache.processor.pre import last_content
from gptcache.adapter import openai as cached_openai
from gptcache.manager import manager_factory

# Embedding backends (choose at CLI): ONNX or SBERT are the simplest & local
from gptcache.embedding import Onnx as EmbOnnx
from gptcache.embedding import SBERT as EmbSBERT

# Similarity evaluators — pick via --sim-eval
from gptcache.similarity_evaluation import (
    ExactMatchEvaluation,
    NumpyNormEvaluation,
    SearchDistanceEvaluation,
    SbertCrossencoderEvaluation,
    OnnxModelEvaluation,
    CohereRerankEvaluation,
    KReciprocalEvaluation,
)

# Answer correctness judge dependencies
from sentence_transformers import SentenceTransformer
import numpy as np
from gptcache.utils.error import NotInitError

# DEBUG LOGGING: show the real error behind "failed to save"
import logging
logging.getLogger("gptcache").setLevel(logging.DEBUG)
os.environ["GPTCACHE_LOG_LEVEL"] = "DEBUG"
os.environ["SQLALCHEMY_ECHO"] = "1"


# --- Silence noisy transformers/torch warnings for cleaner logs ---
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
try:
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
except Exception:
    pass


# =========================
# Data loading (BANKING77)
# =========================
@dataclass
class Example:
    text: str
    intent: str

def load_banking77(split: str = "train", limit: int = 0, seed: int = 42) -> List[Example]:
    """
    Load BANKING77 via the mteb mirror (preferred), fallback to PolyAI parquet branch if needed.
    Keeps behavior identical to your original loader.
    """
    def _subsample_and_shuffle(rows: List[Example]) -> List[Example]:
        if limit and limit < len(rows):
            rnd = random.Random(seed)
            rows = rnd.sample(rows, limit)
        random.Random(seed).shuffle(rows)
        return rows

    try:
        ds = load_dataset("mteb/banking77", split=split)
        if "label_text" in ds.column_names:
            rows = [Example(text=r["text"], intent=r["label_text"]) for r in ds]
        else:
            names = ds.features["label"].names
            rows = [Example(text=r["text"], intent=names[int(r["label"])]) for r in ds]
        return _subsample_and_shuffle(rows)
    except Exception:
        ds = load_dataset("PolyAI/banking77", split=split, revision="refs/convert/parquet")
        if "label_text" in ds.column_names:
            rows = [Example(text=r["text"], intent=r["label_text"]) for r in ds]
        else:
            names = ds.features["label"].names
            rows = [Example(text=r["text"], intent=names[int(r["label"])]) for r in ds]
        return _subsample_and_shuffle(rows)

def load_banking77_k_per_intent(
    split: str = "train", num_intents: int = 10, per_intent: int = 4, seed: int = 42
) -> List[Example]:
    """
    Build a stream with `num_intents` intents and `per_intent` paraphrases each, then shuffle.
    Guarantees repeats so the cache has real hit opportunities.
    """
    ds = load_dataset("mteb/banking77", split=split)
    if "label_text" in ds.column_names:
        def get_intent(rec): return rec["label_text"]
    else:
        names = ds.features["label"].names
        def get_intent(rec): return names[int(rec["label"])]

    by_intent: Dict[str, List[str]] = {}
    for rec in ds:
        by_intent.setdefault(get_intent(rec), []).append(rec["text"])

    rng = random.Random(seed)
    intents = rng.sample(list(by_intent.keys()), min(num_intents, len(by_intent)))
    examples: List[Example] = []
    for it in intents:
        pool = by_intent[it]
        take = min(per_intent, len(pool))
        for txt in rng.sample(pool, take):
            examples.append(Example(text=txt, intent=it))
    rng.shuffle(examples)
    return examples


# -------------------------
# Canonical answers per intent
# -------------------------
def build_canonical_answers(intents: List[str]) -> Dict[str, str]:
    """
    Create a canonical answer for each BANKING77 intent.
    """
    canon: Dict[str, str] = {}
    for it in intents:
        canon[it] = (
            f"[Intent={it}] This is the standard resolution for '{it}'. "
            f"Follow these steps carefully to handle '{it}'."
        )
    return canon


# -------------------------
# Lightweight cosine judge on raw text
# -------------------------
class AnswerJudge:
    """
    SBERT-based cosine similarity between the model's answer and the canonical answer.
    We treat cosine >= threshold as 'correct'.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: Optional[str] = None, threshold_cos: float = 0.60):
        self.model = SentenceTransformer(model_name, device=device)
        self.threshold = float(threshold_cos)

    def score(self, a: str, b: str) -> float:
        ea = self.model.encode([a], normalize_embeddings=True, convert_to_numpy=True)[0]
        eb = self.model.encode([b], normalize_embeddings=True, convert_to_numpy=True)[0]
        return float(np.dot(ea, eb))

    def is_correct(self, answer_text: str, gold_text: str) -> bool:
        return self.score(answer_text, gold_text) >= self.threshold

# =========================
# GPTCache setup helpers
# =========================
def _make_embedder(name: str, sbert_model: str) -> object:
    """
    Create an embedding object compatible with GPTCache (has .to_embeddings and .dimension).
    - "onnx"  → EmbOnnx()          (small, CPU-friendly)
    - "sbert" → EmbSBERT(model)    (best quality per watt, local)
    """
    name = name.lower()
    if name == "onnx":
        return EmbOnnx()
    elif name == "sbert":
        return EmbSBERT(sbert_model)
    else:
        raise ValueError(f"--embedder must be 'onnx' or 'sbert'; got {name!r}")

def _make_evaluator(
    sim_eval_name: str,
    *,
    sbert_xenc_model: str,
    onnx_model: str,
    cohere_model: str,
    cohere_api_key: Optional[str],
):
    """
    Translate CLI --sim-eval to a GPTCache SimilarityEvaluation object.
    We also allow 'cosine' as a user-friendly alias that maps to NumpyNormEvaluation(enable_normal=True).
    """
    n = sim_eval_name.lower()
    if n == "exact":
        return ExactMatchEvaluation()
    if n == "np":
        return NumpyNormEvaluation(enable_normal=True)
    if n == "distance":
        return SearchDistanceEvaluation()
    if n == "sbert":
        return SbertCrossencoderEvaluation(model=sbert_xenc_model)
    if n == "onnx":
        return OnnxModelEvaluation(model=onnx_model)
    if n == "cohere":
        if not (cohere_api_key or os.getenv("COHERE_API_KEY")):
            raise ValueError("Cohere evaluator requires --cohere-api-key or COHERE_API_KEY.")
        return CohereRerankEvaluation(model=cohere_model, api_key=cohere_api_key or os.getenv("COHERE_API_KEY"))
    if n == "cosine":
        # Practical cosine-style acceptance using normalized L2 distance
        return NumpyNormEvaluation(enable_normal=True)
    raise ValueError(f"Unknown --sim-eval '{sim_eval_name}'. "
                     f"Choose from cosine|exact|np|distance|sbert|onnx|cohere|krecip")

def _init_gptcache_once(
    *,
    threshold: float,
    capacity: int,
    policy: str,
    embedder_name: str,
    sbert_model: str,
    sim_eval_name: str,
    sbert_xenc_model: str,
    onnx_model: str,
    cohere_model: str,
    cohere_api_key: Optional[str],
    db_path: Optional[str] = None,
    index_path: Optional[str] = None,
    top_k: int = 5,
    krecip_topk: int = 3,
    krecip_max_distance: float = 4.0,
    krecip_positive: bool = False,
    vector_backend: str = "faiss",
):
    """
    (Re)initialize GPTCache with the requested components for a single run configuration.

    - capacity      → DataManager(max_size=capacity)
    - policy        → eviction policy ("LRU" or "FIFO") supported by GPTCache
    - threshold     → Config(similarity_threshold=threshold) for accept/reject
    - embedder_name → "onnx" or "sbert"
    - sim_eval_name → evaluator for acceptance (e.g., distance/np/exact/sbert/onnx/cohere)
    """
    # 1) Embedding backend
    emb = _make_embedder(embedder_name, sbert_model=sbert_model)

    # Safe wrapper: preserve GPTCache signature and return 1-D float32 for single embedding
    def _embedding_func(data, extra_param=None, **kwargs):
        v = emb.to_embeddings(data, extra_param=extra_param)
        v = np.asarray(v, dtype=np.float32)
        # Flatten a single-row matrix to 1-D (e.g., (1, 384) -> (384,))
        if v.ndim == 2 and v.shape[0] == 1:
            v = v[0]
        return v

    # Probe through the wrapper to determine actual embedding dimension
    try:
        probe = _embedding_func(["hello"])
        print(f"[gptcache] probe_type={type(probe)}")
        try:
            print(f"[gptcache] probe_shape={probe.shape}")
        except Exception:
            pass
        if isinstance(probe, np.ndarray):
            if probe.ndim == 1:
                emb_dim = int(probe.shape[0])
            elif probe.ndim == 2:
                emb_dim = int(probe.shape[1])
            else:
                emb_dim = int(probe.shape[-1])
        else:
            try:
                emb_dim = int(len(probe[0]))
            except Exception:
                emb_dim = int(len(probe))
    except Exception as e:
        print("[gptcache] embedding failed:", e)
        emb_dim = getattr(emb, 'dimension', None) or 0
    print(f"[gptcache] embedder={embedder_name} model={sbert_model} dim={emb_dim}")

    # 2) DataManager = [scalar store: sqlite] + [vector store: faiss]
    # Ensure parent folders exist for provided paths (Windows-safe)
    def _ensure_parent(path_str: Optional[str]) -> None:
        if path_str:
            parent_dir = os.path.abspath(os.path.dirname(path_str))
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)
    _ensure_parent(db_path)
    _ensure_parent(index_path)

    # Build DataManager via manager_factory with top_k support
    backend = (vector_backend or "faiss").lower()
    vector_params = {"dimension": emb_dim, "top_k": int(top_k)}
    if backend == "faiss" and index_path:
        vector_params["index_path"] = index_path
    scalar_params = {"sql_url": db_path} if db_path else None
    print(f"[gptcache] vector_params={vector_params} db_path={db_path} index_path={index_path}")
    try:
        data_manager = manager_factory(
            f"sqlite,{backend}",
            vector_params=vector_params,
            scalar_params=scalar_params,
            eviction_params={"max_size": int(capacity), "policy": policy.upper()},
        )
    except Exception:
        data_manager = manager_factory(
            f"sqlite,{backend}",
            vector_params=vector_params,
            scalar_params=scalar_params,
            eviction_params={"maxsize": int(capacity), "policy": policy.upper()},
        )
    try:
        print(
            "[gptcache] stores:",
            "vector_base=", type(data_manager.vector_base).__name__,
            "scalar_base=", type(data_manager.scalar_base).__name__,
        )
    except Exception:
        pass

    # 3) Similarity evaluator (special-case K-Reciprocal which needs the vector DB)
    if sim_eval_name.lower() == "krecip":
        vectordb = data_manager.vector_base
        evaluator = KReciprocalEvaluation(
            vectordb=vectordb,
            top_k=int(krecip_topk),
            max_distance=float(krecip_max_distance),
            positive=bool(krecip_positive),
        )
    else:
        evaluator = _make_evaluator(
            sim_eval_name,
            sbert_xenc_model=sbert_xenc_model,
            onnx_model=onnx_model,
            cohere_model=cohere_model,
            cohere_api_key=cohere_api_key,
        )

    # Optional scale hint for distance-based evaluator
    if sim_eval_name.lower() in ("distance",):
        print("Note: distance-based evaluator typically treats LOWER values as more similar; tune τ accordingly.")

    # 4) Global similarity threshold
    cfg = Config(similarity_threshold=float(threshold))

    # 5) Initialize GPTCache
    #    pre_embedding_func=last_content extracts the last chat message content
    cache.init(
        embedding_func=_embedding_func,
        data_manager=data_manager,
        similarity_evaluation=evaluator,
        pre_embedding_func=last_content,
        config=cfg,
    )

    # Probe a save path via adapter to surface storage errors early
    try:
        _ = cached_openai.ChatCompletion.create(
            model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
            messages=[{"role": "user", "content": "__probe__"}],
            max_tokens=8,
            temperature=0.0,
            timeout=30,
        )
        print("[gptcache] probe save passed (adapter path)")
    except Exception as e:
        print("[gptcache] probe save failed:", e)


# =========================
# Experiment loop (GPTCache)
# =========================
def run_eval(
    *,
    split: str,
    limit: int,
    thresholds: List[float],
    embedder_name: str,
    sbert_model: str,
    capacity: int,
    policies: List[str],
    provider: str,
    rpm: float,
    km_intents: int,
    km_per_intent: int,
    sim_eval_name: str,
    sbert_xenc_model: str,
    onnx_model: str,
    cohere_model: str,
    cohere_api_key: Optional[str],
    seed: int,
    answer_judge: str,
    answer_threshold: float,
    answer_model: str,
    db_path: Optional[str],
    index_path: Optional[str],
    topk: int,
    out_dir: str,
    krecip_topk: int,
    krecip_max_distance: float,
    krecip_positive: bool,
    vector_backend: str,
):
    """
    Execute evaluation with GPTCache as the cache backend.
    """
    # --- Choose data stream (plain or K×M to force repeats) ---
    if km_intents and km_per_intent:
        stream = load_banking77_k_per_intent(
            split=split, num_intents=km_intents, per_intent=km_per_intent, seed=seed
        )
    else:
        stream = load_banking77(split=split, limit=limit, seed=seed)

    total = len(stream)

    # Build canonical answers and optional correctness judge
    all_intents = sorted({ex.intent for ex in stream})
    canonical = build_canonical_answers(all_intents)
    judge = AnswerJudge(model_name=answer_model, threshold_cos=answer_threshold) if answer_judge == "sbert" else None

    # Seeding for reproducibility
    try:
        import torch  # type: ignore
    except Exception:
        torch = None
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        try:
            torch.manual_seed(seed)
        except Exception:
            pass
    os.environ["PYTHONHASHSEED"] = str(seed)

    # --- Configure OpenAI-compatible client for Ollama (or keep OPENAI defaults) ---
    # Your partner's snippet: point OpenAI SDK to Ollama and dummy key "ollama".
    import openai as openai_sdk
    if provider == "ollama":
        openai_sdk.api_base = os.getenv("OLLAMA_API_BASE", "http://localhost:11434/v1")
        openai_sdk.api_key = os.getenv("OLLAMA_API_KEY", "ollama")
    elif provider == "openai":
        openai_sdk.api_base = os.getenv("OPENAI_API_BASE", openai_sdk.api_base)
        openai_sdk.api_key = os.getenv("OPENAI_API_KEY", openai_sdk.api_key)
    # else: leave defaults as-is

    # Simple RPM limiter (applied on misses only; safer for free tiers without slowing hits)
    interval = 60.0 / max(1.0, float(rpm))
    last_call = 0.0
    def _rate_limit():
        nonlocal last_call
        now = time.time()
        delta = now - last_call
        if delta < interval:
            time.sleep(interval - delta)
        last_call = time.time()

    # Warn if distance thresholds look off-scale
    if sim_eval_name.lower() == "distance" and any(float(t) > 1.0 for t in thresholds):
        print("WARN: --sim-eval distance with thresholds > 1.0 may be on the wrong scale.")

    # Ensure output directory for CSV exists and create run id
    out_dir = out_dir or "experiments/outputs"
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception:
        pass
    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    csv_path = os.path.join(out_dir, f"eval_cache_bank77_hf_{run_id}.csv")
    try:
        with open(os.path.join(out_dir, f"args_{run_id}.txt"), "w", encoding="utf-8") as f:
            f.write(" ".join(sys.argv))
    except Exception:
        pass

    for policy in policies:
        p = policy.lower()
        if p not in {"lru", "fifo"}:
            raise ValueError(f"GPTCache supports only LRU/FIFO eviction; got '{policy}'.")

        for th in thresholds:
            # Fresh cache per (policy, threshold) combo
            _init_gptcache_once(
                threshold=th,
                capacity=capacity,
                policy=p,
                embedder_name=embedder_name,
                sbert_model=sbert_model,
                sim_eval_name=sim_eval_name,
                sbert_xenc_model=sbert_xenc_model,
                onnx_model=onnx_model,
                cohere_model=cohere_model,
                cohere_api_key=cohere_api_key,
                db_path=db_path,
                index_path=index_path,
                top_k=topk,
                krecip_topk=krecip_topk,
                krecip_max_distance=krecip_max_distance,
                krecip_positive=krecip_positive,
                vector_backend=vector_backend,
            )

            # Post-init ping for Ollama connectivity (routes through GPTCache adapter)
            if provider == "ollama":
                try:
                    _ = cached_openai.ChatCompletion.create(
                        model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
                        messages=[{"role": "user", "content": "ping"}],
                    )
                except NotInitError:
                    print("Init error: call cache.init() before using gptcache.adapter.openai.")
                    raise
                except Exception as e:
                    print(f"ERROR: Ollama not reachable or model not pulled. Detail: {e}")
                    raise

            should_hit = 0
            hits = 0
            false_hit_lower_bound = 0
            llm_time_sum = 0.0
            llm_calls = 0
            cache_time_sum = 0.0

            # Correctness counters
            correct_total = 0
            correct_on_hits = 0
            correct_on_miss = 0
            n_hits_seen = 0
            n_miss_seen = 0

            seen_intents = set()

            # For percentile metrics
            cache_times: List[float] = []
            miss_times: List[float] = []

            for ex in stream:
                if ex.intent in seen_intents:
                    should_hit += 1

                t0 = time.time()
                try:
                    resp = cached_openai.ChatCompletion.create(
                        model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
                        messages=[{"role": "user", "content": ex.text}],
                        max_tokens=64,
                        temperature=0.0,
                        timeout=120,
                    )
                except Exception:
                    print("[gptcache] LLM/adapter exception:\n", "".join(traceback.format_exc()))
                    raise
                elapsed = time.time() - t0

                meta = resp.get("gptcache_meta", {}) or {}
                hit = bool(meta.get("hit", False))
                llm_time = float(meta.get("llm_time_s", 0.0))  # >0 when MISS (LLM called)
                total_time = float(meta.get("total_time_s", elapsed))

                # Miss-only rate limit to approximate provider quotas without slowing hits
                if not hit:
                    _rate_limit()

                # Extract answer text for correctness judging
                try:
                    answer_text = resp["choices"][0]["message"]["content"]
                except Exception:
                    answer_text = ""
                if not hit and not answer_text.strip():
                    print("WARN: MISS returned empty answer (provider likely failed). Check Ollama and model status.")

                # Track timing: count calls on misses; fallback when llm_time missing
                if not hit:
                    if llm_time > 0:
                        llm_calls += 1
                        llm_time_sum += llm_time
                        miss_times.append(llm_time)
                    else:
                        llm_calls += 1
                        fallback_time = max(0.0, elapsed)
                        llm_time_sum += fallback_time
                        miss_times.append(fallback_time)
                # Approximate cache lookup time
                cache_lookup_time = max(0.0, total_time - llm_time)
                cache_time_sum += cache_lookup_time
                cache_times.append(cache_lookup_time)

                if hit:
                    hits += 1
                    n_hits_seen += 1
                    # Lower bound: if we hit *before* this intent has ever appeared,
                    # the match must be cross-intent → count as false-hit LB.
                    if ex.intent not in seen_intents:
                        false_hit_lower_bound += 1
                else:
                    n_miss_seen += 1

                # Correctness bookkeeping
                if judge is not None:
                    gold = canonical.get(ex.intent, "")
                    ok = judge.is_correct(answer_text, gold) if gold else False
                    if ok:
                        correct_total += 1
                        if hit:
                            correct_on_hits += 1
                        else:
                            correct_on_miss += 1

                # Ground-truth "intent has been seen" (independent of cache)
                seen_intents.add(ex.intent)

            # ---- Summaries ----
            hit_recall = (hits / should_hit) if should_hit else 0.0
            hit_rate = (hits / total) if total else 0.0
            avg_llm_time = (llm_time_sum / max(1, llm_calls))
            avg_cache_time = (cache_time_sum / max(1, total))

            print("\n" + "-" * 68)
            print(f"POLICY={p.upper()} | THRESHOLD={th:.2f} | CAPACITY={capacity} | SIM_EVAL={sim_eval_name.upper()}")
            km_str = f"| KM={km_intents}x{km_per_intent}" if (km_intents and km_per_intent) else ""
            print(f"SPLIT={split} | N={total} | LIMIT={limit} {km_str}")
            print(f"Should-Hit={should_hit} | Hits={hits} | LBound False-Hit={false_hit_lower_bound}")
            print(f"Hit-Recall={hit_recall:.3f} | Hit-Rate={hit_rate:.3f}")
            print(f"LLM calls={llm_calls} | Avg LLM time={avg_llm_time:.3f}s | Avg cache time={avg_cache_time:.4f}s")
            if judge is not None:
                overall_acc = (correct_total / total) if total else 0.0
                acc_hits = (correct_on_hits / n_hits_seen) if n_hits_seen else 0.0
                acc_miss = (correct_on_miss / n_miss_seen) if n_miss_seen else 0.0
                print(f"Answer-Acc (overall)={overall_acc:.3f} | on HITs={acc_hits:.3f} | on MISSES={acc_miss:.3f}")
            print("-" * 68)

            # ---- CSV output per (policy, threshold) ----
            hit_precision = ((hits - false_hit_lower_bound) / hits) if hits else 0.0
            hit_precision = max(0.0, min(1.0, hit_precision))
            p95_cache_ms = float(np.percentile(cache_times, 95) * 1000.0) if cache_times else 0.0
            p95_miss_ms = float(np.percentile(miss_times, 95) * 1000.0) if miss_times else 0.0

            header = [
                "run_id",
                "embedder",
                "sbert_model",
                "split",
                "limit",
                "policy",
                "sim_eval",
                "topk",
                "tau",
                "should_hit",
                "hits",
                "false_hit_lb",
                "hit_recall",
                "hit_rate",
                "hit_precision_lb",
                "p95_cache_ms",
                "p95_miss_ms",
                "llm_calls",
                "avg_llm_time_s",
                "avg_cache_time_s",
            ]
            row = [
                run_id,
                embedder_name,
                sbert_model,
                split,
                int(limit),
                p.upper(),
                sim_eval_name,
                int(topk),
                float(th),
                int(should_hit),
                int(hits),
                int(false_hit_lower_bound),
                float(hit_recall),
                float(hit_rate),
                float(hit_precision),
                float(p95_cache_ms),
                float(p95_miss_ms),
                int(llm_calls),
                float(avg_llm_time),
                float(avg_cache_time),
            ]
            write_header = True
            if os.path.exists(csv_path):
                try:
                    write_header = os.path.getsize(csv_path) == 0
                except Exception:
                    write_header = False
            with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(header)
                writer.writerow(row)

            # Flush to avoid file descriptor buildup between sweeps
            cache.flush()


# =========================
# CLI
# =========================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate GPTCache on BANKING77 (LRU/FIFO, threshold sweeps).")
    p.add_argument("--split", type=str, default="train", choices=["train", "test"])
    p.add_argument("--limit", type=int, default=200, help="Subsample size (0 = full split).")
    p.add_argument("--thresholds", type=float, nargs="+", default=[],
                   help="Similarity thresholds. If omitted, sensible defaults per evaluator are used.")
    p.add_argument("--embedder", type=str, default="sbert", choices=["onnx", "sbert"],
                   help="Embedding backend for GPTCache vectors.")
    p.add_argument("--sbert-model", type=str, default="all-MiniLM-L6-v2",
                   help="Sentence-Transformers model (when --embedder sbert).")
    p.add_argument("--capacity", type=int, default=256, help="Max cache entries (DataManager.max_size).")
    p.add_argument("--policies", type=str, nargs="+", default=["lru", "fifo"],
                   help="Eviction strategies to compare (GPTCache supports LRU, FIFO).")
    p.add_argument("--provider", type=str, default="ollama", choices=["ollama", "openai"],
                   help="LLM provider behind the adapter (OpenAI-compatible).")
    p.add_argument("--rpm", type=float, default=12.0, help="Coarse per-process rate limit (calls/min).")
    p.add_argument("--sim-eval", type=str, default="cosine",
                   choices=["cosine", "exact", "np", "distance", "sbert", "onnx", "cohere", "krecip"],
                   help="Hit decision evaluator (normalized to [0,1] threshold in GPTCache).")
    p.add_argument("--topk", type=int, default=5, help="Top K for FAISS vector search.")
    p.add_argument("--sbert-xenc-model", type=str, default="cross-encoder/quora-distilroberta-base",
                   help="Cross-encoder for --sim-eval sbert.")
    p.add_argument("--onnx-model", type=str, default="GPTCache/albert-duplicate-onnx",
                   help="Model for --sim-eval onnx.")
    p.add_argument("--cohere-model", type=str, default="rerank-english-v2.0",
                   help="Cohere reranker for --sim-eval cohere.")
    p.add_argument("--cohere-api-key", type=str, default=None, help="Or set COHERE_API_KEY.")
    p.add_argument("--km-intents", type=int, default=0, help="If >0 with --km-per-intent, use K×M loader.")
    p.add_argument("--km-per-intent", type=int, default=0,
                   help="If >0 with --km-intents, M paraphrases per selected intent.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--db-path", type=str, default=None,
                   help="Path to SQLite file for scalar store (avoids default path issues).")
    p.add_argument("--index-path", type=str, default=None,
                   help="Path to FAISS index file (ensure dim matches chosen embedder).")
    p.add_argument("--answer-judge", type=str, default="sbert",
                   choices=["none", "sbert"],
                   help="Evaluate answer correctness vs canonical answers.")
    p.add_argument("--answer-threshold", type=float, default=0.60,
                   help="Cosine threshold for correctness when using --answer-judge sbert.")
    p.add_argument("--answer-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                   help="Sentence-Transformers model for answer judging.")
    p.add_argument("--out-dir", type=str, default="experiments/outputs", help="Directory to write CSV outputs.")
    p.add_argument("--krecip-topk", type=int, default=3, help="Top-K neighbors for K-Reciprocal check.")
    p.add_argument("--krecip-max-distance", type=float, default=4.0, help="Max distance bound for K-Reciprocal.")
    p.add_argument("--krecip-positive", action="store_true",
                   help="Set True if the underlying similarity score increases with similarity (rare for distance metrics).")
    p.add_argument("--vector-backend", type=str, default="faiss", choices=["faiss", "hnswlib"],
                   help="Vector backend to use for ANN search.")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # Per-evaluator default thresholds if none provided explicitly
    DEFAULT_TAU = {
        "cosine":   [0.6, 0.7, 0.8, 0.9],
        "np":       [0.6, 0.7, 0.8, 0.9],
        "sbert":    [0.5, 0.6, 0.7, 0.8],
        "onnx":     [0.5, 0.6, 0.7, 0.8],
        "cohere":   [0.4, 0.5, 0.6, 0.7],
        "distance": [0.1, 0.2, 0.3, 0.4],
        "krecip":   [0.6, 0.7, 0.8],
    }
    if not args.thresholds:
        args.thresholds = DEFAULT_TAU.get(args.sim_eval, args.thresholds)
    run_eval(
        split=args.split,
        limit=args.limit,
        thresholds=args.thresholds,
        embedder_name=args.embedder,
        sbert_model=args.sbert_model,
        capacity=args.capacity,
        policies=args.policies,
        provider=args.provider,
        rpm=args.rpm,
        km_intents=args.km_intents,
        km_per_intent=args.km_per_intent,
        sim_eval_name=args.sim_eval,
        sbert_xenc_model=args.sbert_xenc_model,
        onnx_model=args.onnx_model,
        cohere_model=args.cohere_model,
        cohere_api_key=args.cohere_api_key,
        seed=args.seed,
        answer_judge=args.answer_judge,
        answer_threshold=args.answer_threshold,
        answer_model=args.answer_model,
        db_path=args.db_path,
        index_path=args.index_path,
        topk=args.topk,
        out_dir=args.out_dir,
        krecip_topk=args.krecip_topk,
        krecip_max_distance=args.krecip_max_distance,
        krecip_positive=args.krecip_positive,
        vector_backend=args.vector_backend,
    )
