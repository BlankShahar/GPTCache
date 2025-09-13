"""
scripts/eval_cache_bank77_hf.py
----------------------------------------------------------------------

Purpose
  Evaluate LLM response caching on BANKING77 using intent–paraphrase queries.
  The script simulates an online request stream, measures cache quality/savings,
  and compares eviction and similarity strategies.

What this script does
  • Loads BANKING77 via 🤗 Datasets (or a K×M sampler that forces repeats).
  • Treats each intent as a class with a canonical (model-generated) answer.
  • For each query:
      – If a semantically similar request is in cache above a threshold → HIT.
      – Otherwise → MISS: call the chosen LLM provider, cache the answer.
  • Reports precision/recall of hits, false-hit rate, and latency savings.

Key knobs (CLI flags)
  --policies         Eviction policy: LRU / LFU (compare multiple in one run).
  --capacity         Max cache entries.
  --sim-eval         Similarity evaluator for hit decision:
                       - cosine      : cosine similarity on sentence embeddings
                       - exact       : GPTCache ExactMatchEvaluation
                       - np          : GPTCache NumpyNormEvaluation
                       - distance    : GPTCache SearchDistanceEvaluation
                       - sbert       : GPTCache SbertCrossencoderEvaluation
                       - onnx        : GPTCache OnnxModelEvaluation
                       - cohere      : GPTCache CohereRerankEvaluation
                     All evaluators are normalized to [0,1] so thresholds are comparable.
  --thresholds       One or more thresholds (e.g., 0.65 0.75 0.85).
  --embedder         Sentence-Transformers model for embedding-based evaluators
                     (default: sentence-transformers/all-MiniLM-L6-v2).
  --provider         Backend used on MISSes:
                       sim | gemini | hf | ollama
                     sim     = sleep-only (no external calls)
                     gemini  = Google Gemini via google-generativeai
                     hf      = Hugging Face Inference API (serverless)
                     ollama  = Local OpenAI-compatible server (http://localhost:11434)
  --rpm              Simple per-process rate limiter for API providers (calls/min).
  --km-intents / --km-per-intent
                     Use a controlled K×M sampler (K intents × M paraphrases each)
                     to guarantee repeat queries (i.e., realistic hit opportunities).
  --seed             RNG seed for reproducible sampling/shuffle.

Printed metrics
  Precision        = correct_hits / hits
  Hit-Recall       = hits / should_hit
  Correct-Recall   = correct_hits / should_hit
  False-Hit-Rate   = (hits - correct_hits) / should_hit
  Avg LLM time     = mean latency over actual LLM calls (misses + false hits)
  Avg cache time   = mean latency of cache lookups per query

How thresholds work
  • cosine: directly the cosine score in [0,1].
  • exact / np / distance / sbert / onnx / cohere: raw evaluator scores are
    mapped to [0,1] using the evaluator’s reported range() so a single threshold
    value is comparable across evaluators.

Quick examples
  # 1) Fast baseline: cosine + Ollama, compare LRU vs LFU
  python scripts/eval_cache_bank77_hf.py \
    --provider ollama --policies lru lfu --capacity 64 --thresholds 0.65 0.75 \
    --km-intents 8 --km-per-intent 5

  # 2) SBERT cross-encoder similarity (heavier but often stronger)
  python scripts/eval_cache_bank77_hf.py \
    --sim-eval sbert --sbert-model cross-encoder/quora-distilroberta-base \
    --provider sim --thresholds 0.75 --limit 200

  # 3) ONNX similarity model (offline)
  python scripts/eval_cache_bank77_hf.py \
    --sim-eval onnx --onnx-model GPTCache/albert-duplicate-onnx \
    --provider sim --thresholds 0.7

  # 4) Hugging Face Inference API (set HF_TOKEN)
  python scripts/eval_cache_bank77_hf.py \
    --provider hf --rpm 10 --thresholds 0.7 --limit 50

  # 5) Gemini free tier (set GOOGLE_API_KEY) with rate limiting
  python scripts/eval_cache_bank77_hf.py \
    --provider gemini --rpm 8 --thresholds 0.7 --km-intents 6 --km-per-intent 4

Environment variables (.env supported)
  # Providers
  GOOGLE_API_KEY=<your_gemini_key>
  HF_TOKEN=<your_huggingface_inference_token>
  HF_MODEL=HuggingFaceH4/zephyr-7b-beta            # optional override
  HF_API_URL=https://api-inference.huggingface.co/models/${HF_MODEL}
  COHERE_API_KEY=<optional_if_using_cohere_evaluator>

  # Ollama (local) – no key needed; ensure the server is running:
  #   ollama serve
  # And that a chat model is pulled, e.g.:
  #   ollama pull llama3.1:8b

Dependencies
  pip install -U \
    datasets sentence-transformers scikit-learn torch \
    gptcache python-dotenv
  # Optional depending on usage:
  pip install google-generativeai requests cohere onnxruntime

Notes & tips
  • SBERT/ONNX/Cohere evaluators are slower than cosine; use them for
    quality comparisons rather than throughput baselines.
  • Cohere rerank calls an external API per candidate → keep cache small
    or expect high latency and rate limits.
  • Free-tier providers have strict RPM/DPQ. Use --rpm to avoid 429s.
  • For Windows + conda, ensure a CPU-compatible torch wheel is installed.

"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


from dataclasses import dataclass
from typing import Dict, List, Tuple
from typing import Any, Optional
import argparse
import random
import time
import numpy as np
import os

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# --- silence noisy deprecation warnings from transformers/torch ---
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Reduce Hugging Face transformers log verbosity (errors only)
try:
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
except Exception:
    pass

from providers import get_provider


# ----------------------------
# Simple per-process rate limiter
# ----------------------------
class RateLimiter:
    """Simple per-process rate limiter: at most `rpm` calls/minute."""
    def __init__(self, rpm: float):
        self.interval = 60.0 / max(1.0, float(rpm))
        self.last = 0.0

    def wait(self):
        now = time.time()
        delta = now - self.last
        if delta < self.interval:
            time.sleep(self.interval - delta)
        self.last = time.time()


# ----------------------------
# GPTCache similarity evaluator helpers
# ----------------------------
try:
    from gptcache.similarity_evaluation import (
        ExactMatchEvaluation,
        NumpyNormEvaluation,
        SearchDistanceEvaluation,
        SbertCrossencoderEvaluation,
        OnnxModelEvaluation,
        CohereRerankEvaluation,
    )
except Exception:
    ExactMatchEvaluation = NumpyNormEvaluation = SearchDistanceEvaluation = None
    SbertCrossencoderEvaluation = OnnxModelEvaluation = CohereRerankEvaluation = None


def _build_gptcache_evaluator(
    name: str,
    *,
    sbert_model: str = "cross-encoder/quora-distilroberta-base",
    onnx_model: str = "GPTCache/albert-duplicate-onnx",
    cohere_model: str = "rerank-english-v2.0",
    cohere_api_key: Optional[str] = None,
    distance_max: float = 2.0,
    distance_positive: bool = False,
):
    """
    Create a GPTCache SimilarityEvaluation by name.
    Returns (evaluator, (min_score, max_score)).
    """
    name = name.lower()
    if name == "exact":
        from gptcache.similarity_evaluation import ExactMatchEvaluation as _Exact
        ev = _Exact()
    elif name == "np":
        from gptcache.similarity_evaluation import NumpyNormEvaluation as _Np
        ev = _Np(enable_normal=True)
    elif name == "distance":
        from gptcache.similarity_evaluation import SearchDistanceEvaluation as _Dist
        ev = _Dist(max_distance=distance_max, positive=distance_positive)
    elif name == "sbert":
        from gptcache.similarity_evaluation import SbertCrossencoderEvaluation as _Sbert
        ev = _Sbert(model=sbert_model)
    elif name == "onnx":
        from gptcache.similarity_evaluation import OnnxModelEvaluation as _Onnx
        ev = _Onnx(model=onnx_model)
    elif name == "cohere":
        if not cohere_api_key:
            raise ValueError(
                "Cohere reranker selected but no API key provided. "
                "Pass --cohere-api-key or set COHERE_API_KEY."
            )
        from gptcache.similarity_evaluation import CohereRerankEvaluation as _Cohere
        ev = _Cohere(model=cohere_model, api_key=cohere_api_key)
    elif name == "cosine":
        return None, (0.0, 1.0)
    else:
        raise ValueError(f"Unknown --sim-eval '{name}'. Choose from cosine|exact|np|distance|sbert|onnx|cohere")

    try:
        rmin, rmax = ev.range()
    except Exception:
        rmin, rmax = (0.0, 1.0)
    return ev, (float(rmin), float(rmax))


def _normalize_score(score: float, rmin: float, rmax: float) -> float:
    """Map raw evaluator score to [0,1] for consistent thresholds."""
    if rmax <= rmin:
        return 0.0
    x = (score - rmin) / (rmax - rmin)
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else float(x)


# ----------------------------
# VectorQ per-entry stats
# ----------------------------
class VQStats:
    def __init__(self, bins: int = 20, alpha: float = 1.0, beta: float = 1.0):
        self.bins = int(bins)
        self.pos = np.zeros(self.bins, dtype=np.int32)
        self.tot = np.zeros(self.bins, dtype=np.int32)
        self.alpha = float(alpha)
        self.beta = float(beta)

    def bin_idx(self, s: float) -> int:
        i = int(min(self.bins - 1, max(0, s * self.bins)))
        return i

    def update(self, s: float, correct: int) -> None:
        i = self.bin_idx(s)
        self.tot[i] += 1
        if correct:
            self.pos[i] += 1

    def est_p(self, s: float) -> float:
        i = self.bin_idx(s)
        return (self.pos[i] + self.alpha) / (self.tot[i] + self.alpha + self.beta)

    def confident(self, s: float, min_count: int = 5, target: float = 0.9) -> Optional[bool]:
        i = self.bin_idx(s)
        if self.tot[i] < int(min_count):
            return None
        return bool(self.est_p(s) >= float(target))


# ----------------------------
# Data model
# ----------------------------
@dataclass
class Example:
    text: str
    intent: str


def load_banking77(split: str = "train", limit: int = 0, seed: int = 42) -> List[Example]:
    """
    Load BANKING77 in a way compatible with datasets>=4:
      1) Preferred: Parquet mirror `mteb/banking77` (columns: 'text', 'label', 'label_text').
      2) Fallback: try `PolyAI/banking77` on a parquet conversion branch if present.

    Args:
      split: "train" or "test"
      limit: if > 0, subsample to N examples
      seed:  RNG seed for reproducible subsampling/shuffle
    """

    def _subsample_and_shuffle(rows: List[Example]) -> List[Example]:
        if limit and limit < len(rows):
            rnd = random.Random(seed)
            rows = rnd.sample(rows, limit)
        random.Random(seed).shuffle(rows)
        return rows

    # 1) Preferred: mteb/banking77 (Parquet)
    try:
        ds = load_dataset("mteb/banking77", split=split)
        if "label_text" in ds.column_names:
            rows: List[Example] = [Example(text=rec["text"], intent=rec["label_text"]) for rec in ds]
        else:
            label_names = ds.features["label"].names
            rows = [Example(text=rec["text"], intent=label_names[int(rec["label"])]) for rec in ds]
        return _subsample_and_shuffle(rows)
    except Exception as e1:
        # 2) Fallback: try parquet conversion branch of PolyAI/banking77 if available
        try:
            ds = load_dataset("PolyAI/banking77", split=split, revision="refs/convert/parquet")
            if "label_text" in ds.column_names:
                rows = [Example(text=rec["text"], intent=rec["label_text"]) for rec in ds]
            else:
                label_names = ds.features["label"].names
                rows = [Example(text=rec["text"], intent=label_names[int(rec["label"])]) for rec in ds]
            return _subsample_and_shuffle(rows)
        except Exception as e2:
            raise RuntimeError(
                "Unable to load BANKING77 with datasets>=4. "
                "Use the mteb/banking77 mirror (preferred) or pin datasets<4.0. "
                f"Errors: mteb/banking77 -> {e1} | PolyAI/banking77@refs/convert/parquet -> {e2}"
            )


# ----------------------------
# K×M sampler: force repeats per intent
# ----------------------------
def load_banking77_k_per_intent(
    split: str = "train",
    num_intents: int = 10,
    per_intent: int = 4,
    seed: int = 42,
) -> List[Example]:
    """
    Build a stream containing `num_intents` intents and `per_intent` paraphrases each,
    then shuffle. Ensures repeats so the cache has real hit opportunities.
    """
    ds = load_dataset("mteb/banking77", split=split)

    # Choose label field
    if "label_text" in ds.column_names:
        def get_intent(rec):
            return rec["label_text"]
    else:
        names = ds.features["label"].names
        def get_intent(rec):
            return names[int(rec["label"])]

    by_intent: Dict[str, List[str]] = {}
    for rec in ds:
        it = get_intent(rec)
        by_intent.setdefault(it, []).append(rec["text"])

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

# ----------------------------
# Canonical answers per intent
# ----------------------------
def build_canonical_answers(intents: List[str]) -> Dict[str, str]:
    """
    Create a canonical answer template per intent. This keeps the script
    self-contained; you can replace with a curated mapping if you prefer.

    Example:
      "balance" -> "Intent=balance: Here's how to handle that …"
    """
    canon: Dict[str, str] = {}
    for it in intents:
        canon[it] = (
            f"[Canonical answer for intent='{it}'] "
            f"This is a template response describing how we usually resolve '{it}'."
        )
    return canon


# ----------------------------
# Capacity-bounded vector cache with eviction policies
# ----------------------------
class EvictingVectorCache:
    """
    A small, self-contained vector cache with LRU / LFU eviction.
    - Each entry stores: embedding vector, (intent, answer), freq, last_access, created_at.
    - Query performs cosine similarity against all stored vectors (small-scale friendly).
    - Insert computes the embedding and evicts one entry if capacity is exceeded.

    Eviction:
      • LRU: evict the item with the smallest last_access (oldest use).
      • LFU: evict the item with the smallest frequency; ties break by oldest last_access.
    """
    def __init__(self, embedder_name="sentence-transformers/all-MiniLM-L6-v2",
                 capacity: int = 256, policy: str = "lru",
                 sim_eval_name: str = "cosine", eval_kwargs: Optional[Dict[str, Any]] = None, vq_bins: int = 20):
        # Prefer GPU if available (big speedup); otherwise CPU.
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

        self.model = SentenceTransformer(embedder_name, device=device)
        self.capacity = max(1, int(capacity))
        self.policy = policy.lower()
        assert self.policy in {"lru", "lfu", "fifo"}, "policy must be 'lru' or 'lfu' or 'fifo'"

        # evaluator setup
        eval_kwargs = eval_kwargs or {}
        self.sim_eval_name = sim_eval_name.lower()
        self.sim_eval, self.eval_range = _build_gptcache_evaluator(self.sim_eval_name, **eval_kwargs)

        # storage
        self.vecs: List[np.ndarray] = []
        self.texts: List[str] = []                # store original prompt
        self.payloads: List[Tuple[str, str]] = []   # (intent, answer)
        self.freq: List[int] = []
        self.last_access: List[int] = []
        self.created_at: List[int] = []
        # VectorQ per-entry stats
        self.vq_bins = int(vq_bins)
        self.vq: List[VQStats] = []
        self._clock = 0  # monotonically increasing "tick" for recency

    # ---- internal helpers ----
    def _tick(self) -> int:
        self._clock += 1
        return self._clock

    def _evict_index(self) -> int:
        """Return the index to evict based on policy."""
        if self.policy == "lru":
            # Evict oldest last_access
            return int(np.argmin(self.last_access))
        elif self.policy == "lfu":
            # LFU: min frequency; break ties by oldest last_access
            min_freq = min(self.freq)
            candidates = [i for i, f in enumerate(self.freq) if f == min_freq]
            if len(candidates) == 1:
                return candidates[0]
            # tie-break by LRU among the least-frequent
            la = np.array([self.last_access[i] for i in candidates])
            return candidates[int(np.argmin(la))]
        elif self.policy == "fifo":
            # Oldest creation time → evict first
            return int(np.argmin(self.created_at))
        else:
            raise ValueError(f"unknown policy: {self.policy}")

    def _replace_at(self, idx: int, v: np.ndarray, intent: str, answer: str, text: str) -> None:
        self.vecs[idx] = v
        self.texts[idx] = text
        self.payloads[idx] = (intent, answer)
        self.freq[idx] = 1
        now = self._tick()
        self.last_access[idx] = now
        self.created_at[idx] = now

    # ---- public API ----
    def embed(self, text: str) -> np.ndarray:
        return self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

    def query(self, q: str, threshold: float, vq_mode: str = "off", vq_target: float = 0.9, vq_min_count: int = 5, vq_sample: float = 0.3, rng: Optional[random.Random] = None) -> Tuple[bool, str, str, int, float]:
        """
        Return (is_hit, predicted_intent, answer). On hit, update recency/frequency.
        """
        if not self.vecs:
            return False, "", "", -1, 0.0
        now = self._tick()
        # Fast-path: cosine similarity using embeddings
        if self.sim_eval_name == "cosine":
            qv = self.embed(q).reshape(1, -1)
            sims = cosine_similarity(qv, np.vstack(self.vecs))[0]  # (N,)
            j = int(np.argmax(sims))
            # Normalize cosine from [-1,1] to [0,1] for consistent thresholds
            s = float(sims[j])
            s01 = (s + 1.0) / 2.0
            # VectorQ gating if enabled
            if vq_mode == "per-entry":
                rng = rng or random.Random()
                verdict = self.vq[j].confident(s01, min_count=vq_min_count, target=vq_target)
                if verdict is True:
                    self.last_access[j] = now
                    self.freq[j] += 1
                    intent, ans = self.payloads[j]
                    return True, intent, ans, j, s01
                elif verdict is None:
                    # probe with probability, else fall back to static threshold
                    if rng.random() < float(vq_sample):
                        return False, "", "", j, s01
                    # static threshold fallback
                    if s01 >= threshold:
                        self.last_access[j] = now
                        self.freq[j] += 1
                        intent, ans = self.payloads[j]
                        return True, intent, ans, j, s01
                    return False, "", "", j, s01
                else:
                    # verdict is False → MISS
                    return False, "", "", j, s01
            # Static threshold path
            if s01 >= threshold:
                self.last_access[j] = now
                self.freq[j] += 1
                intent, ans = self.payloads[j]
                return True, intent, ans, j, s01
            return False, "", "", j, s01

        # GPTCache evaluator path: compute normalized [0,1] score
        best_idx = -1
        best_score = -1.0

        qv = None
        if self.sim_eval_name in ("np", "distance"):
            qv = self.embed(q).reshape(-1)

        rmin, rmax = self.eval_range

        for i in range(len(self.vecs)):
            if self.sim_eval_name == "exact":
                src = {"question": q}
                cand = {"question": self.texts[i]}
            elif self.sim_eval_name == "sbert":
                src = {"question": q}
                cand = {"question": self.texts[i]}
            elif self.sim_eval_name == "onnx":
                src = {"question": q}
                cand = {"question": self.texts[i]}
            elif self.sim_eval_name == "cohere":
                src = {"question": q}
                cand = {"answer": self.payloads[i][1]}
            elif self.sim_eval_name == "np":
                src = {"question": q, "embedding": qv}
                cand = {"question": self.texts[i], "embedding": self.vecs[i]}
            elif self.sim_eval_name == "distance":
                if qv is None:
                    qv = self.embed(q).reshape(1, -1)
                sim = float(cosine_similarity(qv.reshape(1, -1), self.vecs[i].reshape(1, -1))[0, 0])
                dist = 1.0 - sim
                src = {}
                cand = {"search_result": (dist, None)}
            else:
                raise RuntimeError(f"Unhandled sim-eval: {self.sim_eval_name}")

            score = float(self.sim_eval.evaluation(src, cand))
            score01 = _normalize_score(score, rmin, rmax)
            if score01 > best_score:
                best_score = score01
                best_idx = i

        if best_idx < 0:
            return False, "", "", -1, 0.0
        s = float(best_score)
        # VectorQ gating if enabled
        if vq_mode == "per-entry":
            rng = rng or random.Random()
            verdict = self.vq[best_idx].confident(s, min_count=vq_min_count, target=vq_target)
            if verdict is True:
                self.last_access[best_idx] = now
                self.freq[best_idx] += 1
                intent, ans = self.payloads[best_idx]
                return True, intent, ans, best_idx, s
            elif verdict is None:
                if rng.random() < float(vq_sample):
                    return False, "", "", best_idx, s
                if s >= threshold:
                    self.last_access[best_idx] = now
                    self.freq[best_idx] += 1
                    intent, ans = self.payloads[best_idx]
                    return True, intent, ans, best_idx, s
                return False, "", "", best_idx, s
            else:
                return False, "", "", best_idx, s
        # Static threshold path
        if s >= threshold:
            self.last_access[best_idx] = now
            self.freq[best_idx] += 1
            intent, ans = self.payloads[best_idx]
            return True, intent, ans, best_idx, s
        return False, "", "", best_idx, s

    def update_vq(self, matched_index: int, score01: float, was_correct: bool) -> None:
        if 0 <= matched_index < len(self.vq):
            self.vq[matched_index].update(score01, int(was_correct))

    def insert(self, text: str, intent: str, answer: str) -> None:
        """
        Insert (text -> embedding) with its payload. If capacity is full, evict one.
        New entries start with freq=1 and recency=now.
        """
        v = self.embed(text)
        now = self._tick()
        if len(self.vecs) < self.capacity:
            self.vecs.append(v)
            self.texts.append(text)
            self.payloads.append((intent, answer))
            self.freq.append(1)
            self.last_access.append(now)
            self.created_at.append(now)
            self.vq.append(VQStats(bins=self.vq_bins))
        else:
            idx = self._evict_index()
            self._replace_at(idx, v, intent, answer, text)
            # reset VQ stats on replacement
            if 0 <= idx < len(self.vq):
                self.vq[idx] = VQStats(bins=self.vq_bins)


# ----------------------------
# Experiment loop
# ----------------------------
def run_eval(
    split: str = "train",
    limit: int = 200,
    thresholds: List[float] = [0.65, 0.75, 0.85],
    embedder: str = "sentence-transformers/all-MiniLM-L6-v2",
    llm_latency_s: float = 0.05,
    capacity: int = 256,
    policies: List[str] = ("lru", "lfu"),
    provider_name: str = "sim",
    rpm: float = 12.0,
    km_intents: int = 0,
    km_per_intent: int = 0,
    sim_eval_name: str = "cosine",
    eval_kwargs: Optional[Dict[str, Any]] = None,
    vq_mode: str = "off",
    vq_target: float = 0.9,
    vq_bins: int = 20,
    vq_min_count: int = 5,
    vq_sample: float = 0.3,
    seed: int = 42,
) -> None:
    """
    Execute the cache simulation over a BANKING77 split.

    Metrics:
      - Precision: correct_hits / hits
      - Recall:    hits / should_hit
      - Timing:    avg LLM time (simulated) vs avg cache time
    """
    if km_intents and km_per_intent:
        data = load_banking77_k_per_intent(
            split=split, num_intents=km_intents, per_intent=km_per_intent, seed=seed
        )
    else:
        data = load_banking77(split=split, limit=limit, seed=seed)
        all_intents = sorted({ex.intent for ex in data})
        canonical = build_canonical_answers(all_intents)

    # Build provider once
    llm = get_provider(provider=provider_name, temperature=0.0, sleep_s=llm_latency_s)

    for policy in policies:
        policy = policy.lower()
        assert policy in {"lru", "lfu", "fifo"}, "policy must be 'lru' or 'lfu' or 'fifo'"
        rng = random.Random(seed)
        for th in thresholds:
            cache = EvictingVectorCache(
                embedder_name=embedder,
                capacity=capacity,
                policy=policy,
                sim_eval_name=sim_eval_name,
                eval_kwargs=eval_kwargs,
                vq_bins=vq_bins,
            )

            total = len(data)
            should_hit = 0        # #queries whose intent has been seen before (ground-truth)
            hits = 0              # #queries served from cache (true+false)
            correct_hits = 0      # #hits whose intent matches gold
            llm_time = 0.0
            llm_calls = 0
            cache_time = 0.0
            seen_truth = set()
            rl = RateLimiter(rpm)

            for ex in data:
                # 1) Count should-hit strictly by ground-truth history (not cache ops)
                if ex.intent in seen_truth:
                    should_hit += 1

                # 2) Query the cache
                t0 = time.time()
                is_hit, pred_intent, _, j, score01 = cache.query(
                    ex.text, threshold=th,
                    vq_mode=vq_mode, vq_target=vq_target,
                    vq_min_count=vq_min_count, vq_sample=vq_sample,
                    rng=rng,
                )
                cache_time += time.time() - t0

                if is_hit and pred_intent == ex.intent:
                    # True hit
                    hits += 1
                    correct_hits += 1
                    if j >= 0:
                        cache.update_vq(j, score01, True)
                elif is_hit and pred_intent != ex.intent:
                    # False hit: count it, but treat as miss for update so cache learns
                    hits += 1
                    if provider_name in ("gemini", "hf"):
                        rl.wait()
                    ans_text, dt = llm.chat(ex.text)
                    llm_time += dt
                    llm_calls += 1
                    if j >= 0:
                        cache.update_vq(j, score01, False)
                    cache.insert(ex.text, ex.intent, ans_text)
                else:
                    # Miss
                    if provider_name in ("gemini", "hf"):
                        rl.wait()
                    # teach VectorQ about the candidate you almost accepted
                    if j >= 0:
                        was_correct = (cache.payloads[j][0] == ex.intent)
                        cache.update_vq(j, score01, was_correct)
                    ans_text, dt = llm.chat(ex.text)
                    llm_time += dt
                    llm_calls += 1
                    cache.insert(ex.text, ex.intent, ans_text)

                # 3) Mark this intent as seen in the ground-truth sense
                seen_truth.add(ex.intent)

            precision = (correct_hits / hits) if hits else 0.0
            hit_recall = (hits / should_hit) if should_hit else 0.0
            correct_recall = (correct_hits / should_hit) if should_hit else 0.0
            false_hit_rate = ((hits - correct_hits) / should_hit) if should_hit else 0.0
            avg_llm = (llm_time / max(1, llm_calls))
            avg_cache = (cache_time / total)

            print("\n" + "-" * 66)
            mode_str = (
                f"VECTORQ(target={vq_target:.2f})" if vq_mode == "per-entry" else "STATIC"
            )
            print(f"POLICY={policy.upper()} | THRESHOLD={th:.2f} | CAPACITY={capacity} | SIM_EVAL={sim_eval_name.upper()} | MODE={mode_str}")
            print(f"SPLIT={split} | N={total} | LIMIT={limit}")
            print(f"Should-Hit={should_hit} | Hits={hits} | Correct-Hits={correct_hits}")
            print(
                f"Precision={precision:.3f} | Hit-Recall={hit_recall:.3f} | "
                f"Correct-Recall={correct_recall:.3f} | False-Hit-Rate={false_hit_rate:.3f}"
            )
            hit_rate = hits / total if total else 0.0
            print(f"Hit-Rate={hit_rate:.3f} | LLM calls={llm_calls} | Avg LLM time={avg_llm:.3f}s | Avg cache time={avg_cache:.4f}s")
            print("-" * 66)


# ----------------------------
# CLI
# ----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate cache behavior on BANKING77 via 🤗 Datasets, with LRU/LFU policies.")
    p.add_argument("--split", type=str, default="train", choices=["train", "test"],
                   help="Dataset split to evaluate.")
    p.add_argument("--limit", type=int, default=200,
                   help="Randomly subsample this many examples (0 = use full split).")
    p.add_argument("--thresholds", type=float, nargs="+", default=[0.65, 0.75, 0.85],
                   help="One or more cosine similarity thresholds to sweep.")
    p.add_argument("--embedder", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                   help="Sentence-Transformers model name for embeddings.")
    p.add_argument("--llm-latency-s", type=float, default=0.05,
                   help="Simulated LLM latency (seconds) for misses).")
    p.add_argument("--capacity", type=int, default=256,
                   help="Cache capacity (number of entries).")
    p.add_argument("--policies", type=str, nargs="+", default=["lru", "lfu"],
                   help="Eviction policies to evaluate, e.g., lru lfu fifo")
    p.add_argument("--provider", type=str, default="sim", choices=["sim", "gemini", "hf", "ollama"],
                   help="Which backend to use for MISSes: sim|gemini|hf|ollama")
    p.add_argument("--rpm", type=float, default=12.0,
                   help="Max requests per minute to the provider (Gemini free tier is ~15).")
    p.add_argument("--sim-eval", type=str, default="cosine",
                   choices=["cosine", "exact", "np", "distance", "sbert", "onnx", "cohere"],
                   help="Similarity evaluator to use for hit decision.")
    p.add_argument("--sbert-model", type=str, default="cross-encoder/quora-distilroberta-base",
                   help="Model for --sim-eval sbert")
    p.add_argument("--onnx-model", type=str, default="GPTCache/albert-duplicate-onnx",
                   help="Model for --sim-eval onnx")
    p.add_argument("--cohere-model", type=str, default="rerank-english-v2.0",
                   help="Model for --sim-eval cohere")
    p.add_argument("--cohere-api-key", type=str, default=None,
                   help="API key for Cohere (or set COHERE_API_KEY)")
    p.add_argument("--distance-max", type=float, default=2.0,
                   help="max_distance for --sim-eval distance (cosine distance in [0,2])")
    p.add_argument("--distance-positive", action="store_true",
                   help="If set for --sim-eval distance, larger distance = more similar (usually False).")
    p.add_argument("--vq-mode", type=str, default="off", choices=["off", "per-entry"],
                   help="Adaptive acceptance policy (VectorQ): 'off' or 'per-entry'.")
    p.add_argument("--vq-target", type=float, default=0.9,
                   help="Target acceptance precision for VectorQ (acts like threshold on estimated correctness).")
    p.add_argument("--vq-bins", type=int, default=20,
                   help="Histogram bins for VectorQ similarity score in [0,1].")
    p.add_argument("--vq-min-count", type=int, default=5,
                   help="Minimum samples per bin to consider confident in VectorQ.")
    p.add_argument("--vq-sample", type=float, default=0.3,
                   help="Probe probability in VectorQ when confidence is unknown.")
    p.add_argument("--km-intents", type=int, default=0,
                   help="If >0 with --km-per-intent, switch to K×M loader with this many intents.")
    p.add_argument("--km-per-intent", type=int, default=0,
                   help="If >0 with --km-intents, number of paraphrases per selected intent.")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for sampling/shuffle.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    eval_kwargs = {
        "sbert_model": args.sbert_model,
        "onnx_model": args.onnx_model,
        "cohere_model": args.cohere_model,
        "cohere_api_key": args.cohere_api_key or os.getenv("COHERE_API_KEY"),
        "distance_max": args.distance_max,
        "distance_positive": bool(args.distance_positive),
    }
    run_eval(
        split=args.split,
        limit=args.limit,
        thresholds=args.thresholds,
        embedder=args.embedder,
        llm_latency_s=args.llm_latency_s,
        capacity=args.capacity,
        policies=args.policies,
        provider_name=args.provider,
        rpm=args.rpm,
        km_intents=args.km_intents,
        km_per_intent=args.km_per_intent,
        sim_eval_name=args.sim_eval,
        eval_kwargs=eval_kwargs,
        vq_mode=args.vq_mode,
        vq_target=args.vq_target,
        vq_bins=args.vq_bins,
        vq_min_count=args.vq_min_count,
        vq_sample=args.vq_sample,
        seed=args.seed,
    )
