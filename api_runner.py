import argparse
from pathlib import Path
from typing import List, Optional

import openai
from pydantic import BaseModel

from gptcache import cache
from gptcache.adapter import openai as cached_openai
from gptcache.manager import get_data_manager, CacheBase, VectorBase
from gptcache.similarity_evaluation import SearchDistanceEvaluation


# -------------------- argparse for POLICY --------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--policy",
    default="LFU",
    choices=["LFU", "LRU", "FIFO", "RR", "LRFU", "LRU2"],
    help="Eviction policy for GPTCache (default: LFU)",
)
args = parser.parse_args()
POLICY = args.policy
# -------------------------------------------------------------


# Set Ollama as LLM
openai.api_base = "http://localhost:11434/v1"
openai.api_key = "ollama"


# Embedding function: accept **kwargs and return a single vector for a single str
def ollama_embed(text, **kwargs):
    if isinstance(text, str):
        resp = openai.Embedding.create(model="nomic-embed-text", input=[text])
        return resp["data"][0]["embedding"]
    resp = openai.Embedding.create(model="nomic-embed-text", input=list(text))
    return [d["embedding"] for d in resp["data"]]


# Find embedding dimension once for FAISS
_dim = len(ollama_embed("dimension_probe"))

data_manager = get_data_manager(
    CacheBase("sqlite"),
    VectorBase("faiss", dimension=_dim),
    eviction=POLICY,
    max_size=256,
    clean_size=1,
)
cache.init(
    embedding_func=ollama_embed,
    data_manager=data_manager,
    similarity_evaluation=SearchDistanceEvaluation(),
)


class Answer(BaseModel):
    response: str
    original_response: Optional[str] = None
    is_hit: bool
    llm_time: float  # seconds
    total_time: float  # seconds


class Result(BaseModel):
    answers: List[Answer]


def ask(prompt: str) -> Answer:
    result = cached_openai.ChatCompletion.create(
        model="llama3.2:1b",
        messages=[{"role": "user", "content": prompt}],
    )
    response = result["choices"][0]["message"]["content"]
    result_metadata = result.get("gptcache_meta", {})
    return Answer(
        response=response,
        is_hit=result_metadata.get("hit"),
        llm_time=result_metadata.get("llm_time_s"),
        total_time=result_metadata.get("total_time_s"),
    )


from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm


def run_experiment():
    ds = load_dataset("OpenAssistant/oasst1")
    train = ds["train"]
    val = ds["validation"]

    full_ds = concatenate_datasets([train, val])
    sorted_ds = full_ds.sort("created_date")
    sorted_ds = sorted_ds.select(range(10))  # <-- stays a Dataset (not dict!)

    answers = {}  # message_id -> Answer

    for record in tqdm(sorted_ds):
        if record["role"] == "prompter":
            answer = ask(record["text"])
            answers[record["message_id"]] = answer
        else:  # assistant (LLM)
            parent_id = record["parent_id"]
            if parent_id in answers:
                answers[parent_id].original_response = record["text"]

    result = Result(answers=list(answers.values()))
    Path(f"experiment_result-{POLICY}.json").write_text(
        result.model_dump_json(indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    run_experiment()
