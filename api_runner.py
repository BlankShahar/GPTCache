from pathlib import Path
from typing import List

import openai
from pydantic import BaseModel

from gptcache import cache
from gptcache.adapter import openai as cached_openai
from gptcache.manager import get_data_manager, CacheBase, VectorBase
from gptcache.similarity_evaluation import SearchDistanceEvaluation

# Set Ollama as LLM
openai.api_base = "http://localhost:11434/v1"
openai.api_key = "ollama"


# Embedding function: accept **kwargs and return a single vector for a single str
def ollama_embed(text, **kwargs):
    # GPTCache will pass a single string after pre_embedding_func
    if isinstance(text, str):
        resp = openai.Embedding.create(model="nomic-embed-text", input=[text])
        return resp["data"][0]["embedding"]
    # (defensive) if a list is ever passed, return list of vectors
    resp = openai.Embedding.create(model="nomic-embed-text", input=list(text))
    return [d["embedding"] for d in resp["data"]]


# Find embedding dimension once for FAISS
_dim = len(ollama_embed("dimension_probe"))

data_manager = get_data_manager(
    CacheBase("sqlite"),
    VectorBase("faiss", dimension=_dim),
    eviction="AP",
    max_size=2,
    clean_size=1
)
cache.init(
    embedding_func=ollama_embed,
    data_manager=data_manager,
    similarity_evaluation=SearchDistanceEvaluation(),
)


class Answer(BaseModel):
    response: str
    original_response: str | None = None
    is_hit: bool
    llm_time: float  # seconds
    total_time: float  # seconds


class Result(BaseModel):
    answers: List[Answer]


def ask(prompt: str) -> Answer:
    result = cached_openai.ChatCompletion.create(
        model="llama3.1:8b",
        messages=[{"role": "user", "content": prompt}],
    )
    response = result["choices"][0]["message"]["content"]
    result_metadata = result.get("gptcache_meta", {})
    return Answer(
        response=response,
        is_hit=result_metadata.get('hit'),
        llm_time=result_metadata.get('llm_time_s'),
        total_time=result_metadata.get('total_time_s')
    )


def run_example():
    # 1) First prompt -> MISS: llm_time_s > 0
    print("1")
    ask("Write me a short script of calculator in python")
    print('-------------')
    # 2) Similar prompt -> HIT: llm_time_s == 0
    print("2")
    ask("Make a simple calculator in python")
    print("3")
    ask("Hi")
    print("4")
    ask("Best footballer in the world?")


from datasets import load_dataset, concatenate_datasets
from datetime import datetime
from tqdm import tqdm


def run_experiment():
    # --- Load from HF Hub OR parquet -----------------------------------------
    # Option A: HF Hub
    ds = load_dataset("OpenAssistant/oasst1")
    train = ds["train"]  # ~95%
    val = ds["validation"]  # ~5%

    # (If you have local parquet shards instead, comment the three lines above
    #  and uncomment this single line to create `full_ds` directly from parquet)
    # full_ds = load_dataset("parquet", data_files="path/to/oasst_full/*.parquet", split="train")

    # --- Merge splits (if using HF Hub route) ---------------------------------
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
    Path('experiment_result.json').write_text(result.model_dump_json(indent=2), 'utf-8')


if __name__ == '__main__':
    # run_example()
    run_experiment()
