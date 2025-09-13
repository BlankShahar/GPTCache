import openai

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
    eviction="LFU",
    max_size=2,
    clean_size=1
)
cache.init(
    embedding_func=ollama_embed,
    data_manager=data_manager,
    similarity_evaluation=SearchDistanceEvaluation(),
)


def ask(prompt: str) -> str:
    result = cached_openai.ChatCompletion.create(
        model="llama3.1:8b",
        messages=[{"role": "user", "content": prompt}],
    )
    # response = resp["choices"][0]["message"]["content"]
    result_metadata = result.get("gptcache_meta", {})
    print(
        f"hit={result_metadata.get('hit')}  "
        f"llm_time_s={result_metadata.get('llm_time_s')}  "
        f"total_time_s={result_metadata.get('total_time_s')}"
    )
    # print(response)
    return result


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


if __name__ == '__main__':
    run_example()
