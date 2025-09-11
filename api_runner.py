import openai

from gptcache import cache
from gptcache.adapter import openai as cached_openai
from gptcache.processor.pre import get_prompt

# Set Ollama as LLM
openai.api_base = "http://localhost:11434/v1"
openai.api_key = "ollama"

cache.init(pre_embedding_func=get_prompt)


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
    ask("Write me a short script of calculator in python")
    print('-------------')
    # 2) Similar prompt -> HIT: llm_time_s == 0
    ask("Make a simple calculator in python")


if __name__ == '__main__':
    run_example()
