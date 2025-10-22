"""
Custom Memory Cache Eviction with Adaptive Pipeline support
"""
from typing import Any, Callable, List, Tuple
import cachetools
from adaptive_pipeline import AdaptivePipelineCache
from gptcache.manager.eviction.base import EvictionBase


def popitem_wrapper(func, wrapper_func, clean_size):
    def wrapper(*args, **kwargs):
        keys = []
        try:
            keys = [func(*args, **kwargs)[0] for _ in range(clean_size)]
        except KeyError:
            pass
        wrapper_func(keys)
    return wrapper


class AdaptiveMemoryCacheEviction(EvictionBase):
    """
    Memory Cache Eviction with support for Adaptive Pipeline policy.
    
    :param policy: eviction strategy (LRU, LFU, FIFO, RR, or AP/Adaptive-Pipeline)
    :type policy: str
    :param maxsize: the maxsize of cache data
    :type maxsize: int
    :param clean_size: will clean the size of data when cache reaches max size
    :type clean_size: int
    :param on_evict: the function for cleaning the data in the store
    :type on_evict: Callable[[List[Any]], None]
    """
    def __init__(
            self,
            policy: str = "LRU",
            maxsize: int = 1000,
            clean_size: int = 0,
            on_evict: Callable[[List[Any]], None] = None,
            **kwargs,
    ):
        self._policy = policy.upper()
        if self._policy == "LRU":
            self._cache = cachetools.LRUCache(maxsize=maxsize, **kwargs)
        elif self._policy == "LFU":
            self._cache = cachetools.LFUCache(maxsize=maxsize, **kwargs)
        elif self._policy == "FIFO":
            self._cache = cachetools.FIFOCache(maxsize=maxsize, **kwargs)
        elif self._policy == "RR":
            self._cache = cachetools.RRCache(maxsize=maxsize, **kwargs)
        elif self._policy in ("AP", "ADAPTIVE-PIPELINE", "ADAPTIVEPIPELINE"):
            self._cache = AdaptivePipelineCache(maxsize=maxsize, **kwargs)
        else:
            raise ValueError(f"Unknown policy {policy}")
        
        self._cache.popitem = popitem_wrapper(self._cache.popitem, on_evict, clean_size)

    def put(self, objs: List[Tuple[int, Tuple[float, int]]]):
        for obj in objs:
            if isinstance(obj, tuple):
                self._cache[obj[0]] = obj[1]  # Fixed: was objs[1], should be obj[1]

    def get(self, obj: Any):
        return self._cache.get(obj)

    @property
    def policy(self) -> str:
        return self._policy



