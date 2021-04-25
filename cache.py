"""
Compatibility helper for recent features of python
"""
try:
    from functools import cache
except ImportError:
    from functools import lru_cache

    def cache(f):
        return lru_cache(maxsize=None)(f)


try:
    from functools import cached_property
except ImportError:

    def cached_property(f):
        return property(cache(f))
