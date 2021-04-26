"""
Compatibility helper for recent features of python
"""

import sys

try:
    from functools import cache
except ImportError:
    print(
        "WARNING: Could not import functools.cache (new in 3.9). "
        "Using lru_cache(maxsize=None) instead.",
        file=sys.stderr,
    )
    from functools import lru_cache

    def cache(f):
        return lru_cache(maxsize=None)(f)


try:
    from functools import cached_property
except ImportError:
    print(
        "WARNING: Could not import functools.cached_property (new in 3.8). "
        "Using property(cache(f)) instead.",
        file=sys.stderr,
    )

    def cached_property(f):
        return property(cache(f))
