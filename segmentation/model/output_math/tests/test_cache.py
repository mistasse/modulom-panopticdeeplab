# ------------------------------------------------------------------------------
# Tests for the caching mechasnism behind output mathematics
# Written by  Maxime Istasse (maxime.istasse@uclouvain.be)
# ------------------------------------------------------------------------------

from .. import utils


class SampleMath(utils.WithCache, dict):
    # This class also acts like a dict, WithCache only acts on
    # methods defined after it in MRO, dict calls shouldn't be
    # cached

    def cached(self):
        # Always cached
        return object()

    def _uncached(self):
        # Never cached because of heading "_"
        return object()

    @utils.WithCache.no_cache
    def uncached(self):
        # Never cached because of "@utils.WithCache.no_cache"
        return object()

    @utils.WithCache.no_cache()
    def uncached2(self):
        # Never cached because of "@utils.WithCache.no_cache()"
        return object()

    @utils.WithCache.no_cache('a is not None')
    def optional_cache(self, a=None, *, b=None, **kwargs):
        # Cached when a is None, b is unspecified and kwargs are empty
        return object()


def test_cache():
    math = SampleMath(var='dict calls shouldnt be cached')
    math2 = SampleMath(var='dict calls shouldnt be cached')
    assert type(math).__name__ == "SampleMathWithCache"
    assert type(math) is type(math2)

    assert math.pop('var', None) == 'dict calls shouldnt be cached'
    assert math.pop('var', None) is None

    assert math.cached() is math.cached()

    assert math._uncached() is not math._uncached()
    assert math.uncached() is not math.uncached()
    assert math.uncached2() is not math.uncached2()

    assert math.optional_cache() is math.optional_cache()
    assert math.optional_cache(a=None) is math.optional_cache(a=None)

    assert math.optional_cache(a=3) is not math.optional_cache(a=3)
    assert math.optional_cache(b=3) is not math.optional_cache(b=3)
    assert math.optional_cache(c=5) is not math.optional_cache(c=5)
