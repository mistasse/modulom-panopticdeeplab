from types import FunctionType
import collections
import functools
import textwrap
import inspect


def override(original):
    def decorator(f):
        f.__doc__ == original.__doc__
        return f
    return decorator


_NOVALUE = object()

_kw_kinds = (inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.VAR_KEYWORD,
            )
NEVER_CACHE_CONDITION = 'True'
def cached_method(f, no_cache_condition=None):
    """Wraps a function so as to add a caching mechanism.

    Rationale behind is that providing keyword-only parameters
    disables the cache, while positional parameters compose
    the cache key.
    """
    if no_cache_condition == NEVER_CACHE_CONDITION:
        return f

    signature = inspect.signature(f)
    parameters = list(signature.parameters.values())[1:]
    parameters = [
        param
        for param in parameters
        if param.kind not in _kw_kinds
        ]
    pos_decl = ", ".join(str(param) for param in parameters)
    pos_pass = ", ".join(param.name for param in parameters)
    condition = "kwargs"
    if no_cache_condition is not None:
        condition = '%s or %s' % (condition, no_cache_condition)
    code = """
    def wrapper(self, {pos_decl}, **kwargs):
        if {condition}:
            return f(self, {pos_pass}, **kwargs)
        key = (f.__name__, {pos_pass})
        cache = self.__cache__
        value = cache.get(key, NOVALUE)
        if value is NOVALUE:
            value = f(self, {pos_pass})
            cache[key] = value
        return value
    """
    code = textwrap.dedent(code)
    code = code.format(pos_decl=pos_decl, pos_pass=pos_pass, condition=condition)
    code = code.replace(", ,", ",")

    env = dict(f=f, NOVALUE=_NOVALUE)
    exec(code, env)
    wrapper = env["wrapper"]
    return functools.wraps(f)(wrapper)


class WithCache:
    """On subclass instantiation, rather use a class with caching
    capabilities.

    Methods f are replaced by cached_method(f) unless they are
    decorated with @WithCache.no_cache.
    """

    @staticmethod
    def no_cache(condition=NEVER_CACHE_CONDITION):
        def decorator(f):
            setattr(f, "_no_cache_condition", condition)
            return f

        if callable(condition):
            f, condition = condition, NEVER_CACHE_CONDITION
            return decorator(f)
        return decorator

    def __new__(cls, *args, **kwargs):
        extension = cls.__dict__.get("__caching__", None)
        if extension is None:
            dct = {}
            for name in dir(cls):
                if name.startswith("_"):
                    continue
                f = _NOVALUE
                for c in cls.mro():
                    if c is WithCache: break  # don't add cache to methods behind WithCache
                    f = c.__dict__.get(name, _NOVALUE)
                    if f is not _NOVALUE: break
                if f is _NOVALUE or not isinstance(f, FunctionType):
                    continue
                f = cached_method(f, getattr(f, "_no_cache_condition", None))
                dct[name] = f

            extension = type(cls.__name__+"WithCache", (cls,), dct)
            cls.__caching__ = extension
        ret = super().__new__(extension) # *args, **kwargs
        ret.__cache__ = {}
        return ret

    @property
    def _cache(self):
        """Disentangles the cache per function. This makes cache visualization easier,
        not meant to be used in the code
        """
        ret = collections.defaultdict(dict)
        for k, v in self.__cache__.items():
            name = k[0]
            ret[name][k[1:]] = v
        return dict(ret)


class OutputMath(WithCache):
    """A session to keep operations done on one batch in-memory.

    This class' subclasses will be instrumented in order to be
    added a cache when methods are called without keyword-only
    parameters.
    No cache will be added on methods accompanied by a @OutputMath.no_cache
    """
    config = None

    def __init__(self, raw_batch, *, device='cpu', module=None):
        self.raw_batch = raw_batch
        self.module = module
        self.device = device

    def __getitem__(self, key):
        if isinstance(key, str):
            key = (key,)
        name, *args = key
        return getattr(self, name)(*args)

    def __setitem__(self, key, value):
        if isinstance(key, str):
            key = (key,)
        self.__cache__[key] = value
