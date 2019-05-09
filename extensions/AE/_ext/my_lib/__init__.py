
from torch.utils.ffi import _wrap_function
from ._my_lib import lib as _lib, ffi as _ffi

__all__ = []
def _import_symbols(locals):
    for symbol in dir(_lib):
        # get _lib attribute of symbol
        fn = getattr(_lib, symbol)
        # check whether contains __call() function in fn
        if callable(fn):
            locals[symbol] = _wrap_function(fn, _ffi)
        else:
            locals[symbol] = fn
        __all__.append(symbol)

_import_symbols(locals())
