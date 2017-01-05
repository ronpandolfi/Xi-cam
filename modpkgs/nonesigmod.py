# REF: http://stackoverflow.com/questions/23728401/pyside-crashing-python-when-emitting-none-between-threads

_PYSIDE_NONE_SENTINEL = object()


def pyside_none_wrap(var):
    """None -> sentinel. Wrap this around out-of-thread emitting."""
    if var is None:
        return _PYSIDE_NONE_SENTINEL
    return var


def pyside_none_deco(func):
    """sentinel -> None. Decorate callbacks that react to out-of-thread
    signal emitting.

    Modifies the function such that any sentinels passed in
    are transformed into None.
    """

    def sentinel_guard(arg):
        if arg is _PYSIDE_NONE_SENTINEL:
            return None
        return arg

    def inner(*args, **kwargs):
        newargs = map(sentinel_guard, args)
        newkwargs = {k: sentinel_guard(v) for k, v in kwargs.iteritems()}
        return func(*newargs, **newkwargs)

    return inner