#TODO it might be wiser to use the inspect module here

def get_arg_count(function):
    return function.__code__.co_argcount

def get_arg_names(function):
    return function.__code__.co_varnames[:get_arg_count(function)]

def get_arg_defaults(function):
    try:
        values = function.__defaults__
        keys = get_arg_names(function)[-len(values):]
        return dict(zip(keys, values))
    except TypeError:
        return {}
