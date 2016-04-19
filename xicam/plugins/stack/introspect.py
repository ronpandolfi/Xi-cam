

def get_arg_count(function):
    return function.__code__.co_argcount


def get_arg_names(function):
    return function.__code__.co_varnames[:get_arg_count(function)]

def get_arg_defaults(function):
    values = function.__defaults__
    keys = get_arg_names(function)[-len(values):]
    return dict(zip(keys, values))

if __name__ == '__main__':
    import tomopy
    print get_arg_defaults(tomopy.remove_stripe_fw)