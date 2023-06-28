
def fullname(o):
    try:
        # if o is a class or function, get module directly
        module = o.__module__
    except AttributeError:
        # then get module from o's class
        module = o.__class__.__module__
    try:
        # if o is a class or function, get name directly
        name = o.__qualname__
    except AttributeError:
        # then get o's class name
        name = o.__class__.__qualname__
    # if o is a method of builtin class, then module will be None
    if module == 'builtins' or module is None:
        return name
    return module + '.' + name
