from itertools import islice
from functools import reduce
# generator/functional type tools


def stagger(iterable, offsets=(1,)):
    '''stagger iterable with offsets offsets.
    EG:
    stagger((1, 2, 4, 8, 16), offsets=(1, 3))
    will yield
    (1, 2, 8),
    (2, 4, 16)
     ^  ^  ^- third column offset by 3
     |  |---- second column offset by 1
     |------- first column offset by 0
    '''
    return zip(*(islice(iter(iterable), ofs, None) for ofs in (0, *offsets)))


class pospartial:

    def __init__(self, callable, *args, **kwargs):
        self.kwargs = kwargs
        self.args = args
        self.callable = callable

    def __call__(self, *args, **kwargs):
        arglist = []
        argiter = iter(args)
        for arg in self.args:
            if arg is ...:
                try:
                    arglist.append(next(argiter))
                except StopIteration:
                    raise ValueError(f'{self.callable.__name__} '
                    'expected more positional arguments')
            else:
                arglist.append(arg)
        arglist.extend(argiter)
        return self.callable(*arglist, **kwargs)


def composite2(f, g):
    def rv(*x, **y):
        return f(g(*x, **y))
    return rv


def composite(*f):
    return reduce(composite2, f)


