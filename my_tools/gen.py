import operator as op
from collections import deque, Iterator as It
from functools import reduce
from typing import Iterator, TypeVar, Sequence
# generator/functional type tools


T = TypeVar('T')


def stagger(iterator: Iterator[T], offsets: Sequence[int] = (1,),
            shortest: bool = True, padvalue: T = None
            ) -> Iterator[tuple[T, ...]]:
    '''stagger iterable with offsets.

    parameters:
        iterator:
            iterator or iterable of T - object to iterate over
        offsets:
            indexes on which to offset. Must be integers
        shortest:
            boolean to decide weather to pad the output to allow every index in
            offsets to return every value in iterator
        padvalue:
            value with which to pad output as outlined in shortest. Is ignored
            if shortest = False.

    EG:
    stagger((1, 2, 4, 8, 16), offsets=(1, 3))
    will yield
    (1, 2, 8),
    (2, 4, 16)
     ^  ^  ^- third column offset by 3
     |  |---- second column offset by 1
     |------- first column offset by 0

    >>> gen = stagger((1, 2, 4, 8, 16), offsets=(1, 3))
    >>> next(gen)
    (1, 2, 8)
    >>> next(gen)
    (2, 4, 16)

    the shortest argument controls weather to pad output with <padvalue> to
    ensure each offset accesses each element:
    >>> x = iter(range(5))
    >>> list(stagger(x, offsets=(1, 2), shortest=False))
    [(None, None, 0), (None, 0, 1), (0, 1, 2), (1, 2, 3), (2, 3, 4),\
 (3, 4, None), (4, None, None)]

    other test cases:
    >>> x = iter(range(5))
    >>> list(stagger(x))
    [(0, 1), (1, 2), (2, 3), (3, 4)]

    >>> x = iter(range(5))
    >>> list(stagger(x, offsets=(1, 2)))
    [(0, 1, 2), (1, 2, 3), (2, 3, 4)]

    >>> x = iter(range(8))
    >>> list(stagger(x, offsets=(4,), shortest=False))
    [(None, 0), (None, 1), (None, 2), (None, 3), (0, 4), (1, 5), (2, 6),\
 (3, 7), (4, None), (5, None), (6, None), (7, None)]

    >>> x = iter([1, 2, None, 3, 4, None])
    >>> list(stagger(x, offsets=(2,), shortest=False))
    [(None, 1), (None, 2), (1, None), (2, 3), (None, 4), (3, None), (4, None),\
 (None, None)]
    '''
    if not isinstance(iterator, It):
        yield from stagger(iter(iterator), offsets=offsets, shortest=shortest,
                           padvalue=padvalue)
        return

    maxoff = max(offsets)
    mem = deque([padvalue] * maxoff, maxlen=maxoff + 1)
    extractor = op.itemgetter(0, *offsets)
    try:
        mem.append(next(iterator))
    except StopIteration:
        return

    # yield first maxoff items (if required) and populate deque
    for _ in range(maxoff):
        if not shortest:
            yield extractor(mem)
        mem.append(next(iterator))

    # yield most of the iterator
    while True:
        yield extractor(mem)
        try:
            mem.append(next(iterator))
        except StopIteration:
            break

    # yield last items (if required)
    if not shortest:
        while maxoff > 0:
            maxoff -= 1
            mem.append(None)
            yield extractor(mem)


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


