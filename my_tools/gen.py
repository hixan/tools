import operator as op
from collections import OrderedDict
import hashlib
import pickle
from collections import deque, Iterator as It
from functools import reduce
from typing import Iterator, TypeVar, Sequence, Iterable
from pathlib import Path
from functools import wraps
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
        try:
            mem.append(next(iterator))
        except StopIteration:
            return

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


import inspect

class DiskCache:

    def __init__(self,
                 cache_directory: str = '.cache',
                 name_length: int = None,
                 name_str: str = '{hash}',
                 identifier_argument_name: str = 'cache_identifier',
                 ):
        self.name_length = name_length
        self.cache_directory = Path(cache_directory)
        self.argname = identifier_argument_name
        self.name_str = name_str
        print(self.cache_directory.resolve())

        if not self.cache_directory.exists():
            print('abc')
            self.cache_directory.mkdir(parents=True, exist_ok=True)

    def __call__(self, func):
        fparams = inspect.signature(func).parameters
        if self.argname in fparams:
            raise ValueError(f'DiskCache cannot overwrite {func.__name__} '
                             f'argument {self.argname}. Please provide an '
                             f'alternative identifier_argument_name')

        @wraps(func)
        def rv(*args, **kwargs):
            # get the name to save to
            self._fill_defaults(args, kwargs, fparams)
            cachefile = self._get_cachefile(args, kwargs)

            if cachefile.exists():
                # load the results
                with cachefile.open('rb') as f:
                    return pickle.load(f)
            else:
                # save the results
                retv = func(*args, **kwargs)
                with cachefile.open('wb') as f:
                    pickle.dump(retv, f)
                return retv
        
        return rv

    def _fill_defaults(self, args: tuple, kwargs: dict,
                       parameters: OrderedDict):
        # fill in default values from kwargs that have not been consumed so all
        # arguments are passed to the function
        for n, p in list(parameters.items())[len(args):]:
            if p.default is inspect._empty:  # type: ignore
                continue  # ignore empty parameters
            # update keyword arguments with default parameters if not contained
            if n not in kwargs:
                kwargs[n] = p.default

    def _get_cachefile(self, args: tuple, kwargs: dict) -> Path:
        # retrieve the Path object
        if self.argname in kwargs:
            rv = self.cache_directory / kwargs[self.argname]
            del kwargs[self.argname]  # remove from keyword arguments
            return rv
        return self.cache_directory / self.name_str.format(hash=self._get_hash(args, kwargs))

    def _get_hash(self, args: tuple, kwargs: dict) -> str:
        # Attempts to hash all objects in args, kwargs to a stable hash.
        # Raises a value error if this cannot be done.
        hash = hashlib.sha512()
        reduce(lambda x, y: 0, map(hash.update, self._make_hashable(args)))
        reduce(lambda x, y: 0, map(hash.update, self._make_hashable((kwargs,))))
        return hash.hexdigest()

    def _make_hashable(self, args: Iterable) -> Iterator:
        for a in args:
            if isinstance(a, Iterator):
                raise TypeError(f'cannot hash iterator')
            if type(a) is int:
                nbytes = (a.bit_length() + 5) // 4  # 4 bits to a byte
                yield a.to_bytes(nbytes, byteorder='big')
            elif type(a) is bytes:
                yield a
            elif type(a) is str:
                yield b'str'
                yield bytes(a, 'utf-8')
            elif type(a) is OrderedDict:
                yield b'odict'
                yield from self._make_hashable(a.items())
            elif type(a) is dict:
                yield b'dict'
                yield from self._make_hashable(sorted(a.items()))
            elif type(a) is list:
                yield bytes(str(type(a)), 'utf-8')
                yield from self._make_hashable(a)
            elif type(a) is set or type(a) is frozenset:
                yield b'set'
                try:
                    yield from self._make_hashable(frozenset(a))
                except TypeError:
                    yield from self._make_hashable(sorted(a))
            elif type(a) is tuple:
                yield from self._make_hashable(a)
            else:
                raise TypeError(f'could not hash {a}')



if __name__ == '__main__':

    @DiskCache()
    def foo(a, /, b, x='abc'):
        pass

    foo(1, 2)
