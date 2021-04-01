import inspect
import datetime
import pickle
from functools import wraps
import matplotlib.pyplot as plt
from pathlib import Path
from timeit import default_timer as timer
from collections import Counter, defaultdict
import numpy as np
from functools import reduce
import operator as op
import logging
from typing import Callable, Mapping, Tuple, Sequence


def truncate(string: str, maxlength: int = 80, elipses: bool = True):
    ''' Truncates a string nicely in the center

    :param string: string to truncate
    :param maxlength: maximum length of string to output
    :param elipses: set to true to explicitly include elipses '…' at the point
        of truncation

    if len(string) > maxlength this will always output a string of length
    maxlength.
    '''
    if maxlength <= 0:
        return ''
    if len(string) > maxlength:
        if elipses:
            maxlength -= 1
            endsectionlen = maxlength // 2
            if (maxlength / 2) % 1 != 0:
                startsectionlen = 1 + endsectionlen
            else:
                startsectionlen = endsectionlen
            rv = string[:startsectionlen] + '…' + string[-endsectionlen:]
        else:
            endsectionlen = maxlength // 2
            if (maxlength / 2) % 1 != 0:
                startsectionlen = 1 + endsectionlen
            else:
                startsectionlen = endsectionlen
            rv = string[:startsectionlen] + string[-endsectionlen:]
    else:
        rv = string
    return rv


def loudfunction(outputlen: int = 80, print_first: bool = True):
    outputlen -= 2
    def rvfunc(function: Callable,
               print_first: bool = True,
               outputlen: int = outputlen):
        def f(first, *args, **kwargs):
            def print_args(args):
                # possibly skip first argument
                if print_first:
                    printargs = first, *args
                else:
                    printargs = args

                # print unnamed arguments
                if len(printargs) > 0:
                    # len(' -> ')
                    fmt = '{value} -> {type}'
                    for arg in printargs:
                        t = type(arg).__name__
                        v = str(arg)
                        t = truncate(t, outputlen // 3)
                        v = truncate(v, outputlen - len(t) - 4)
                        print('│',
                              f"{fmt.format(value=v, type=t): <{outputlen}}",
                              '│', sep='')

            def print_kwargs(kwargs):
                # print named arguments: argname = 20 -> int
                if len(kwargs) > 0:
                    fmt = '{name} = {value} -> {type}'
                    for argname in kwargs:
                        value = kwargs[argname]
                        olen = outputlen - 7
                        # name is important - should be // 3
                        n = truncate(argname, (olen // 2))
                        t = truncate(type(value).__name__, olen - len(n) // 2)
                        v = truncate(str(kwargs[argname]), olen - len(t) - len(n))
                        print('│', f"{fmt.format(name=n, value=v, type=t): <{outputlen}}", '│', sep='')

            # call function: -----start call to 'function'------

            fname = f"'{function.__name__}'"
            # print starting line ------fname-------
            print('╒', truncate(f'{fname:═^{outputlen}}', maxlength=outputlen,
                                elipses=False), '╕', sep='')
            print_args(args)
            print_kwargs(kwargs)
            print('├', truncate(f'{"Start call "+fname:─^{outputlen}}',
                                maxlength=outputlen),
                  '┤', sep='')
            st = timer()
            rv = function(first, *args, **kwargs)
            et = timer()
            print('├', truncate(
                f'{"End call "+fname+f" - {et-st:.2f}s elapsed":─^{outputlen}}',
                maxlength=outputlen),
                  '┤', sep='')
            print_args(args)
            print_kwargs(kwargs)
            fmt = 'rv: {value} -> {type}'
            olen = outputlen - 8
            v = truncate(str(rv), olen // 2)
            t = truncate(type(rv).__name__, olen - len(v))
            print('│', f"{fmt.format(value=v, type=t): <{outputlen}}", '│', sep='')
            print('╘', '═' * outputlen, '╛', sep='')
            return rv

        return f

    return rvfunc


def loudmethod(outputlen=80):
    return loudfunction(outputlen=outputlen, print_first=False)


def plot_frequencies(categorical):
    '''plot frequencies of an itterable containing categorical data'''
    labels, frequencies = zip(*Counter(categorical).items())
    x = np.arange(len(labels))
    plt.bar(x, frequencies, width=1)
    plt.xticks(x, labels=labels)


def confusion_matrix(true, pred, labels=None):
    if labels is None:
        labels = list(set(true))
    rv = np.zeros([len(labels)]*2)
    for t, p in zip(true, pred):
        rv[labels.index(p), labels.index(t)] += 1
    return rv


def plot_confusion_matrix(mat, labels):
    '''
    :param mat: confusion matrix
    :param labels: ordered labels to show on graph
    '''
    plt.imshow(mat)
    plt.xticks(range(len(labels)), labels, rotation='vertical')
    plt.yticks([-0.5] + list(range(len(labels))) + [len(labels) - .5],
               [''] + list(labels) + [''],
               rotation='horizontal')


def plot_images(images, size=(10, 10)):
    ''' plot first prod(size) images in a grid with size dimensions '''
    plt.subplots_adjust(wspace=0, hspace=0)
    msize = reduce(op.mul, size, 1) - 1
    for i, image in enumerate(images):
        plt.subplot(*size[::-1], i + 1)
        plt.imshow(image)
        plt.axis('off')
        if i == msize:
            # only show first prod(size) images
            return


class DefaultArgumentsFilter(logging.Filter):
    ''' Filter that will assign default values when an attribute is missing '''

    def __init__(self, name, **defaults):
        super(DefaultArgumentsFilter, self).__init__(name)
        self.defaults = defaults

    def filter(self, record):
        for key in self.defaults:
            if not hasattr(record, 'user_id'):
                record.__dict__[key] = self.defaults[key]
        return True


def init_logging(fp: Path,
                 sql_db: Mapping[str, str] = None):

    logger = logging.Logger('')

    fh = logging.FileHandler(fp)
    sh = logging.StreamHandler()

    fmtstrs = (
        '{levelno:0>2}:{name:<25}(l:{lineno:0>4}):{levelname:>8}: {message}',
        '{asctime:-^44} : ')
    fmtstr = '\n'.join(fmtstrs)
    sh.setFormatter(logging.Formatter(fmtstr, style='{'))

    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger


def remove_outliers(df, column, low_percentile=0, high_percentile=1):
    low_cutoff = df[column].quantile(low_percentile)
    high_cutoff = df[column].quantile(high_percentile)
    return df[(df[column] >= low_cutoff) & (df[column] <= high_cutoff)]


def debug_print(*args, **kwargs):
    # ignore printing to a file
    if 'file' in kwargs:
        print(*args, **kwargs)
    
    frame = inspect.stack()[1][0]
    info = inspect.getframeinfo(frame)
    print(info.filename, info.code_context, info.function, info.lineno,
            ':', *args, **kwargs)


def var_covar_matrix(X, mean=None, axis=0):
    assert len(X.shape) == 2, 'must operate on a matrix of 2 dimensions'
    if axis == 1:  # calculate on transpose
        return var_covar_matrix(X.T, mean=mean)
    elif axis != 0:
        raise ValueError('axis must 0 or 1')
    # axis is now == 0

    if mean is None:
        mean = np.mean(X, axis=axis)
    diff = X - mean

    # sum of outer products for each vector divided by the number of vectors
    rv = (diff.T @ diff) / X.shape[0]
    return rv


def iterable_filter(filter):
    def rv(gen):
        return (value for value in gen if filter(value))
    return rv


def sm_apply(eqn, *methods):
    '''
    apply methods to sympy equation. A method is defined as follows:

    (callable method, *arguments, dict(**kwargs))
    '''
    rv = eqn
    for method, *args, kwargs in methods:
        print(*args, kwargs)
        rv = method(eqn, *args, **kwargs)
    return rv


calls = defaultdict(lambda : 0)
def inspect_array(arr, name=''):
    calls[name] += 1
    print(f'''{name} {calls[name]}:
    shape: {arr.shape}
    dtype: {arr.dtype}
    mean: {np.mean(arr)}
    counter: {Counter(arr.flatten())}
    ''')


def composite2(f, g):
    def rv(*x, **y):
        return f(g(*x, **y))
    return rv


def composite(*f):
    return reduce(composite2, f)


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


class DivCounter(Counter):
    ''' a Counter object that supports log and power operations '''

    def _operation(self, other, operation):

        if issubclass(type(other), Counter):
            return DivCounter({k: operation(self[k], other[k]) for k in self})
        return DivCounter({k: operation(self[k], other) for k in self})

    def __add__(self, other):
        return self._operation(other, op.add)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        # only need to worry about the shortest as if an entry is missing
        # multiplying by 0 results in nothing anyway

        return self._operation(other, op.mul)

    def __rmul__(self, other):
        return self._operation(other, op.mul)

    def __pow__(self, exponent):
        return self._operation(exponent, op.pow)

    def __rpow__(self, other):
        return self._operation(other, lambda x, y: y ** x)

    def __truediv__(self, other):
        return self._operation(other, op.truediv)

    def __rtruediv__(self, other):
        return self._operation(other, lambda x, y: y / x)

    def __gt__(self, other):
        return self._operation(other, op.gt)

    def __lt__(self, other):
        return self._operation(other, op.lt)

    def __rgt__(self, other):
        return self.__lt__(other)

    def __rlt__(self, other):
        return self.__gt__(other)

    def __ge__(self, other):
        return self._operation(other, op.ge)

    def __le__(self, other):
        return self._operation(other, op.le)

    def __rge__(self, other):
        return self.__le__(other)

    def __rle__(self, other):
        return self.__ge__(other)

    # numpy log
    def log(self):
        new = DivCounter()
        for v in self:
            new[v] = np.log(self[v])
        return new


class Timer:
    
    def __init__(self, name=None, units='seconds'):
        self.name = name
        self.started = False
        self.laps = []
        
        self.units = units
        self.to_units = {
            'seconds': lambda x: x.total_seconds(),
        }[str(units)]
        
    def __enter__(self):
        self.started = True
        self.start = datetime.datetime.now()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        rv = self.lap()
        self.started = False
    
    def lap(self):
        '''Save a checkpoint for later access.'''
        self.laps.append(self.get_time())
        return self.laps[-1]
    
    def get_time(self):
        '''get the current timer reading'''
        if self.started:
            return self.to_units(datetime.datetime.now() - self.start)
        # return None otherwise

    def __str__(self):
        name = " " + repr(self.name) if self.name is not None else ""
        if self.started:
            return (f'Timer{name} at '
                    f'{self.get_time()} {self.units}')
        if len(self.laps) > 0:
            return (f'Stopped Timer{name} at '
                    f'{self.laps[-1]} {self.units}')
        return f'Timer{name}'
    
    def time_taken(self):
        '''get the total time taken (must be a timer that was started and
        completed)'''
        assert not self.started, f'Timer {repr(self.name)} still running!'
        return self.laps[-1]


def cachify(function):
    '''Wrap function to save its returned value to a file if not previously run,
    and load the file instead of re-computing if it has been.

    The wrapped function needs an additional argument: filename.
    An optional extra argument 'overwrite' allows the function to always
    recalculate and save output, overwriting previously cached calls.
    '''
    root = Path('./.cache')
    @wraps(function)  # keep original docstring and name
    def rv(*args, filename=None, overwrite=False, **kwargs):
        # sanitize filename (construct default)
        if filename is None:
            raise TypeError(f'{function.__name__} missing 1 required keyword '
                    'argument: \'filename\'\n'
                    'This argument is required as the function has been '
                    'cachified.')
        
        file = root / filename
        if file.exists() and not overwrite:
            # load saved value
            with file.open('rb') as f:
                return pickle.load(f)
        
        fret = function(*args, **kwargs)

        # save the value
        with file.open('wb') as f:
            pickle.dump(fret, f)

        return fret

    return rv

