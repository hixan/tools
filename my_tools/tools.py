import inspect
import matplotlib.pyplot as plt
from pathlib import Path
from timeit import default_timer as timer
from collections import Counter
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

