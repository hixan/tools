import matplotlib.pyplot as plt
from pathlib import Path
from timeit import default_timer as timer
from collections import Counter
import numpy as np
from functools import reduce
import operator as op
import logging
from typing import Callable


def truncate(string: str, maxlength: int = 80, elipses: bool = True):
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
    def rvfunc(function: Callable,
               print_first: bool = True,
               outputlen: int = outputlen):
        def f(first, *args, **kwargs):
            fname = f"'{function.__name__}'"
            # print starting line ------fname-------
            print(
                truncate(f'{fname:-^{outputlen}}',
                         maxlength=outputlen,
                         elipses=False))

            # possibly skip first argument
            if print_first:
                printargs = first, *args
            else:
                printargs = args

            # print unnamed arguments
            if len(printargs) > 0:
                # len(' -> ')
                maxlenarg = (outputlen - 4) // 2
                fmt = '{value} -> {type}'
                for arg in printargs:
                    t = type(arg).__name__
                    v = str(arg)
                    t = truncate(t, outputlen // 3)
                    v = truncate(v, outputlen - len(t) - 4)
                    print(fmt.format(value=v, type=t))

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
                    print(fmt.format(name=n, value=v, type=t))

            # call function: -----start call to 'function'------
            print(
                truncate(f'{"Start call "+fname:-^{outputlen}}',
                         maxlength=outputlen))
            st = timer()
            rv = function(first, *args, **kwargs)
            et = timer()
            print(
                truncate(
                    f'{"End call "+fname+f" - {et-st:.2f}s elapsed":-^{outputlen}}',
                    maxlength=outputlen))
            fmt = 'rv: {value} -> {type}'
            olen = outputlen - 8
            v = truncate(str(rv), olen // 2)
            t = truncate(type(rv).__name__, olen - len(v))
            print(fmt.format(value=v, type=t))
            print('-' * outputlen)
            return rv

        return f

    return rvfunc


def loudmethod(outputlen=80):
    return loudfunction(outputlen=outputlen, print_first=False)


def plot_frequencies(categorical):
    ''' plot frequencies of an itterable containing categorical data'''
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


def init_logging(name: str,
                 debug=None,
                 fp: Path = Path('.'),
                 file_level=logging.NOTSET,
                 stream_level=logging.WARNING):

    if debug is not None:
        raise Warning(
            'using init_logging with debug is deprecated and has no effect')

    logger = logging.getLogger(name)
    logger.setLevel(0)

    fh = logging.FileHandler(fp / f'{name}.log')
    fh.setLevel(file_level)
    fmtstrs = (
        '{levelno:0>2}:{name:<25}(l:{lineno:0>4}):{levelname:>8}: {message}',
        '{asctime:-^44} : {args}')
    fmtstr = '\n'.join(fmtstrs)
    fh.setFormatter(logging.Formatter(fmtstr, style='{'))
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter(fmtstr, style='{'))
    sh.setLevel(stream_level)

    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger


def remove_outliers(df, column, low_percentile=0, high_percentile=1):
    low_cutoff = df[column].quantile(low_percentile)
    high_cutoff = df[column].quantile(high_percentile)
    return df[(df[column] >= low_cutoff) & (df[column] <= high_cutoff)]
