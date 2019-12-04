import matplotlib.pyplot as plt  # type: ignore
from collections import Counter
import numpy as np  # type: ignore
from functools import reduce
import operator as op
import logging


def loudfunction(function, print_first=True):
    def f(first, *args, **kwargs):
        es = '---------------'
        start = f'Calling function {function.__name__} with arguments:{es}'
        print(start)
        if print_first:
            print(f'{type(first)} -> {first}')
        print(*map(lambda x: f'{type(x)} -> {x}', args), sep='\n')
        print(
            *map(
                lambda x: f'{x[0]} = {type(x[1])} -> {x[1]}',
                kwargs.items()
            ),
            sep='\n'
        )
        print(es)
        rv = function(*args, **kwargs)
        end = f'End call to function {function.__name__}{es}'
        end += '-' * (len(start) - len(end))
        print(end)
        print(f'return value: {type(rv)} .. {rv}')
        return rv
    return f


def loudmethod(method):
    return loudfunction(method, print_first=False)


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
    plt.yticks(
        [-0.5] + list(range(len(labels))) + [len(labels) - .5],
        [''] + list(labels) + [''],
        rotation='horizontal'
    )


def plot_images(images, size=(10, 10)):
    ''' plot first prod(size) images in a grid with size dimensions '''
    plt.subplots_adjust(wspace=0, hspace=0)
    msize = reduce(op.mul, size, 1) - 1
    for i, image in enumerate(images):
        plt.subplot(*size[::-1], i+1)
        plt.imshow(image)
        plt.axis('off')
        if i == msize:
            # only show first prod(size) images
            return


def init_logging(name: str, debug=False):

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(f'{name}.log')
    fh.setLevel(logging.INFO)
    fmtstr = '%(name)s:%(levelname)s: %(asctime)s \n%(message)s'
    fh.setFormatter(logging.Formatter(fmtstr))
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(fmtstr.replace('\n', '')))
    if debug:
        ch.setLevel(logging.DEBUG)
    else:
        ch.setLevel(logging.WARNING)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def remove_outliers(df, column, low_percentile=0, high_percentile=1):
    low_cutoff = df[column].quantile(low_percentile)
    high_cutoff = df[column].quantile(high_percentile)
    return df[(df[column] >= low_cutoff) & (df[column] <= high_cutoff)]

