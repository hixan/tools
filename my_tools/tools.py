import inspect
from copy import copy
import datetime
import pickle
from functools import wraps, reduce
import matplotlib.pyplot as plt
from pathlib import Path
from timeit import default_timer as timer
from collections import Counter, defaultdict
import numpy as np
from functools import reduce, partial
import operator as op
from itertools import chain, filterfalse
import logging
import os
from typing import Callable, Union


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


#def loudfunction(outputlen: int = None, print_first: bool = True):
#    if outputlen is None:
#        outputlen = int(os.environ['COLUMNS'])
#
#    outputlen -= 2
#    def rvfunc(function: Callable,
#               print_first: bool = True,
#               outputlen: int = outputlen):
#        def f(first, *args, **kwargs):
#            def print_args(args):
#                # possibly skip first argument
#                if print_first:
#                    printargs = first, *args
#                else:
#                    printargs = args
#
#                # print unnamed arguments
#                if len(printargs) > 0:
#                    # len(' -> ')
#                    fmt = '{value} -> {type}'
#                    for arg in printargs:
#                        t = type(arg).__name__
#                        v = str(arg)
#                        t = truncate(t, outputlen // 3)
#                        v = truncate(v, outputlen - len(t) - 4)
#                        print('│',
#                              f"{fmt.format(value=v, type=t): <{outputlen}}",
#                              '│', sep='')
#
#            def print_kwargs(kwargs):
#                # print named arguments: argname = 20 -> int
#                if len(kwargs) > 0:
#                    fmt = '{name} = {value} -> {type}'
#                    for argname in kwargs:
#                        value = kwargs[argname]
#                        olen = outputlen - 7
#                        # name is important - should be // 3
#                        n = truncate(argname, (olen // 2))
#                        t = truncate(type(value).__name__, olen - len(n) // 2)
#                        v = truncate(str(kwargs[argname]), olen - len(t) - len(n))
#                        print('│', f"{fmt.format(name=n, value=v, type=t): <{outputlen}}", '│', sep='')
#
#            # call function: -----start call to 'function'------
#
#            fname = f"'{function.__name__}'"
#            # print starting line ------fname-------
#            print('╒', truncate(f'{fname:═^{outputlen}}', maxlength=outputlen,
#                                elipses=False), '╕', sep='')
#            print_args(args)
#            print_kwargs(kwargs)
#            print('├', truncate(f'{"Start call "+fname:─^{outputlen}}',
#                                maxlength=outputlen),
#                  '┤', sep='')
#            st = timer()
#            rv = function(first, *args, **kwargs)
#            et = timer()
#            print('├', truncate(
#                f'{"End call "+fname+f" - {et-st:.2f}s elapsed":─^{outputlen}}',
#                maxlength=outputlen),
#                  '┤', sep='')
#            print_args(args)
#            print_kwargs(kwargs)
#            fmt = 'rv: {value} -> {type}'
#            olen = outputlen - 8
#            v = truncate(str(rv), olen // 2)
#            t = truncate(type(rv).__name__, olen - len(v))
#            print('│', f"{fmt.format(value=v, type=t): <{outputlen}}", '│', sep='')
#            print('╘', '═' * outputlen, '╛', sep='')
#            return rv
#
#        return f
#
#    return rvfunc


class loudfunction:

    _instances = {}  # type: ignore
    _initialized = False  # dont run __init__ after first call
    
# ─━│┃┄┅┆┇┈┉┊┋┌┍┎┏┐┑┒┓└┕┖┗┘┙┚┛├┝┞┟┠┡┢┣┤┥┦┧┨┩┪┫┬┭┮┯┰┱┲┳┴┵┶┷┸┹┺┻┼┽┾┿╀╁╂╃╄╅╆╇╈╉╊╋╌
# ╍╎╏═║╒╓╔╕╖╗╘╙╚╛╜╝╞╟╠╡╢╣╤╥╦╧╨╩╪╫╬╭╮╯╰╱╲╳╴╵╶╷╸╹╺╻╼╽╾╿
    sections = dict(
        single_pipe_vert = '│',
        right_single_single_t = '├',
        left_single_single_t = '┤',
        right_down_double_single = '╒',
        left_down_double_single = '╕',
        double_pipe_horiz = '═',
        single_pipe_horiz = '─',
        right_up_double_single = '╘',
        left_up_double_single = '╛',
    )

    def __new__(cls, *args, file: str = None, **kwargs):
        if file in cls._instances:
            return cls._instances[file]
        rv = super(loudfunction, cls).__new__(cls)
        cls._instances[file] = rv
        return cls._instances[file]

    def __init__(self, file: Path = None, print_first_arg: bool = True,
                 print_formats: dict[Union[str, int], str] = None,
                 color: bool = True, max_depth: int = 50, columns: int = None):
        #if self._initialized:
        #    return
        if columns is None:
            columns = int(os.environ['COLUMNS'])
        self.columns = columns
        self._initialized = True
        self._file = file
        self._pfa_ix = int(not print_first_arg)
        self.stack: list[str] = []
        self.print_formats = print_formats
        self.color = file is None and color

        self.colors = copy(BColors.all)
        self.colors = list(chain.from_iterable(zip(self.colors[1::2],
                                                   self.colors[::2])))
        self.colors *= int(max_depth // len(BColors.all) + 1)

        self.color_reset = BColors.reset if self.color else ''

    @property
    def current_color(self):
        if self.color:
            return self.colors[len(self.stack) - 1]
        else:
            return ''

    def __call__(self, func):
        @wraps(func)
        def louded_function(*args, **kwargs):
            self.stack.append(func.__name__)

            # print output

            # first line
            start0, slen0 = self._t_section_base(
                right_key='right_down_double_single',
                left_key='left_down_double_single')
            start1, slen1 = self._t_section_base(
                right_key='right_down_double_single',
                left_key='left_down_double_single', reverse=True)
            start_inner_len = self.columns - slen0 - slen1
            if start_inner_len > 5:
                start_inner_s = loudfunction._pad_str(
                    (func.__name__, len(func.__name__)),
                    start_inner_len, self.sections['double_pipe_horiz'])
                print(start0 + start_inner_s + start1)
                # arguments
                print(self._arguments_section(args[self._pfa_ix:], kwargs, self.columns))
                # seperator
                start0, slen0 = self._t_section_base(
                    right_key='right_single_single_t',
                    left_key='left_single_single_t')
                start1, slen1 = self._t_section_base(
                    right_key='right_single_single_t',
                    left_key='left_single_single_t', reverse=True)
                mid_inner_len = self.columns - slen0 - slen1
                mid_inner_s = loudfunction._pad_str(
                    ('', 0), mid_inner_len, self.sections['single_pipe_horiz'])
                print(start0 + mid_inner_s + start1)

            # call function
            rv = func(*args, **kwargs)
            if start_inner_len > 5:
                # seperator
                print(start0 + mid_inner_s + start1)

                # print the rest of the output
                if type(rv) is tuple:
                    print(self._arguments_section(rv, {}, self.columns))
                else:
                    print(self._arguments_section((rv,), {}, self.columns))
                start0, slen0 = self._t_section_base(
                    right_key='right_up_double_single',
                    left_key='left_up_double_single')
                start1, slen1 = self._t_section_base(
                    right_key='right_up_double_single',
                    left_key='left_up_double_single', reverse=True)
                mid_inner_len = self.columns - slen0 - slen1
                mid_inner_s = loudfunction._pad_str(
                    ('', 0), mid_inner_len, self.sections['double_pipe_horiz'])
                print(start0 + mid_inner_s + start1)

            assert self.stack.pop() == func.__name__
            return rv
        return louded_function

    def _pipe_section(self, offset=0, reverse=False, pkey='single_pipe_vert'
                      ) -> tuple[str, int]:
        n_chars = len(self.stack) - offset
        lpipe = self.sections[pkey]
        if reverse:
            lpipe = ''.join(reversed(lpipe))
        if self.color:
            rv = ''
            cls = self.colors[:n_chars]
            if reverse:
                cls = list(reversed(cls))
            for col in cls:
                rv += self.color_reset + col + lpipe
            return rv + self.color_reset, len(lpipe) * n_chars
        else:
            return lpipe * n_chars, len(lpipe) * n_chars

    def _t_section_base(self, right_key = 'right_single_single_t',
                        left_key = 'left_single_single_t', reverse=False
                        ) -> tuple[str, int]:

        ft = self.sections[right_key]
        bt = self.sections[left_key]
        if not reverse:
            value, l = self._pipe_section(offset=1)
            if self.color:
                value += self.color_reset + self.current_color + ft
            else:
                value += ft
            l += len(ft)
        else:
            if self.color:
                value = self.color_reset + self.current_color + bt
            else:
                value = bt
            v, l = self._pipe_section(offset=1, reverse=True)
            value += v
            l += len(bt)
        return value, l

    def _arguments_section(self, args: tuple, kwargs: dict, maxlen: int) -> str:
        first_pipe_section, fps_len = self._pipe_section()
        second_pipe_section, sps_len = self._pipe_section(reverse=True)

        outs = []
        inner_len = maxlen - fps_len - sps_len
        strs = loudfunction._allarg_string(
            args, kwargs, inner_len, self.current_color,
            self.color_reset)
        for s in strs:
            outs.append(first_pipe_section
                        + loudfunction._pad_str(s, inner_len, ' ')
                        + second_pipe_section)
        return '\n'.join(outs)

    @staticmethod
    def _pad_str(string: tuple[str, int], to_length: int, pad_value: str) -> str:
        return string[0] + pad_value * (to_length - string[1])

    @staticmethod
    def _allarg_string(args, kwargs, maxlen, curr_color, reset_color, /):
        sts = []
        pad = ' ' * maxlen
        for arg in args:
            sts.append(loudfunction._default_arg_string(arg, maxlen, curr_color, reset_color))
        for k, v in kwargs.items():
            sts.append((reset_color + k[:maxlen-1] + ':', min(len(k)+1, maxlen)))
            sts.append(loudfunction._default_arg_string(v, maxlen, curr_color, reset_color))
        return sts

    @staticmethod
    def _default_arg_string(arg, maxlen, curr_color, reset_color, /) -> tuple[str, int]:
        tpe = str(type(arg)).replace('\n', ' ')
        st = repr(arg).replace('\n', ' ')
        la = len(tpe) + len(st) + 3
        if len(tpe) < maxlen / 5 or la < maxlen:
            return f'{tpe} {curr_color}-{reset_color} {st}', la
        else:
            return st[:maxlen], min(len(st), maxlen)


def strlen_printable(string: str) -> int:
    return len(list(filterfalse(op.methodcaller('isprintable'), string)))


def loudmethod(outputlen=None):
    return loudfunction(print_first_arg=False)


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


def init_logging(fp: Path):

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


class BColors:
    black    = '\u001b[30m'
    red      = '\u001b[31m'
    green    = '\u001b[32m'
    yellow   = '\u001b[33m'
    blue     = '\u001b[34m'
    magenta  = '\u001b[35m'
    cyan     = '\u001b[36m'
    white    = '\u001b[37m'
    bblack   = '\u001b[30;1m'
    bred     = '\u001b[31;1m'
    bgreen   = '\u001b[32;1m'
    byellow  = '\u001b[33;1m'
    bblue    = '\u001b[34;1m'
    bmagenta = '\u001b[35;1m'
    bcyan    = '\u001b[36;1m'
    bwhite   = '\u001b[37;1m'

    all: list[str] = []

    reset    = '\u001b[0m'

BColors.all = [
        BColors.black, BColors.red, BColors.green, BColors.yellow,
        BColors.blue, BColors.magenta, BColors.cyan, BColors.white,
        BColors.bblack, BColors.bred, BColors.bgreen, BColors.byellow,
        BColors.bblue, BColors.bmagenta, BColors.bcyan, BColors.bwhite,
    ]


class Timer:
    
    def __init__(self, name=None, units='seconds', print_exit: bool = False):
        self.name = name
        self.started = False
        self.laps = []
        self.print_exit = print_exit
        
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
        if self.print_exit:
            print(self)
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
