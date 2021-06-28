import operator as op
from functools import partial
from typing import Callable, Dict, Iterable, Union

import numpy as np
import torch
from matplotlib import pyplot as plt

from skorch import NeuralNet
from skorch.callbacks import Callback, BatchScoring


class PlotterCallback(Callback):
    '''callback to plot model performance over epochs and save to file

    parameters:
        filename:
            filename with {key} in it somewhere (to format on save for
            every key in keys)
        keys:
            list of strings, must all be present in net.history prepended
            by both 'train_' and 'valid_'.
    '''

    def __init__(self,
                 keys: Iterable[str],
                 colors: Iterable[str] = ('blue', 'grey'),
                 filename: Union[str, Callable[[NeuralNet], str]] = None,
                 ):
        if isinstance(filename, str):
            self.filename = lambda _: filename
        else:
            self.filename = filename
        self.keys = keys
        self.colors = colors

    def on_train_begin(self, net, X, y):
        if self.filename is None:
            return
        self.filenames = {}
        fn = self.filename(net)
        for key in self.keys:
            self.filenames[key] = fn.format(key=key)

    def _color_lines_boxplot(self, bp, color):
        element_getter = op.itemgetter('boxes', 'whiskers', 'fliers', 'means',
                                       'medians', 'caps')
        list(map(partial(plt.setp, color=color), element_getter(bp)))

    def _color_fill_boxplot(self, bp, color):
        element_getter = op.itemgetter('boxes', 'whiskers', 'fliers', 'means',
                                       'medians', 'caps')
        list(map(partial(plt.setp, color=color), element_getter(bp)))

    def on_train_end(self, net, *args, **kwargs):
        for key in self.keys:

            key_train = 'train_' + key
            key_valid = 'valid_' + key
            ks = []
            if key_train in net.history[-1]:
                ks.append(key_train)
            if key_valid in net.history[-1]:
                ks.append(key_valid)

            plt.figure()
            for k, c in zip(ks, self.colors):
                # first axis: epochs, second axis: batch
                # key data over batches
                bdat = np.array(net.history[:, 'batches', :, k])

                edat = np.array(net.history[:, k])

                pos = np.arange(len(bdat))

                # do boxplot of data
                bp = plt.boxplot(bdat.T, widths=0.3, positions=pos - 0.2,
                                  patch_artist=True, zorder=0, showmeans=False,
                                  showfliers=False)
                # color the boxplot with the appropriate color
                self._color_lines_boxplot(bp, 'light' + c)
                # plot the epochs as a line graph
                plt.plot(edat, label=k.replace('_', ' ').title(), zorder=1, c=c)
                n_ticks = 20
                xt = np.arange(len(bdat))[::1 if len(bdat) < n_ticks
                                          else len(bdat) // n_ticks]
                plt.xticks(xt, xt)
            plt.xlabel('Epoch')
            key_title = key.replace('_', ' ').title()
            plt.ylabel(key_title)
            plt.legend()
            plt.title(f'{key_title} over Epochs')

            if self.filename is None:
                plt.show()
            else:
                plt.savefig(self.filenames[key])
                plt.close()


class ResultSaveCallback(Callback):

    def __init__(self, filename: Union[str, Callable[[NeuralNet], str]]):
        if isinstance(filename, str):
            self.filename = lambda x: filename
        else:
            self.filename = filename

    def on_train_end(self, net, *args, **kwargs):
        net.history.to_file(self.filename(net))


class BatchScoringWithUnpacker(BatchScoring):

    def __init__(self, *args, unpacker: Callable[..., Dict], **kwargs):
        '''

        unpacker should accept any kwargs, but specifically useful will be:
        - y
        - y_pred

        must return updated transformation as a dict, that will get called:
            kwargs.update(rv)
        '''
        super(BatchScoring, self).__init__(*args, **kwargs)
        self.transformer = unpacker

    def on_batch_end(self, **kwargs):
        kwargs.update(self.transformer(kwargs))
        return super().on_batch_end(**kwargs)


class AccuracyCallback(Callback):

    def __init__(self, unpacker_function = None):
        """
        unpacker_function is the function to call to 'unpack' prediction
        values and target values. This may be useful if using multi-task
        learning for example, if each example has multiple classes to predict,
        this can unpack the multiple dimensions into one dimension.
        """

        if unpacker_function is not None:
            self.unpacker = unpacker_function
        else:
            self.unpacker = lambda x: x

    def on_batch_end(self, net, X, y, training, loss, y_pred):
        if X is None or len(X) == 0:
            return  # nothing to plot
        y, y_pred = map(self.unpacker, [y, y_pred])
        bacc_total = (y_pred.argmax(1).cpu() == y).float().sum().item()
        bacc = bacc_total / y_pred.shape[0]
        prefix = 'train' if training else 'valid'
        net.history.record_batch(prefix + '_acc', bacc)
        net.history.record_batch(prefix + '_acc_total', bacc_total)
        net.history.record_batch(prefix + '_acc_len', y.shape[0])

    def on_epoch_end(self, net, dataset_train, dataset_valid):
        if dataset_train is not None and len(dataset_train) > 0:
            x = np.array(net.history[-1, 'batches', :, 'train_acc_total'])
            t = x.sum() / np.sum(net.history[-1, 'batches', :, 'train_acc_len'])
            net.history.record('train_acc', t)

        if dataset_valid is not None and len(dataset_valid) > 0:
            y = np.array(net.history[-1, 'batches', :, 'valid_acc_total'])
            v = y.sum() / np.sum(net.history[-1, 'batches', :, 'valid_acc_len'])
            net.history.record('valid_acc', v)


class LogisticAccuracyCallback(Callback):

    def on_batch_end(self, net, X, y, training, loss, y_pred):
        bacc = ((y_pred > 0.5).int() == y).float().mean().item()
        prefix = 'train' if training else 'valid'
        net.history.record_batch(prefix + '_acc', bacc)

    def on_epoch_end(self, net, dataset_train, dataset_valid):
        x = np.array(net.history[-1, 'batches', :, 'train_acc'])
        y = np.array(net.history[-1, 'batches', :, 'valid_acc'])
        t = x.sum() / len(dataset_train)
        v = y.sum() / len(dataset_valid)
        net.history.record('train_acc', t)
        net.history.record('valid_acc', v)


class F1Callback(Callback):

    def on_batch_end(self, net, X, y, training, loss, y_pred):
        if X is None or len(X) == 0:
            return  # nothing to plot

        prefix = 'train' if training else 'valid'
        y = y.cpu()
        y_pred = y_pred.cpu()

        tp = torch.logical_and(y_pred == 1, y == 1).float().sum().item()
        tn = torch.logical_and(y_pred == 0, y == 0).float().sum().item()
        fn = torch.logical_and(y_pred == 1, y == 0).float().sum().item()
        fp = torch.logical_and(y_pred == 0, y == 1).float().sum().item()

        net.history.record_batch(prefix + '_tp_total', tp)
        net.history.record_batch(prefix + '_tn_total', tn)
        net.history.record_batch(prefix + '_fn_total', fn)
        net.history.record_batch(prefix + '_fp_total', fp)
        precision = tp / (tp + tn)  # type: ignore
        recall = tp / (tp + fn)  # type: ignore

        net.history.record_batch(prefix + '_f1',
                                 2 * precision * recall / (precision + recall))

    def on_epoch_end(self, net, dataset_train, dataset_valid):

        prefixes = []
        if dataset_train is not None and len(dataset_train) > 0:
            prefixes.append('train')
        if dataset_valid is not None and len(dataset_valid) > 0:
            prefixes.append('valid')

        for prefix in prefixes:
            tp = np.array(net.history[-1, 'batches', :, prefix + '_tp_total']
                          ).sum()
            tn = np.array(net.history[-1, 'batches', :, prefix + '_tn_total']
                          ).sum()
            fn = np.array(net.history[-1, 'batches', :, prefix + '_fn_total']
                          ).sum()
            fp = np.array(net.history[-1, 'batches', :, prefix + '_fp_total']
                          ).sum()
            precision = tp / (tp + tn)
            recall = tp / (tp + fn)

            net.history.record(prefix + '_f1',
                               2 * precision * recall / (precision + recall))
