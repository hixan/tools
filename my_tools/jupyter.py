import numpy as np
from matplotlib import pyplot as plt
from itertools import chain
import torch


class LivePlot:
    """Live updates to a plot in jupyter notebook.

    Usage (inside a jupyter notebook):

    %matplotlib notebook
    import time
    liveplot = LivePlot(3, labels=list('abc'))
    for i in range(10):
        time.sleep(.1)
        liveplot.update(
            np.random.rand(3)
        )
    """
    
    def __init__(self, nseries, series=None, labels=None, colors=None, xlab='x axis', ylab='', title=''):
        
        fig, ax = plt.subplots(1,1)
        
        if series is None:
            series = [[] for _ in range(nseries)]
            ax.set_ylim(0, 1)
        else:
            ax.set_ylim(np.min(series), np.max(series))
        
        if colors is not None:
            for s, c in zip(series, colors):
                ax.plot(s, c=c)
        else:
            for s in series:
                ax.plot(s)
        
        ax.set_xlim(-.1, len(series[0]))
        
        if labels is not None:
            ax.legend(labels)
        
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_title(title)
        
        self.series = series
        self.nseries = nseries
        self.ax = ax
        self.fig = fig

    def update(self, points=None):             
        
        # update series
        if points is not None:
            for s, p in zip(self.series, points):
                s.append(p)

            for line, s in zip(self.ax.lines, self.series):
                line.set_data(range(len(s)), s)
            
            nnone = list(y for y in chain.from_iterable(self.series) if y is not None)
            if len(nnone) == 0:
                miny = 0
                maxy = 1
            else:
                miny = min(nnone)
                maxy = max(nnone)
                
                
            # update limits
            self.ax.set_xlim(0, len(s) - 1)
            self.ax.set_ylim(miny, maxy)

        self.fig.canvas.draw()


def train_model(train, test, model, criterion, optimizer, nepochs=100,
        batched=False, device=None):
    """Train a pytorch model
    
    Plotting requires `%matplotlib notebook` command be run before call.

    :param train: Training dataset
        (iterable of batches if batched else iterable of data)
    :param test: Testing/Validation dataset (for plotting purposes only)
        same condition as with train
    :param model: Pytorch model
    :param criterion: Pytorch criterion object
    :param optimizer: Pytorch optimizer object
    :param nepochs: number of epochs to train until
    :param batched: flag - input data is iterable of batches if True, otherwise
        iterable of data points
    :param device: optional: Pytorch device object to run training on.
    """
    if device is None:
        # use GPU if available
        device = torch.device("cuda:0" if torch.cuda.is_available()
                              else "cpu")
        
        # move to appropriate device
        model.to(device)
    
    print(f'Training model \n{model}\n on device {device} for {nepochs} epochs.')
    
    lossplot = LivePlot(2)
    
    if not batched:
        ds_train = (train,)
        ds_test = (test,)
    else:
        ds_train = train
        ds_test = test

    overall_time = 0
    for epoch in range(nepochs + 1):
        
        with Timer('EpochTimer') as et:
            # evaluate test metrics
            loss_test = 0
            test_n = 0
            for X, Y in ds_test:
                X = X.float().to(device)
                Y = Y.long().to(device)
                pred = model(X)
                loss = criterion(pred, Y)

                loss_test += loss.item() * X.shape[0]
                test_n += X.shape[0]

            # train and evaluate train metrics
            loss_train = 0
            train_n = 0
            for X, Y in ds_train:
                X = X.float().to(device)
                Y = Y.long().to(device)
                pred = model(X)

                loss = criterion(pred, Y)

                loss_train += loss.item() * X.shape[0]
                train_n += X.shape[0]

                if epoch != nepochs:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            lossplot.update((loss_train / train_n, loss_test / test_n))
        taken = et.time_taken()
        overall_time += taken
        epochs_left = nepochs - epoch
        time_left = (overall_time / (epoch + 1)) * epochs_left
        print(f'epoch {epoch}, tr{loss_train / train_n:.2f} ts{loss_test / test_n:.2f} t{taken:.2f}s left {time_left / 60:.0f}m {time_left % 60:.0f}s', end='\r')
