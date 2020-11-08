import numpy as np
from matplotlib import pyplot as plt
from itertools import chain

class LivePlot:
    
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
