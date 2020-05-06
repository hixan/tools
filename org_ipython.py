import os
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from collections import namedtuple

_metavals = {'container': Path('./.images')}


def set_default_container(container: Path):
    _metavals['container'] = container


def showplot_plt(name: str = None, fmt: str = 'png', container: Path = None):
    ''' saves a matplotlib plot and prints (and returns) an org-friendly link
    to the image. '''
    if container is None:
        container = _metavals['container'] / Path('plt/')

    # create directory structure for output images
    if not container.exists():
        container.mkdir(parents=True)

    # create filename based on the time
    if name is None:
        name = datetime.now().strftime('plot_at_%Y%m%d-%H%M%S%f')

    filepath = container / f'{name}.{fmt}'

    # save the figure to file
    plt.savefig(filepath)
    plt.clf()  # clear the output as it would with plt.show()

    # return a string that orgmode recognises as a link to an image
    rv = f'[[file:{str(filepath)}]]'
    # print it out for output mode.
    print(rv)
    # return it for value mode.
    return rv
