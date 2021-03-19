from itertools import islice
# generator type tools


def stagger(iterable, offsets=(1,)):
    '''stagger iterable with offsets offsets.
    EG:
    stagger((1, 2, 4, 8, 16), offsets=(1, 3))
    will yield
    (1, 2, 8),
    (2, 4, 16)
     ^  ^  ^- third column offset by 3
     |  |---- second column offset by 1
     |------- first column offset by 0
    '''
    return zip(*(islice(iter(iterable), ofs, None) for ofs in (0, *offsets)))

