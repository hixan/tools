from itertools import islice
# generator type tools


def stagger(iterable, offsets=(1,)):
    yield from zip(*(islice(iter(iterable), ofs, None) for ofs in (0, *offsets)))

