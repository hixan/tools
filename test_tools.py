from tools.tools import batch


def test_batch():
    inpt = list(range(10))
    print(inpt)
    output = list(map(tuple, batch(inpt, 3)))
    print(output)
    assert output == [(0, 1, 2), (3, 4, 5), (6, 7, 8), (9,)]
