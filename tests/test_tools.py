import numpy as np
from my_tools.tools import var_covar_matrix

np.set_printoptions(precision=2)


def test_var_covar_matrix():

    expected = np.array([
        [1, .2, .7],
        [.2, 1, 0],
        [.7, 0, 1],
    ])
    # check positive semi-definite
    assert np.all(np.linalg.eigvals(expected) >= 0)
    # check symmetric
    assert np.all(expected == expected.T)
    data = np.random.multivariate_normal(np.random.rand(expected.shape[0]) * 10, expected, 5000)
    print('expected value')
    print(expected)
    print()
    print('calculated value')
    got = var_covar_matrix(data) 
    print(got)
    print()
    print('difference')
    diff = got - expected
    print(diff)
    assert np.all(diff < .05)

    # calculate on other axis
    got = var_covar_matrix(data.T, axis=1)

def test_confusion_matrix():
    Y = [1, 2, 3, 2, 3]
    p = [1, 3, 3, 2, 3]
    m = confusion_matrix(Y, p, labels=[3, 1, 2])
    print(m)
    assert np.all(m == np.array([[2, 0, 1],
                                 [0, 1, 0],
                                 [0, 0, 1]]))




