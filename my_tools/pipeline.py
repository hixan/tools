from sklearn.base import BaseEstimator, TransformerMixin
import cv2
import numpy as np
import face_recognition
from pykalman import KalmanFilter
from sklearn.pipeline import Pipeline


class IdentityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._repr_start = f'{type(self).__name__}('

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X

    def __repr__(self):
        return self._repr_start + '??)'


class ImageRescale(IdentityTransformer):

    def __init__(self, x_factor: float, y_factor: float = None):
        '''resizes images by x and y factors. If y factor is None, resizes both by x factor.'''
        self.x_factor = x_factor
        if y_factor:
            self.y_factor = float(y_factor)
        else:
            self.y_factor = float(x_factor)

    def transform(self, X, y=None):
        return cv2.resize(X, (0, 0), fx=self.x_factor, fy=self.y_factor)

    def __repr__(self):
        if self.y_factor == self.x_factor:
            return f'{self._repr_start})'
        else:
            return f'{self._repr_start}, {self.y_factor})'


class ImageResize(ImageRescale):

    def __init__(self, width, height=None):
        '''rescales images to be width x height. If height is None, makes width x width.'''
        super(ImageResize, self).__init__(width, height)

    def transform(self, X, y=None):
        return cv2.resize(X, (self.x_factor, self.y_factor))


class VectorRescale(BaseEstimator, TransformerMixin):

    def __init__(self, rescale_vec):
        """rescales vectors by element-wise multiplication"""
        self.rescale_vec = np.array(rescale_vec)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # element-wise multiplication
        return X * self.rescale_vec

    def __repr__(self):
        return f'{type(self).__name__}({self.rescale_vec})'


class Image2Greyscale(IdentityTransformer):

    def transform(self, X, y=None):
        return cv2.cvtColor(X, cv2.COLOR_BGR2GRAY)


class FacialFeatureExtraction(IdentityTransformer):
    def transform(self, X, y=None):
        return face_recognition.face_landmarks(X)


class FastFacialFeatures(IdentityTransformer):

    def __init__(self, scaling: float, greyscale: bool = False):
        self._scaling = float(scaling)
        self._greyscale = bool(greyscale)
        prep_plne = [
            ('rescale', ImageRescale(self._scaling)),
            ]
        if self._greyscale:
            prep_plne.append(('grey', Image2Greyscale()))
        self.pipeline = Pipeline(steps=prep_plne)

        self.unpack = VectorRescale(np.array([1/self._scaling] * 2))

    def transform(self, X, y=None):
        rv = []
        for landmarks in face_recognition.face_landmarks(self.pipeline.transform(X)):
            curr = {k: self.unpack.transform(np.array(landmarks[k])) for k in landmarks}
            rv.append(curr)
        return rv

    def __repr__(self):
        return f'{self._repr_start}{self._scaling}, {self._greyscale})'

class FittedKalmanFilter(IdentityTransformer):

    def __init__(self):
        super(FittedKalmanFilter, self).__init__()
        self._variance = 1
        self._mean = 0

    def fit(self, X, y=None):
        self._mean = np.mean(X, dim=0)
        self._variance = np.var(X, axis=0)

        
