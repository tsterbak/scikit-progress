import numpy as np
from tqdm.notebook import tqdm_notebook
from copy import deepcopy
import warnings


class SGDProgressor(object):

    def __init__(self, clf):
        self.clf = deepcopy(clf)
        self.max_iter = clf.max_iter
        
    def fit(self, X, y=None, sample_weight=None):
        self.classes = np.unique(y)
        for _ in tqdm_notebook(range(self.max_iter)):
            r = self.clf.partial_fit(X, y, self.classes, sample_weight)
        return r
    
    def __getattr__(self, name):
        def _missing(*args, **kwargs):
            try:
                return getattr(self.clf, name)(args[0])
            except:
                try:
                    return getattr(self.clf, name)(kwargs)
                except:
                    return getattr(self.clf, name)()
        return _missing


class GLMProgressor(object):

    def __init__(self, clf):
        self.clf = deepcopy(clf)
        self.max_iter = clf.max_iter
        
    def fit(self, X, y=None):
        self.clf.set_params(warm_start=True)
        with tqdm_notebook(range(self.max_iter)) as pbar:
            for i in range(self.max_iter):
                self.clf.set_params(max_iter=1)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    r = self.clf.fit(X, y)
                pbar.update(1)
        self.clf.set_params(warm_start=False, max_iter=self.max_iter)
        return r
        
    def predict(self, X):
        return self.clf.predict(X)
