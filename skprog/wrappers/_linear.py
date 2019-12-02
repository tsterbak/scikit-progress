from ._base import BaseProgressor

import warnings
import numpy as np
try:
    shell = get_ipython().__class__.__name__
    if shell == 'ZMQInteractiveShell':
        from tqdm.notebook import tqdm_notebook as tqdm
    elif shell == 'TerminalInteractiveShell':
        from tqdm import tqdm
    else:
        from tqdm import tqdm
except NameError:
    from tqdm.tqdm import tqdm


class SGDProgressor(BaseProgressor):

    def __init__(self, clf):
        super().__init__(clf)
        self.max_iter = self.clf.max_iter

    def fit(self, X, y=None, sample_weight=None):
        self.classes = np.unique(y)
        for _ in tqdm(range(self.max_iter)):
            r = self.clf.partial_fit(X, y, self.classes, sample_weight)
        return r


class GLMProgressor(BaseProgressor):

    def __init__(self, clf):
        super().__init__(clf)
        self.max_iter = self.clf.max_iter

    def fit(self, X, y=None):
        self.clf.set_params(warm_start=True)
        with tqdm(range(self.max_iter)) as pbar:
            for i in range(self.max_iter):
                self.clf.set_params(max_iter=1)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    r = self.clf.fit(X, y)
                pbar.update(1)
        self.clf.set_params(warm_start=False, max_iter=self.max_iter)
        return r
