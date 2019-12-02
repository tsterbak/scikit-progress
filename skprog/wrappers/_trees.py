import numpy as np
from tqdm.notebook import tqdm_notebook
from copy import deepcopy
import warnings


class TreesProgressor(object):

    def __init__(self, clf):
        self.clf = deepcopy(clf)
        try:
            # for forest oob support
            self.oob = clf.oob_score
        except:
            self.oob = False
        try:
            # for multi-core support
            self.n_steps = int(self.clf.n_estimators / self.clf.n_jobs)
        except:
            self.n_steps = self.clf.n_estimators
        
    def fit(self, X, y=None, sample_weight=None):
        self.clf.set_params(warm_start=True)
        with tqdm_notebook(range(self.n_steps)) as pbar:
            for i in range(self.n_steps):
                self.clf.set_params(n_estimators=self.clf.n_jobs*(i + 1))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    r = self.clf.fit(X, y, sample_weight)
                if self.oob:
                    pbar.set_description("OOB Score: {:.1%}".format(self.clf.oob_score_))
                pbar.update(1)
        return r
        
    def predict(self, X):
        return self.clf.predict(X)
