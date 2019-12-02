from ._base import BaseProgressor

import warnings
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


class TreesProgressor(BaseProgressor):

    def __init__(self, clf):
        super().__init__(clf)
        try:
            # for forest oob support
            self.oob = self.clf.oob_score
        except:
            self.oob = False
        try:
            # for multi-core support
            self.n_steps = int(self.clf.n_estimators / self.clf.n_jobs)
        except:
            if self.clf.n_estimators == "warn":
                self.n_steps = 10
            else:
                self.n_steps = self.clf.n_estimators
        if not self.clf.n_jobs:
            self.n_jobs = 1

    def fit(self, X, y=None, sample_weight=None):
        self.clf.set_params(warm_start=True)
        with tqdm(range(self.n_steps)) as pbar:
            for i in range(self.n_steps):
                self.clf.set_params(n_estimators=self.n_jobs*(i + 1))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    r = self.clf.fit(X, y, sample_weight)
                if self.oob:
                    pbar.set_description("OOB Score: {:.1%}"
                                         .format(self.clf.oob_score_))
                pbar.update(1)
        return r
