class BaseProgressor(object):

    def __init__(self, clf):
        self.clf = clf

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

    def get_params(self):
        return self.clf.get_params()

    def set_params(self, **kwargs):
        return self.clf.set_params(kwargs)
