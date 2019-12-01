import numpy as np
import pymc3 as pm
import theano
import scipy.stats as st

class StudentsT:

    def __init__(self, nu=5, index=None, columns=None):
        self.nu = nu
        self.index = index
        self.columns = columns

    def model(self):
        with pm.Model() as model:
            mu = pm.Normal('mu', mu=0, sd=10)
            sigma = pm.HalfCauchy('sigma', 10, testval=.1)
            obs = pm.StudentT('obs', nu=self.nu, mu=mu, sigma=sigma, observed=self.obs)
        return model

    def fit(self, data):
        if self.index is None and self.columns is None:
            self.df = data.val.to_frame()
        else:
            self.df = data.pivot(index=self.index, columns=self.columns, values='val')

        self.obs = theano.shared(np.zeros(self.df.shape[0]), borrow=True)
        model = self.model()
        params = {}
        for col in self.df.columns:
            self.obs.set_value(self.df[col].values)
            params[col] = pm.find_MAP(model=model)
        self.params = params

    def score(self, data):
        if self.index is None and self.columns is None:
            df = data.val.to_frame()
        else:
            df = data.pivot(self.index, columns=self.columns, values='val')

        loss = np.array(
            [-st.t.logpdf(df[col], self.nu, self.params[col]['mu'], self.params[col]['sigma']) for col in df.columns]
        ).T
        return loss

    def score2(self, data):
        if self.index is None and self.columns is None:
            df = data.val.to_frame()
        else:
            df = data.pivot(self.index, columns=self.columns, values='val')

        loss = np.array(
            [st.t.cdf(df[col], self.nu, self.params[col]['mu'], self.params[col]['sigma']) for col in df.columns]
        ).T
        return loss