import numpy as np

from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn import pipeline


class PCAPipeline(pipeline.Pipeline):

    def __init__(self, n_components, mean_center=True, uv_scale=True):
        super(PCAPipeline, self).__init__([
            ('scale', preprocessing.StandardScaler(with_mean=mean_center,
                                                   with_std=uv_scale)),
            ('pca', PCA(n_components))
        ])
        self.fitted_scores = None
        self.fitted_residual_ss = None

    def fit_transform(self, X, y=None, **fit_params):
        scores = super(PCAPipeline, self).fit_transform(X, y, **fit_params)
        self.fitted_scores = scores
        self.fitted_residual_ss = self.residual_sum_of_squares(X, scores)
        return scores

    def fit(self, X, y=None, **fit_params):
        super(PCAPipeline, self).fit(X, y, **fit_params)
        self.fitted_scores = self.transform(X)
        self.fitted_residual_ss = self.residual_sum_of_squares(X, self.fitted_scores)
        return self

    @property
    def components_(self):
        return self.named_steps['pca'].components_

    def hotellings_t2(self, scores):
        """ Calculate observation-wise Hotellings T2 of PCA projections.

        Parameters
        ----------
        scores : array_like
            PCA projections.

        Returns
        -------
        t2 : array_like
            Observation-wise Hotellings T2.
        """
        pca_ = self.named_steps['pca']
        loadings = pca_.components_
        X_hat = pca_.inverse_transform(scores)
        R2 = pca_.explained_variance_

        cov = loadings.T.dot(np.diag(R2)).dot(loadings) / X_hat.shape[1] - 1
        w = np.linalg.solve(cov, X_hat.T)
        t2 = (X_hat.T * w).sum(axis=0)
        return t2

    def residuals(self, data, scores=None):
        """ Calculate model residuals for `data`.

        Parameters
        ----------
        data : array_like
            Input data.
        scores : array_like, optional
            Projections if `data`

        Returns
        -------
        array_like
            Model residuals.
        """

        if scores is None:
            scores = self.transform(data)

        predicted_data = self.inverse_transform(scores)
        return data - predicted_data

    def residual_sum_of_squares(self, data, scores=None, axis=1):
        """ Calculate residual sum of squares.

        Parameters
        ----------
        data : array_like
            Input data.
        scores : array_like, optional
            Projections of `data`.
        axis : int
            Array axis index (default 1 / row-wise).

        Returns
        -------
        array_like
            Model residual sum of squares.
        """
        residuals = self.residuals(data, scores)
        return (residuals **2).sum(axis=axis)