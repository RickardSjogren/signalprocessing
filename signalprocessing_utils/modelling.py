import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn import pipeline


class PCAPipeline(pipeline.Pipeline):

    """ Data scaling and Principal Component Analysis (PCA).

    This class encapsulates a simple pipeline consisting
    of data scaling and PCA-fitting. It also extends the
    Scikit-learn PCA-class with calculation of Hotelling's T2
    statistics, model residuals and observation/variable-wise
    residual sums of squares.

    Parameters
    ----------
    n_components : int
        Number of components to use.
    mean_center : bool (default True)
        If True, mean center data columns.
    uv_scale : bool (default True)
        If True, scale data columns to equal variance.

    Attributes
    ----------
    fitted_scores : array, [n_observations, n_components]
        Scores of observations used to fit model.
    fitted_residual_ss : array, [n_observations]
        Observation-wise residual sum of squares of
        observations used to fit model.
    """

    def __init__(self, n_components, mean_center=True, uv_scale=True):
        super(PCAPipeline, self).__init__([
            ('scale', preprocessing.StandardScaler(with_mean=mean_center,
                                                   with_std=uv_scale)),
            ('pca', PCA(n_components))
        ])
        self.fitted_scores = None
        self.fitted_residual_ss = None
        self._variable_names = None

    def fit_transform(self, X, y=None, **fit_params):
        scores = super(PCAPipeline, self).fit_transform(X, y, **fit_params)
        self.fitted_scores = scores
        self.fitted_residual_ss = self.residual_sum_of_squares(X, scores)
        try:
            self._variable_names = X.columns
        except AttributeError:
            pass

        col_names = ['t[{}]'.format(i) for i, _ in enumerate(scores.T, 1)]
        return _make_df(scores, X, columns=col_names)

    def _transform(self, X):
        scores = super(PCAPipeline, self)._transform(X)
        col_names = ['t[{}]'.format(i) for i, _ in enumerate(scores.T, 1)]
        return _make_df(scores, X, columns=col_names)

    def fit(self, X, y=None, **fit_params):
        super(PCAPipeline, self).fit(X, y, **fit_params)
        self.fitted_scores = self.transform(X)
        self.fitted_residual_ss = self.residual_sum_of_squares(X, self.fitted_scores)

        try:
            self._variable_names = X.columns
        except AttributeError:
            pass

        return self

    @property
    def components_(self):
        components = self.named_steps['pca'].components_
        if self._variable_names is not None:
            row_names = ['p[{}]'.format(i) for i, _ in enumerate(components, 1)]
            return _make_df(components, None, index=row_names,
                            columns=self._variable_names)
        else:
            return components

    def hotellings_t2(self, scores):
        """ Calculate observation-wise Hotellings T2 of PCA projections.

        Parameters
        ----------
        scores : array
            PCA projections.

        Returns
        -------
        t2 : array
            Observation-wise Hotellings T2.
        """
        pca_ = self.named_steps['pca']
        loadings = pca_.components_
        X_hat = pca_.inverse_transform(scores)
        R2 = pca_.explained_variance_

        cov = loadings.T.dot(np.diag(R2)).dot(loadings) / X_hat.shape[1] - 1
        w = np.linalg.solve(cov, X_hat.T)
        t2 = (X_hat.T * w).sum(axis=0)
        return _make_series(t2, scores)

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
        array
            Model residuals.
        """

        if scores is None:
            scores = self.transform(data)

        predicted_data = self.inverse_transform(scores)
        return _make_df(data - predicted_data, data)

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
        array
            Model residual sum of squares.
        """
        residuals = self.residuals(data, scores)
        return _make_series((residuals **2).sum(axis=axis), data)


def _make_df(array, template, index=True, columns=True, **kwargs):
    """ Turn `array` into DataFrame given `template`

    Parameters
    ----------
    array : array
        Input data.
    template : pandas.DataFrame, array
        If DataFrame, use as template for making `array` into DataFrame.
        If array, return `array`
    index : bool, list[Any], NoneType
        If True, copy rows. If sequence use as index. If None, do nothing.
    columns : bool, list[Any], NoneType
        If True, copy columns. If sequence, use as column index.
        If None, do nothing.
    **kwargs
        Keyword arguments passed to `pandas.DataFrame`-constructor.

    Returns
    -------
    pandas.DataFrame, array
    """
    if not isinstance(template, pd.DataFrame) \
            and isinstance(index, (bool, type(None))) \
            and isinstance(columns, (bool, type(None))):
        return array

    if isinstance(index, bool) and index == True:
        index = template.index
    elif index is not None:
        index = index
    else:
        index = None

    if isinstance(columns, bool) and columns == True:
        columns = template.columns
    elif columns is not None:
        columns = columns
    else:
        columns = None

    return pd.DataFrame(array, index=index, columns=columns, **kwargs)


def _make_series(array, template, index=True, **kwargs):
    """ Turn `array` into Series given `template`

    Parameters
    ----------
    array : array
        Input data.
    template : pandas.Series, array
        If DataFrame, use as template for making `array` into DataFrame.
        If array, return `array`
    index : bool, list[Any], NoneType
        If True, copy rows. If sequence use as index. If None, do nothing.
    **kwargs
        Keyword arguments passed to `pandas.Series`-constructor.

    Returns
    -------
    pandas.Series, array
    """
    if not isinstance(template, pd.Series) and isinstance(index, bool):
        return array

    if index == True:
        index = template.index
    elif index:
        index = index
    else:
        index = None

    return pd.Series(array, index=index, **kwargs)