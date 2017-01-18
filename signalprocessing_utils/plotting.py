import numpy as np

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import colorbar
from matplotlib import colors


def plot_bwt_coefficients(coeffs, cmap='PRGn'):
    """ Plot Binary Wavelet Transform-spectrogram.

    coeffs : list[list[float]]
        List of lists with BWT-coefficients.
    cmap : str, matplotlib.colors.ColorMap
        Colormap to use.

    Returns
    -------
    matplotlib.pyplot.Figure
    np.ndarray[matplotlib.axes.Axes]
    """
    gs = gridspec.GridSpec(len(coeffs), 2, width_ratios=[20, 1])
    f = plt.figure()

    vmin, vmax = np.percentile(np.concatenate(coeffs), [.01, 99.99])
    axes = list()
    for i, c in enumerate(coeffs):
        ax = f.add_subplot(gs[i, 0])
        ax.pcolorfast(np.atleast_2d(c), cmap=cmap, vmin=vmin, vmax=vmax)

        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([.5])
        ax.set_yticklabels(['{}'.format(len(c))])

        axes.append(ax)

    # Add and draw colorbar.
    cb_ax = f.add_subplot(gs[:, 1])
    colorbar.ColorbarBase(cmap=cmap,
                          norm=colors.Normalize(vmin=vmin, vmax=vmax),
                          ax=cb_ax)
    cb_ax.yaxis.tick_right()
    axes.append(cb_ax)
    f.subplots_adjust(hspace=0)

    return f, np.array(axes)


def airgard_plot_and_spectrogram(spectrum, df, diff_n=100,
                                 figsize=(11, 12.6), dpi=800):
    """ Make plots of derivative of accelerometers, current and spectrogram.

    Parameters
    ----------
    spectrum : array_like
        Spectrum
    df : pandas.DataFrame
        Airgard-data frame with columns Cur, X, Y, Z
    diff_n : int
        Number of observations to use for accelerometer-derivative.
    figsize : tuple[float, float]
        Figure size in inches.
    dpi : int
        Figure DPI.

    Returns
    -------
    matplotlib.pyplot.Figure
    np.ndarray[matplotlib.pyplot.Axes]
    """
    assert len(spectrum) == len(df), 'Lengths does not match'
    f, (ax, ax1, ax2) = plt.subplots(3, 1, figsize=figsize,
                                     dpi=dpi, sharex=True)
    df.drop('Cur', 1).diff(diff_n).plot.line(ax=ax)
    df.Cur.plot.line(ax=ax1)
    ax.set_title('d(Accelerometer)')
    ax2.pcolorfast(spectrum.T, cmap='viridis')
    ax1.set_title('Cur')
    ax2.grid(False)
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_xlabel('Time')
    ax2.set_title('Spectrogram')
    f.tight_layout()

    return f, np.array([ax, ax1, ax2])


def plot_processed_airgard_df(df, line_cols, spectrum_cols, figure_kwargs=None,
                         pcolorfast_kwargs=None, to_db=False):

    n_plots = len(line_cols) + 1
    figure_kwargs = figure_kwargs if figure_kwargs is not None else dict()
    f, axes = plt.subplots(n_plots, 1, sharex=True,
                           **figure_kwargs)

    for ax, cols in zip(axes, line_cols):
        df[cols].plot.line(ax=ax)

    pcolorfast_kwargs = pcolorfast_kwargs if pcolorfast_kwargs is not None \
        else dict()
    if to_db:
        spectrum = 10 * np.log10(df[spectrum_cols]).T
    else:
        spectrum = df[spectrum_cols].T

    masked = np.ma.masked_where(np.isnan(spectrum), spectrum)
    cmap = pcolorfast_kwargs.pop('cmap', 'viridis')
    axes[-1].pcolorfast(masked, cmap=cmap, **pcolorfast_kwargs)
    axes[-1].grid(False)

    return f, axes


def plot_pca_controll_charts(pca, data, scores=None, figsize=None):
    """ Plot PCA-controll charts of fitted PCA-model.

    Parameters
    ----------
    pca : signalprocessing_utils.modelling.PCAPipeline
        Fitted PCA.
    data : pandas.DataFrame
        Data to transform and plot.
    scores : array_like, optional
        Projections of ´data´.
    figsize : tuple[float, float], optional
        Figure size.

    Returns
    -------
    matplotlib.pyplot.Figure
    numpy.ndarray[matplotlib.pyplot.Axes]
    """
    if data is None and scores is None:
        raise ValueError('Either data or scores must be provided.')
    if scores is None:
        scores = pca.transform(data)

    f, axes = plt.subplots(pca.named_steps['pca'].n_components + 2, 1,
                           sharex=True, figsize=figsize)

    for i, (ax, score, m_score) in enumerate(zip(axes, scores.T, pca.fitted_scores.T),
                                             start=1):
        s = m_score.std()
        mu = m_score.mean()

        ax.plot(score)
        ax.axhline(mu, linestyle='--', color='red')
        ax.axhline(mu - 6 * s, linestyle='--', color='green')
        ax.axhline(mu + 6 * s, linestyle='--', color='green')
        r2 = pca.named_steps['pca'].explained_variance_ratio_[i - 1]
        ax.set_title('Component {} ({:.2f} %)'.format(i, r2 * 100))

    axes[-2].plot(pca.residual_sum_of_squares(data))
    axes[-2].set_title('Residuals')

    # Plot residuals.
    res_mu = pca.fitted_residual_ss.mean()
    res_s = pca.fitted_residual_ss.std()
    axes[-2].axhline(res_mu, linestyle='--', color='red')
    axes[-2].axhline(res_mu + 6 * res_s, linestyle='--', color='green')

    # Plot Hotelling's T2.
    fitted_t2 = pca.hotellings_t2(pca.fitted_scores)
    t2_mu = fitted_t2.mean()
    t2_s = fitted_t2.std()
    axes[-1].plot(pca.hotellings_t2(scores))
    axes[-1].set_title('Hotelling\'s T2')
    axes[-1].axhline(t2_mu, linestyle='--', color='red')
    axes[-1].axhline(t2_mu + 6 * t2_s, linestyle='--', color='green')
    return f, axes