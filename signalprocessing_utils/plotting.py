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


def plot_pca_controll_charts(pca, data, scores=None, residuals=None,
                             hotellings_t2=None, figsize=None, dpi=600,
                             **kwargs):
    """ Plot PCA-controll charts of fitted PCA-model.

    Parameters
    ----------
    pca : signalprocessing_utils.modelling.PCAPipeline
        Fitted PCA.
    data : pandas.DataFrame
        Data to transform and plot.
    scores : array_like, optional
        Projections of ´data´. If None, scores will be computed.
    residuals : array_like, NoneType, bool
        Observation-wise residual sum of squares of `data`. If None,
        residuals will not be plotted. If True, residual sum of squares
        will be computed from `data`.
    hotellings_t2 : array_like, NoneType, bool
        Observation-wise Hotelling's T2 of projected `data`. If None,
        Hotelling's T2 will not be plotted. If True, Hotelling's T2
        will be computed from `data`.
    figsize : tuple[float, float], optional
        Figure size in inches.

    Returns
    -------
    matplotlib.pyplot.Figure
    numpy.ndarray[matplotlib.pyplot.Axes]
    """
    if scores is None:
        scores = pca.transform(data)

    if not isinstance(residuals, np.ndarray) and residuals == True:
        residuals = pca.residual_sum_of_squares(data)
    if not isinstance(hotellings_t2, np.ndarray) and hotellings_t2 == True:
        hotellings_t2 = pca.hotellings_t2(scores)

    n_extra = sum([residuals is not None, hotellings_t2 is not None])
    f, axes = plt.subplots(pca.named_steps['pca'].n_components + n_extra, 1,
                           sharex=True, figsize=figsize, dpi=dpi)

    for i, (ax, score, m_score) in enumerate(zip(axes, scores.T, pca.fitted_scores.T),
                                             start=1):
        r2 = pca.named_steps['pca'].explained_variance_ratio_[i - 1]
        title = 'Component {} ({:.2f} %)'.format(i, r2 * 100)
        plot_1d_control_chart(score, m_score, ax, title=title, **kwargs)

    # Plot residuals.
    if residuals is not None:
        ax_i = -1 - int(hotellings_t2 is not None)
        plot_1d_control_chart(residuals, pca.fitted_residual_ss, axes[ax_i],
                              negative=False, title='Residual sum of squares',
                              **kwargs)

    if hotellings_t2 is not None:
        # Plot Hotelling's T2.
        fitted_t2 = pca.hotellings_t2(pca.fitted_scores)
        plot_1d_control_chart(hotellings_t2, fitted_t2, axes[-1],
                              negative=False, title='Hotelling\s T2',
                              **kwargs)

    axes[-1].set_xlim(0, len(scores))
    f.tight_layout()

    return f, axes


def plot_1d_control_chart(data, control_data, ax=None, sigma=6,
                          positive=True, negative=True, title=None,
                          legend=False, bad_color='red'):
    """ Plot a 1D control chart of `data` using control limits
    from `control_data`

    Parameters
    ----------
    data : array_like
        Data to plot.
    control_data : array_like
        Data to calculate controll limits for:
    ax : matplotlib.pyplot.Axes, optional
        Axis to plot at.
    sigma : int, float
        Number of standard deviation to use.
    positive : bool
        If True, draw positive control limit.
    negative : bool
        If True, draw negative control limit.
    title : str, optional
        If provided, set title of plot.
    legend : bool
        If True, draw legend describing control limits.
    bad_color : Any
        Valid matplotlib-color. Color used to color out-of-bounds regions.

    Returns
    -------
    matplotlib.pyplot.Figure
    matplotlib.pyplot.Axes
    """
    if ax is None:
        f, ax = plt.subplots()
    else:
        f = ax.figure
    mean = control_data.mean()
    std = control_data.std()

    upper = mean + sigma * std if positive else float('inf')
    lower = mean - sigma * std if negative else float('-inf')

    # Plot in-bounds data.
    mask = np.ma.masked_outside(data, upper, lower)
    ax.plot(mask)

    # If out-of-bounds data, plot with different color.
    if mask.mask.any():
        m = mask.mask

        # In order to be able to plot continuous time-plots, points
        # adjacent to out-of-bounds points must be located as well.
        m[:-1] = m[:-1] | m[1:]
        m[1:] = m[1:] | m[:-1]
        mask.mask = ~m
        ax.plot(mask, color=bad_color)

    ax.plot([0, len(data)], [mean, mean],
            color='red', linestyle='--', label=r'$\mu$')

    if title is not None:
        ax.set_title(title)

    sigma_label = r'$\mu {} {}\cdot\sigma$'.format(
        r'\pm' if positive and negative else '+' if positive else '-', sigma)
    if positive:
        ax.plot([0, len(data)], [upper, upper],
                color='green', linestyle='--', label=sigma_label)
    if negative:
        ax.plot([0, len(data)], [lower, lower], color='green', linestyle='--',
                label=sigma_label if not positive else None)

    if legend:
        ax.legend(loc=2)

    return f, ax