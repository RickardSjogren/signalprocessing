import matplotlib.pyplot as plt
import numpy as np
from matplotlib import dates, pyplot
from matplotlib import colorbar
from matplotlib import colors
from matplotlib import gridspec

from signalprocessing_utils.misc import iter_cols


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


def make_ax_timespan(ax, datetimes, resolution=None):
    """ Make `ax` X-axis a time-axis.

    In order to function properly `ax.figure.autofmt_xdate` must
    be called after plot is finished.

    Parameters
    ----------
    ax : matplotlib.pyplot.Axes
        Axis-instance to adjust.
    datetimes : list[datetime.DateTime]
        Datetime-index used.
    resolution : str {'month', 'week', 'day', 'hour'}, optional
        Specified resolution. If None, automatic formatting is used.

    Returns
    -------
    np.ndarray
        `datetimes` converted to equispaced numeric array.
    """
    times = dates.date2num(datetimes)
    if resolution == 'month':
        locator = dates.MonthLocator()
    elif resolution == 'week':
        locator = dates.WeekdayLocator()
    elif resolution == 'day':
        locator = dates.DayLocator()
    elif resolution == 'hour':
        locator = dates.HourLocator()
    elif isinstance(resolution, dates.DateLocator):
        locator = resolution
    elif resolution is None:
        locator = dates.AutoDateLocator()

    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(dates.AutoDateFormatter(locator))
    ax.set_xlim([times[0], times[-1]])

    return times


def plot_pca_controll_charts(pca, scores, residuals, x_index=None, axes=None,
                             figsize=None, dpi=600, **kwargs):
    """ Plot PCA-controll charts of fitted PCA-model.

    Parameters
    ----------
    pca : signalprocessing_utils.modelling.PCAPipeline
        Fitted PCA.
    scores : array_like
        Projections of ´data´. If None, scores will be computed.
    residuals : array_like
        Observation-wise residual sum of squares of `data`. If None,
        residuals will be calulcated from data.
    x_index : array
        If provided, `x_index` is used as x-axis index for all plots.
    axes : list[matplotlib.pyplot.Axes], optional
        Sequence of axis-object to plot at.
    figsize : tuple[float, float], optional
        Figure size in inches.

    Returns
    -------
    matplotlib.pyplot.Figure
    numpy.ndarray[matplotlib.pyplot.Axes]
    """
    if axes is None:
        f, axes = plt.subplots(pca.named_steps['pca'].n_components + 1, 1,
                               sharex=True, figsize=figsize, dpi=dpi)
    else:
        f = axes[0].figure

    for i, (ax, score, m_score) in enumerate(zip(axes,
                                                 iter_cols(scores),
                                                 iter_cols(pca.fitted_scores)),
                                             start=1):
        r2 = pca.named_steps['pca'].explained_variance_ratio_[i - 1]
        title = 'Component {} ({:.2f} %)'.format(i, r2 * 100)
        plot_1d_control_chart(score, m_score, x_index,
                              ax=ax, title=title, **kwargs)

    # Plot residuals.
    if residuals is not None:
        plot_1d_control_chart(residuals, pca.fitted_residual_ss, x_index, ax=axes[-1],
                              negative=False, title='Residual sum of squares',
                              **kwargs)

    return f, axes


def plot_1d_control_chart(data, control_data, index=None, ax=None, sigma=6,
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

    if index is None:
        index = np.arange(len(data))

    mean = control_data.mean()
    std = control_data.std()

    upper = mean + sigma * std if positive else float('inf')
    lower = mean - sigma * std if negative else float('-inf')

    # Plot in-bounds data.
    mask = np.ma.masked_outside(data, upper, lower)
    ax.plot(index, mask)

    # If out-of-bounds data, plot with different color.
    if mask.mask.any():
        m = mask.mask

        # In order to be able to plot continuous time-plots, points
        # adjacent to out-of-bounds points must be located as well.
        m[:-1] = m[:-1] | m[1:]
        m[1:] = m[1:] | m[:-1]
        mask.mask = ~m
        ax.plot(index, mask, color=bad_color)

    ax.plot([index[0], index[-1]], [mean, mean],
            color='red', linestyle='--', label=r'$\mu$')

    if title is not None:
        ax.set_title(title)

    sigma_label = r'$\mu {} {}\cdot\sigma$'.format(
        r'\pm' if positive and negative else '+' if positive else '-', sigma)
    if positive:
        ax.plot([index[0], index[-1]], [upper, upper],
                color='green', linestyle='--', label=sigma_label)
    if negative:
        ax.plot([index[0], index[-1]], [lower, lower], color='green', linestyle='--',
                label=sigma_label if not positive else None)

    if legend:
        ax.legend(loc=2)

    return f, ax


def timeseries_heatmap(data, ax, n_points=1200, **kwargs):
    """ Make time-series heatmap.

    Parameters
    ----------
    data : pandas.DataFrame
        Data-frame with `DateTime`-index.
    ax : matplotlib.pyplot.Axes
        Axis instance to plot at.
    n_points : int
        Number of points to down sample data-frame to. If None,
        no down-sampling is performed.
    **kwargs
        Key-word arguments passed to `ax.pcolormesh`

    Returns
    -------
    matplotlib.collections.QuadMesh
        Heatmap quad-mesh.
    """

    if n_points is not None:
        timespan = data.index[-1] - data.index[0]
        minutes = max([1, int((timespan.total_seconds() / 60) / n_points)])
        downsampled = data.resample('{}T'.format(minutes)).mean()
    else:
        downsampled = data

    mask = np.ma.masked_where(np.isnan(downsampled), downsampled)
    y = np.arange(0, data.shape[1])
    index = dates.date2num(downsampled.index.to_pydatetime())
    heatmap = ax.pcolormesh(index, y, mask.T, **kwargs)
    ax.set_ylim([y[0], y[-1]])
    return heatmap