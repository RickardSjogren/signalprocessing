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