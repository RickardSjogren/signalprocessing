import collections
from datetime import timedelta

import numpy as np
import pandas as pd
from matplotlib import dates
from matplotlib import pyplot as plt

from signalprocessing_utils import modelling
from signalprocessing_utils import plotting
from signalprocessing_utils.misc import chunk_df_on_diff
from signalprocessing_utils.preprocessing import _partial_w_name, fft_amplitude


def process_airgard_df(df, start_day, spectral_transform=None,
                       spatial_transform=None, current_transform=None):
    """ Process a single day of Airgard-data store as consecutive
    dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Data to process.
    spectral_transform : Callable, optional
        Function to do spectral processing of 1D-sound. If output has more
        than one dimension the column-wise average will be used. Defaults
        to: :py:`stfft_autopower`.
    spatial_transform : list[Callable], optional
        List of transforms to perform on accelerometer (X Y Z) readings.
        Defaults to average and standard-deviation.
    current_transform : list[Callable], optional
        List of transform to perform on current (Cur) readings. Defaults
        to average and standard-deviation.

    Returns
    -------
    pandas.DataFrame
        Data-chunks in rows and processed variables as columns.
    """
    if spectral_transform is None:
        spectral_transform = _partial_w_name(fft_amplitude, funcname='Z',
                                             N=5000, as_db=True)
    if spatial_transform is None:
        spatial_transform = [
            _partial_w_name(np.mean, axis=0),
            _partial_w_name(np.std, axis=0),
        ]
    if current_transform is None:
        current_transform = [
            _partial_w_name(np.mean, axis=0),
            _partial_w_name(np.std, axis=0)
        ]

    processed = list()
    current_time = start_day
    index = list()
    for chunk, gap in chunk_df_on_diff(df, 'Time', .003):
        index.append(current_time)
        delta = timedelta(seconds=gap)
        current_time += delta
        if chunk is None:
            processed.append([None, None, None])
            continue

        spectrum = spectral_transform(chunk.Mic)
        if spectrum.ndim > 1:
            spectrum = np.mean(spectrum, axis=0)

        spatial = [transform(chunk.drop(['Time', 'Mic', 'Cur'], 1))
                   for transform in spatial_transform]
        current = [transform(chunk['Cur']) for transform in current_transform]

        processed.append((np.concatenate([np.array(s) for s in spatial]),
                          current, spectrum))

    cols = [c + '_' + t.__name__ for t in spatial_transform
            for c in ['X', 'Y', 'Z']]
    cols += ['Cur' + '_' + t.__name__ for t in current_transform]
    cols += ['{}{}'.format(spectral_transform.__name__, i) for
             i, _ in enumerate(processed[0][2], start=1)]

    processed_arr = np.empty((len(processed), len(cols)),
                             dtype=float)

    for i, (spatial, current, spectrum) in enumerate(processed):
        if spatial is None:
            processed_arr[i] = np.nan
            continue

        processed_arr[i, :len(spatial)] = spatial
        processed_arr[i, len(spatial):len(spatial) + len(current)] = current
        processed_arr[i, len(spatial) + len(current):] = spectrum

    processed_df = pd.DataFrame(processed_arr, columns=cols, index=index)
    return processed_df


def fit_and_plot(data, sample_end, n_components=5, only_mahal=False,
                 verbose=True, copy=True, **kwargs):
    """ Fit an n-component PCA-model of subset of processed
     Airgard-data and plot control charts.

    Parameters
    ----------
    data : pandas.DataFrame
        Data to modell and plot.
    sample_end : Any
        Index of sample end observation
    n_components : int
        Number of PCA-components.
    verbose : bool
        If True, print process.

    Returns
    -------
    matplotlib.pyplot.Figure
    numpy.ndarray[matplotlib.pyplot.Axes]
    """
    index = pd.DataFrame(index=data.index)

    if copy:
        data = data.dropna(inplace=False)
    else:
        data.dropna(inplace=True)

    subsample = data[:sample_end]
    if verbose:
        print('Fit {}-component PCA-model'.format(n_components))

    pca = modelling.PCAPipeline(n_components=n_components).fit(subsample)
    if verbose:
        print('Project all observations.')
    scores = pca.transform(data)

    if verbose:
        print('Calculate observation residuals.')

    rss = pca.residual_sum_of_squares(data, scores)

    if verbose:
        print('Joining back missing values.')
    scores = index.join(scores)
    rss = index.join(rss)

    print('Plot results.')
    n_plots = (n_components if not only_mahal else 0) + 2
    f, axes = plt.subplots(n_plots, 1, sharex=True, figsize=(10, 2 * n_plots))
    times = dates.date2num(index.index.to_pydatetime())

    plotting.plot_pca_controll_charts(pca, scores, rss, times,
                                      only_mahal=only_mahal,
                                      axes=axes, sigma=6, **kwargs)

    for ax in axes:
        ax.axvline(times[len(subsample)], linestyle=':', color=(.2, .2, .2))

    plotting.make_ax_timespan(axes[-1], index.index.to_pydatetime())
    f.tight_layout()

    return f, axes


def collapse_spectrum(data, n=10):
    """ Group spectrum `n` by `n` by averaging and return copy
    of `data` with spectral columns replaced by collapsed.

    Parameters
    ----------
    data : pandas.DataFrame
        Input data.
    n : int
        Number of frequencies in each group.

    Returns
    -------
    pandas.DataFrame
    """
    z_cols = [c for c in data.columns if c[0] == 'Z' and c[1] != '_']
    n_cols = len(z_cols)
    spectrum = data[z_cols]

    data = data.drop(z_cols, axis=1)
    collapsed_columns = collections.OrderedDict(
            [('Z{}-{}'.format(i + 1, i + n), spectrum[z_cols[i:i + n]].mean(1))
             for i in range(0, n_cols, n)]
    )
    binned_spectrum = pd.DataFrame(collapsed_columns)

    data = pd.concat((data, binned_spectrum), axis=1)
    return data


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