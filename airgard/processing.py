import collections
from datetime import timedelta

import numpy as np
import pandas as pd

from signalprocessing_utils.misc import chunk_df_on_diff
from signalprocessing_utils.preprocessing import _partial_w_name, fft_amplitude
from signalprocessing_utils import modelling
from signalprocessing_utils import plotting


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


def fit_and_plot(data, sample_start, sample_end, n_components=5, verbose=True):
    """ Fit an n-component PCA-model of subset of processed
     Airgard-data and plot control charts.

    Parameters
    ----------
    data : pandas.DataFrame
        Data to modell and plot.
    sample_start : Any
        Index of start observation.
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
    data = data[~np.isnan(data).any(1)]
    week_data = data[sample_start: sample_end]
    if verbose:
        print('Fit {}-component PCA-model'.format(n_components))

    pca = modelling.PCAPipeline(n_components=n_components).fit(week_data)
    if verbose:
        print('Project all observations.')
    scores = pca.transform(data[sample_start:])

    if verbose:
        print('Calculate observation residuals.')
    residuals = pca.residual_sum_of_squares(data[sample_start:], scores)

    if verbose:
        print('Calculate Hotelling\'s T2')
    hotellings_t2 = pca.hotellings_t2(scores)

    print('Plot results.')
    f, axes = plotting.plot_pca_controll_charts(pca, None, scores,
                                                residuals.values, hotellings_t2,
                                                figsize=(10, 12), sigma=6)
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