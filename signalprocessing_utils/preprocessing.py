import functools

import numpy as np
import pandas as pd


def stfft_autopower(data, fft_size, overlap_fac=0.5, clip=None):
    """ Get autopower spectrogram using Short-Term Fast Fourier Transform
    using Hanning sliding windows.

    Adopted from.
    https://kevinsprojects.wordpress.com/2014/12/13/short-time-fourier-transform-using-python-and-numpy/

    Parameters
    ----------
    data : array_like
        A numpy array containing the signal to be processed
    fft_size : int
        Number of samples in the Fourier transform.
    overlap_fac : float
        To which extent windows will overlap.
    clip : tuple, optional
        If provided, tuple of minimum and maximum value to clip resulting
        spectrum with (in dB).

    Returns
    -------
    spectrogram : array_like
    """
    hop_size = int(np.floor(fft_size * (1 - overlap_fac)))
    total_segments = int(np.ceil(len(data) / hop_size))

    window = np.hanning(fft_size)
    inner_pad = np.zeros(fft_size)

    # Pad end of data with zeros.
    proc = np.concatenate((data, np.zeros(fft_size)))
    spectrogram = np.empty((total_segments, fft_size), dtype=data.dtype)

    for i in range(total_segments):
        current_hop = hop_size * i
        segment = proc[current_hop:current_hop + fft_size]

        # Multiply with Hanning window and zero-pad.
        windowed = segment * window
        padded = np.append(windowed, inner_pad)

        spectrum = np.fft.fft(padded) / fft_size
        autopower = np.abs(spectrum * np.conj(spectrum))
        spectrogram[i, :] = autopower[:fft_size]

    # Scale to dB.
    spectrogram = 20 * np.log10(spectrogram)

    if clip is not None:
        spectrogram = np.clip(spectrogram, *clip)

    return spectrogram


def fft_amplitude(signal, N, as_db=False):
    """ Perform FFT and return amplitude and frequencies.

    Maximum frequency is `len(signal) // 2` according to
    Shannon-Nyqvist sampling theorem.

    Parameters
    ----------
    signal : array_like
        1D-signal.
    N : int, float
        Number of frames per second.
    as_db : bool
        If True, convert amplitudes to dB before return.

    Returns
    -------
    amplitudes : np.ndarray[float]
        Frequency amplitudes of signal.
    frequencies : np.ndarray[float]
        Frequencies retrieved.
    """
    fft = np.fft.fft(signal, N)
    amplitudes = 2 / N * np.abs(fft[:N // 2])

    if as_db:
        amplitudes = 10 * np.log10(amplitudes)

    return amplitudes


def rolling_statistics(df, points, label, columns=None):
    """ Compute rolling mean and std deviation of columns in `df`
    and concatenate with input data.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.
    points : int
        Number of points to use for rolling window.
    label : str
        Column suffix to add to processed columns.
    columns : list[Any], optional
        If provided, only process `columns`

    Returns
    -------
    pandas.DataFrame
    """
    if columns is None:
        columns = df.columns
    rename = lambda stat: dict(
        zip(columns, [c + ' {}_{}'.format(stat, label) for c in columns]))

    mean = df[columns].rolling(points).mean().rename(columns=rename('mean'))
    std = df[columns].rolling(points).std().rename(columns=rename('std'))

    return pd.concat((mean, std), axis=1)


def next_step_split(data, look_back=1):
    """ Split data into X and Y where X is `look_back` number of
    data-points and Y are the points coming directly after the corresponding
    points in X.

    Parameters
    ----------
    data : array
        Data to split.
    look_back : int
        Number of steps to look back at.

    Returns
    -------
    X, Y : array
        X and Y from `data`
    """
    X, Y = [], []
    for i in range(len(data)-look_back-1):
        slice = data[i:(i+look_back)]
        X.append(slice)
        Y.append(data[i + look_back])

    return np.squeeze(X), np.squeeze(Y)


def timestep_reshape(data):
    """ Reshape data into (`n_rows`, `timesteps`, `n_columns`).

    Parameters
    ----------
    data : array_like
        Data to reshape.

    Returns
    -------
    array
        Reshaped `data`.
    """
    return np.reshape(data, (data.shape[0], 1, data.shape[1]))


def _partial_w_name(func, *args, funcname=None, **kwargs):
    """

    Parameters
    ----------
    func : Callable
        Function to partially apply.
    args : tuple
        Positional arguments passed to `functools.partial`.
    funcname : str
        Name to set as `__name__`.
    kwargs : dict
        Keyword arguments passed to `functools.partial`.

    Returns
    -------
    Callable
    """
    new_func = functools.partial(func, *args, **kwargs)
    new_func.__name__ = funcname if funcname is not None else func.__name__
    return new_func
