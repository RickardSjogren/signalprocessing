import numpy as np
import pandas as pd
import functools
from .misc import chunk_df_on_diff


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


def fft_amplitude(signal, frame_rate, as_db=False):
    """ Perform FFT and return amplitude and frequencies.

    Maximum frequency is `len(signal) // 2` according to
    Shannon-Nyqvist sampling theorem.

    Parameters
    ----------
    signal : array_like
        1D-signal.
    frame_rate : int, float
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
    N = len(signal)
    fft = np.fft.fft(signal)
    amplitudes = 2 / N * np.abs(fft[:N // 2])

    if as_db:
        amplitudes = 10 * np.log10(amplitudes)

    return amplitudes


def process_airgard_df(df, spectral_transform=None,
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
                                             frame_rate=5000, as_db=True)
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

    n_transforms = 1 + len(spatial_transform) + len(current_transform)
    processed = list()

    for chunk in chunk_df_on_diff(df, 'Time', .003):
        if chunk is None:
            processed.append([None for _ in range(n_transforms)])

        spectrum = spectral_transform(chunk.Mic)
        if spectrum.ndim > 1:
            spectrum = np.mean(spectrum, axis=0)

        spatial = [transform(chunk.drop(['Time', 'Mic', 'Cur'], 1))
                   for transform in spatial_transform]
        current = [transform(chunk['Cur']) for transform in current_transform]

        processed.append((spatial, current, spectrum))

    cols = [c + t.__name__ for t in spatial_transform
            for c in processed[0][0].index]
    cols += [c + t.__name__ for t in current_transform
             for c in processed[0][0].index]
    cols += ['{}{}'.format(spatial_transform.__name__, i) for
             i, _ in enumerate(processed[0][2], start=1)]

    processed_arr = np.empty((len(processed), len(cols)),
                             dtype=float)

    for i, (spatial, current, spectrum) in enumerate(processed):
        if spatial is None:
            processed_arr[i] = np.nan
            continue

        processed_arr[i, :len(spatial)] = spatial.values
        processed_arr[i, len(spatial):len(spatial) + len(current)] = current.values
        processed_arr[i, len(spatial) + len(current):] = spectrum

    processed_df = pd.DataFrame(processed_arr, columns=cols)
    return processed_df


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
