import numpy as np


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


def strided_sliding_window(data, window):
    """ Make sliding window using striding.

    Consumes a lot of memory.

    Parameters
    ----------
    data : array_like
        Data to slide over.
    window : int
        Size of sliding window.

    Returns
    -------
    array_like
        Strided array.
    """
    shape = data.shape[:-1] + (data.shape[-1] - window + 1, window)
    strides = data.strides + (data.strides[-1],)
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)


def sliding_window(data, window, overlap_fac=0):
    """ Generator function which yields windows from `data`.

    Parameters
    ----------
    data : array_like
        Data to slide over.
    window : int
        Window width.
    overlap_fac : float
        To which extent windows should overlap.

    Yields
    ------
    segment : array_like
        Sliding window segment of length `window`.
    """
    hop_size = int(np.floor(window * (1 - overlap_fac)))
    total_segments = int(np.ceil(len(data) / float(hop_size)))

    for i in range(total_segments):
        current_hop = hop_size * i
        segment = data[current_hop: current_hop + window]
        yield segment
