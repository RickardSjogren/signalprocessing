import numpy as np


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


def chunk_df_on_diff(df, column, cutoff):
    """ Chunk data-frame on difference in `column`.

    Parameters
    ----------
    df : pandas.DataFrame
        Data to chunk.
    column : str
        Column to split at.
    cutoff : float
        Time-diff cutoff.

    Yields
    ------
    chunk : Union[pd.DataFrame, NoneType]
        Consecutive chunks of `df` or None if chunk is missing.
    """
    large = df[column].diff() > cutoff
    cuts = list(np.where(large)[0])
    cuts.append(len(df))
    gaps = df[column].diff()[large]
    median_gap = gaps.median()

    for i, (end, gap) in enumerate(zip(cuts, gaps)):
        size = int(np.round(gap / median_gap))

        for _ in range(size - 1):
            yield None

        start = 0 if i == 0 else cuts[i - 1]
        chunk = df.iloc[start:end]

        yield chunk
