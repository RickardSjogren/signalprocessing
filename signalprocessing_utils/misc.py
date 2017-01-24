import numpy as np
import pandas as pd


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
    diffs = df[column].diff()

    large = diffs > cutoff
    cuts = list(np.where(large)[0])
    cuts.append(len(df))
    gaps = diffs[large]

    median_large_gap = gaps.median()
    where_large = np.concatenate((np.array([0]), np.where(large)[0]))
    median_chunk_size = np.median([df.Time.iloc[j - 1] - df.Time.iloc[i]
                                   for i, j in zip(where_large, where_large[1:])])

    for i, (end, gap) in enumerate(zip(cuts, gaps)):
        size = int(np.round(gap / median_large_gap))

        for _ in range(size - 1):
            gap -= median_large_gap
            yield None, median_large_gap + median_chunk_size

        start = 0 if i == 0 else cuts[i - 1]
        chunk = df.iloc[start:end]

        chunk_size = np.subtract(*chunk[column].iloc[[-1, 1]])
        yield chunk, gap + chunk_size


def iter_cols(data):
    """ Convenience function iterate over columns of `data` which
    might be numpy array or dataframe.

    Parameters
    ----------
    data : numpy.ndarray, pandas.DataFraem
        Data to iterate over.

    Yields
    ------
    pd.Series, np.ndarray
        Columns in `data`.
    """

    if isinstance(data, pd.DataFrame):
        iterator = (col for _, col in data.iteritems())
    else:
        iterator = iter(data.T)

    for col in iterator:
        yield col
