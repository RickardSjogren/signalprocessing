import numpy as np
from scipy.io import wavfile


def array_to_wav(data, sample_rate, out_path):
    """ Write 1D-array to wav-file.

    Parameters
    ----------
    data : array_like
        1D-array to write.
    sample_rate : int
        Frames per second.
    out_path : str
        File to save to.
    """
    scaled = (data/np.max(np.abs(data)) * 32767).astype(np.int16)
    wavfile.write(out_path, sample_rate, scaled)