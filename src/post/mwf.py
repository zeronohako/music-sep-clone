import numpy as np
import norbert

STEMS = ["vocals", "drums", "bass", "other"]

def apply_mwf(mix_S, est_mag_dict, iterations=1):
    """
    mix_S:        complex (2, F, T)  stereo mixture STFT
    est_mag_dict: stem -> (2, F, T)  estimated magnitudes before MWF
    returns:      dict[stem] -> complex (2, F, T)
    """
    # Norbert wants time-first:
    # x: (T, F, C), v: (T, F, C_or_1, S)
    X = np.transpose(mix_S, (2, 1, 0))  # (T, F, 2)

    V_list = []
    for stem in STEMS:
        M = est_mag_dict[stem]            # (2, F, T) magnitudes
        # mono PSD per source (average over channels), then time-first
        V_tf = np.mean(M**2, axis=0).T    # (T, F)
        V_list.append(V_tf[..., None])    # (T, F, 1)

    V = np.stack(V_list, axis=-1)         # (T, F, 1, S)

    # Run multichannel Wiener filter
    Y_tfcs = norbert.wiener(V, X, iterations=iterations)  # (T, F, 2, S)

    # Back to (S, 2, F, T) then to dict of (2, F, T)
    Y_s2ft = np.transpose(Y_tfcs, (3, 2, 1, 0))           # (S, 2, F, T)
    return {stem: Y_s2ft[i] for i, stem in enumerate(STEMS)}
