import numpy as np, museval

def score_track(ref_dict, est_dict):
    """
    ref_dict/est_dict: stem -> wav_stereo (T,2). Truncate to common length.
    Returns: {stem: {"SDR_med": float}}
    """
    stems = ["vocals","drums","bass","other"]
    T = min(*(len(x) for x in list(ref_dict.values())+list(est_dict.values())))
    ref = np.stack([ref_dict[s][:T] for s in stems], axis=0)  # (S,T,2)
    est = np.stack([est_dict[s][:T] for s in stems], axis=0)
    scores = museval.evaluate(ref, est, win=1.0, hop=1.0, mode="v4")  # returns BSSEvalV4 object
    out = {}
    for i, s in enumerate(stems):
        sdr = np.nanmedian(scores.sdr.values[:, i])  # median over windows
        out[s] = {"SDR_med": float(sdr)}
    return out
