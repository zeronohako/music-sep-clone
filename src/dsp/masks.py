import numpy as np

STEMS = ["vocals", "drums", "bass", "other"]

def ideal_ratio_masks(ref_mags_dict):
    """ref_mags_dict[name] = (2,F,Frames) magnitudes"""
    sum_mag = np.sum([ref_mags_dict[k] for k in STEMS], axis=0) + 1e-8
    irm = {k: ref_mags_dict[k] / sum_mag for k in STEMS}
    return irm

def apply_mask(mix_S, mask):
    """mix_S: complex (2,F,Frames); mask: (2,F,Frames) real in [0,1]"""
    return mask * mix_S
