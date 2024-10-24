import numpy as np


def full_comparison_oriented_scores(raw_scores):
    fnorm = (raw_scores - np.min(raw_scores,0)) / (np.max(raw_scores,0) - np.min(raw_scores,0))
    return 1 - fnorm