import numpy as np


#[[[scores] for each functions] for each algorithm]
def full_comparison_oriented_scores(raw_scores):
    fnorm = 1 - ((raw_scores - np.min(raw_scores,0)) / (np.max(raw_scores,0) - np.min(raw_scores,0)))
    return np.mean(fnorm, 2)

"""def full_comparison_oriented_scores(raw_scores):
    fnorm = (raw_scores - np.min(raw_scores,0)) / (np.max(raw_scores,0) - np.min(raw_scores,0))
    return 1 - fnorm"""