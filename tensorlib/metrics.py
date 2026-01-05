# Commonly used error metrics
# Copyright (C) 2025 Corvinus University of Budapest <adambalazs.csapo@uni-corvinus.hu>

import numpy as np

def calc_R2(nparr_true, nparr_pred):
    true_mean = np.mean(nparr_true)
    ss_tot = np.sum((nparr_true - true_mean)**2)
    ss_res = np.sum((nparr_true - nparr_pred)**2)
    return float(1 - (ss_res / ss_tot))

def calc_nRMSE_maxmin(nparr_true, nparr_pred):
    rmse = np.sqrt(np.mean((nparr_true - nparr_pred) ** 2))
    return rmse / (np.max(nparr_true) - np.min(nparr_true))

def calc_nRMSE_iqr(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    iqr = np.percentile(y_true, 75) - np.percentile(y_true, 25)
    return rmse / iqr

def calc_cindex(nparr_true, nparr_pred):
    """
    Computes the concordance index (C-index) for two 1D NumPy arrays.
    
    Parameters:
        nparr_true (np.ndarray): Ground truth values
        nparr_pred (np.ndarray): Predicted values
    
    Returns:
        float: Concordance index in [0, 1]
    """
    assert len(nparr_true) == len(nparr_pred), "Arrays must be the same length"

    concordant = 0
    discordant = 0
    ties = 0

    n = len(nparr_true)

    for i in range(n):
        for j in range(i + 1, n):
            if nparr_true[i] == nparr_true[j]:
                continue  # no informative comparison

            # Check ordering
            true_diff = nparr_true[i] - nparr_true[j]
            pred_diff = nparr_pred[i] - nparr_pred[j]

            if np.sign(true_diff) == np.sign(pred_diff):
                concordant += 1
            elif np.sign(pred_diff) == 0:
                ties += 1
            else:
                discordant += 1

    total_pairs = concordant + discordant + ties
    if total_pairs == 0:
        return np.nan  # No informative pairs

    return (concordant + 0.5 * ties) / total_pairs