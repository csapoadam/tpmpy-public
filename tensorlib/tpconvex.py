# Port of TP model transformation utilities from Matlab TP toolbox
# Copyright (C) 2025 Corvinus University of Budapest <adambalazs.csapo@uni-corvinus.hu>

import numpy as np
import scipy
import time
from .dtensor import dtensor
from .inference import reconstruct

__all__ = [
    'to_cno',
    'to_irno'
]

def rnodiff(U, c):
    """
    Python version of the rnodiff function.
    U: 2D NumPy array
    c: scalar
    Returns: a scalar D
    """
    # 1) Chop off last column
    U0 = U[:, :-1]
    
    # 2) Get shape
    n, r = U0.shape
    
    # 3) Row sums of U0
    s = np.sum(U0, axis=1)  # shape (n,) in NumPy
    
    # 4) U1 = U0 + (c-1)/r * (row sums)
    #    We make s into shape (n,1) so it broadcasts across columns
    U1 = U0 + ((c - 1) / r) * s[:, None] * np.ones((1, r))
    
    # 5) Shift columns so min becomes 0
    #    np.min(U1, axis=0) -> shape (r,) = minima for each column
    U2 = U1 - np.ones((n, 1)) * np.min(U1, axis=0)
    
    # 6) Scale columns to [0,1]
    #    np.max(U2, axis=0) -> shape (r,) = maxima for each column
    U3 = U2 / (np.ones((n, 1)) * np.max(U2, axis=0))
    
    # 7) Scale so max row sum is 1
    #    sum over axis=1 -> each row sum, shape (n,)
    #    then take max -> scalar
    U4 = U3 / np.max(np.sum(U3, axis=1))
    
    # 8) Compute D
    #    min of row sums:
    min_row_sum = np.min(np.sum(U4, axis=1))
    #    max of first column:
    max_first_col = np.max(U4[:, 0])
    D = 1 - min_row_sum - max_first_col
    
    return D


def closeness(polarv, U1, h, hh):
    """
    Computes the closeness of a given INO (Interior Normalized Oriented) simplex
    to the NO (Normalized Oriented) condition, matching MATLAB behavior.

    Parameters:
    polarv : ndarray
        A matrix of shape (r, r-2) representing polar coordinates.
    U1 : ndarray
        An n x r matrix, where rows can contain negative values.
    h : float
        Norm type or parameter used in the closeness calculation.
    hh : float
        Additional weight for the closeness calculation using the first and last rows of U2.

    Returns:
    norma : float
        The computed closeness norm.
    """
    n, r = U1.shape

    if polarv.shape == (r*(r-2),):
        # In this case, closeness is being called by scipy.optimize.minimize...
        # Reshape polarv back to (r, r-2)
        polarv = polarv.reshape(r, r-2)

    # Validate dimensions only
    if polarv.shape != (r, r-2):
        raise ValueError("For U1 of shape (n, r), polarv must be of shape (r, r-2).")
    if n < r:
        raise ValueError("For U1 of shape (n, r), it must hold that n >= r.")

    # Step 1: Compute the vertices of the simplex in Cartesian coordinates
    a = polarv
    v = np.zeros((r, r))
    for i in range(r-2):
        v[:, i] = np.sin(a[:, i]) * np.prod(np.cos(a[:, :i]), axis=1)
    v[:, r-2] = np.prod(np.cos(a), axis=1)
    v[:, r-1] = 1 - np.sum(v, axis=1)


    # Step 2: Shift the simplex so that U1[0, :] lies inside it
    ##v += np.ones((r, 1)) * (U1[0, :] - np.sum(v, axis=0) / r)

    # Step 2: Shift the simplex so that U1[0, :] lies inside it.
    # Note: np.mean(v, axis=0) is equivalent to np.sum(v, axis=0)/r.
    v += (U1[0, :] - np.mean(v, axis=0))[None, :]

    # Step 3: Iteratively adjust the simplex
    for i in range(r):
        U2 = U1 @ np.linalg.pinv(v)  # Use pseudo-inverse for stability
        m = np.min(U2[:, i])  # Find the minimum value in column i of U2
        # Scale the simplex to ensure the opposite face touches the point cloud
        v = v - m * (v - np.ones((r, 1)) * v[i, :])
        v[:, r-1] = 0  # Set the last column of v to 0
        v[:, r-1] = 1 - np.sum(v[:, :r-1], axis=1)  # Adjust the last column to satisfy the simplex property

    # Final computation of U2
    U2 = U1 @ np.linalg.pinv(v)

    # Compute the closeness metric
    with np.errstate(divide='ignore', invalid='ignore'):  # Suppress warnings for log(0)
        log_max_U2 = np.log(np.maximum(U2, 1e-10))  # Avoid log of zero
        norma = np.linalg.norm(np.max(log_max_U2, axis=0), ord=h)

    # Additional penalty term if hh > 0
    if hh > 0:
        ends = np.log(np.sort(np.max(np.vstack((U2[0, :], U2[-1, :])), axis=0)))
        norma += hh * np.linalg.norm(ends[-2:], ord=h)

    return norma


def polarorto(U1, polarv):
    """
    Encapsulates the point cloud U1 with a simplex defined by polarv.

    Parameters:
    U1 : ndarray
        The input point cloud matrix of shape (n, r), where n is the number of points and r is the dimensionality.
    polarv : ndarray
        A matrix of shape (r, r-2) representing the polar coordinates of the simplex.

    Returns:
    U2 : ndarray
        The transformed point cloud such that U1 = U2 @ v.
    v : ndarray
        A matrix representing the vertices of the simplex in Cartesian coordinates.
    """
    n, r = U1.shape

    a = polarv

    # Step 1: Compute the vertices of the simplex in Cartesian coordinates
    v = np.zeros((r, r))
    for i in range(r-2):
        v[:, i] = np.sin(a[:, i]) * np.prod(np.cos(a[:, :i]), axis=1)
    
    # Compute the second-to-last column
    v[:, r-2] = np.prod(np.cos(a), axis=1)
    
    # Compute the last column of v
    v[:, r-1] = 0
    v[:, r-1] = 1 - np.sum(v, axis=1)

    # Step 2: Shift the simplex so that U1[0, :] lies inside it
    v += np.ones((r, 1)) * (U1[0, :] - np.sum(v, axis=0) / r)

    # Step 3: Iteratively adjust the simplex
    for i in range(r):
        U2 = U1 @ np.linalg.inv(v)  # Compute U2 such that U1 = U2 @ v
        m = np.min(U2[:, i])  # Find the minimum value in column i of U2
        # Scale the simplex to ensure the opposite face touches the point cloud
        v = v - m * (v - np.ones((r, 1)) * v[i, :])
        # Ensure the last column satisfies the simplex property
        v[:, r-1] = 0
        v[:, r-1] = 1 - np.sum(v, axis=1)

    # Final computation of U2
    U2 = U1 @ np.linalg.inv(v)

    return U2, v

def decomp_snnn(U, sumtol=1e-5):
    """
    Perform the SNNN decomposition of a matrix U with orthogonal columns.

    Parameters:
    U : ndarray
        Input matrix with shape (n, r).
    sumtol : float, optional
        Tolerance for row sum adjustments (default is 1e-5).

    Returns:
    W : ndarray
        Transformed matrix with normalized and non-negative structure.
    V : ndarray
        Matrix satisfying W @ V = U.
    """
    # Step 1: Compute column sums and initialize X
    Usum = np.sum(U, axis=0)  # Sum of each column
    X = np.diag(Usum)  # Diagonal matrix of column sums

    # Step 2: Adjust X if any column sum is close to zero
    for i in np.where(np.abs(Usum) < sumtol)[0]:
        X[i, i] += 1

        if i == 0:
            if X.shape[1] == 1:  # If only one column, add another column
                X = np.hstack((X, -np.ones((X.shape[0], 1))))
            else:
                X[i, i+1] = -1  # Safe to access i+1
        else:
            X[i, i-1] = -1  # Safe to access i-1


    # Step 3: Check for SN property and compute Usn
    if np.all(np.abs(np.sum(np.matmul(U, U.T), axis=1) - 1) < sumtol):
        # If SN property holds, transform U with X
        if np.all(np.abs(Usum) < sumtol):
            print('SN transformation is close to singular, expect errors')
        Usn = np.matmul(U, X)
    else:
        # If SN property does not hold, append a column to U
        Usn = np.hstack((U, 1 - np.sum(U, axis=1, keepdims=True)))

    # Step 4: Compute M to ensure >= 0 condition
    n = Usn.shape[1]  # Number of columns
    umin = min(np.min(Usn), 0)  # Minimum value in Usn
    umax = max(np.max(Usn), 2/n)  # Maximum value in Usn
    c1 = 1 / (1 - umin * n)
    c2 = 1 / (1 - umax * n)
    c = c2 if c1 < -c2 else c1
    d = (1 - c) / n
    M = d * np.ones((n, n)) + c * np.eye(n)

    # Step 5: Compute W and V
    W = np.matmul(Usn, M)
    V = np.linalg.lstsq(W, U, rcond=None)[0]

    return W, V


def decomp_irno(U_todecomp):
    """
    Python equivalent of the MATLAB rnoino(U0) function.
    Returns (U, fi) just like the MATLAB version.
    """
    U0, _ = decomp_snnn(U_todecomp)

    # Make a local copy of U0 to avoid modifying the caller's data
    U = U0.copy()
    n, r = U.shape

    # i = r - 1 in MATLAB (1-based), same numeric value in Python
    i = r - 1

    # ------------------------------------------------------------------------
    # In MATLAB:
    #    vet(:, i) = U(:, i+1)   --> That picks up the last column (when i=r-1)
    #    U = U(:, 1:i)           --> remove that last column from U
    #
    # Then "U = U + ...; U = U - ...;" adds and subtracts the same quantity,
    # which effectively does nothing to U. Finally, the code appends
    #   [U  1 - sum(U, 2)]
    #
    # The net result is that the last column is replaced by (1 - sum(U, 2)).
    #
    # For fidelity, we replicate these steps in Python, but more simply.
    # ------------------------------------------------------------------------

    # Extract the last column (in MATLAB: vet(:, i)), shape (n,)
    vet = U[:, -1]

    # Truncate U to its first (r-1) columns (in MATLAB: U(:, 1:i))
    U = U[:, :-1]

    # Add then subtract the same Nx1 vector across columns => net zero effect
    U = U + (vet[:, None] * np.ones((1, i)) / i)
    U = U - (vet[:, None] * np.ones((1, i)) / i)

    # Append a new last column "1 - row_sums"
    # (in MATLAB: U=[U 1-sum(U,2)])
    U = np.hstack([U, 1.0 - np.sum(U, axis=1, keepdims=True)])

    # ------------------------------------------------------------------------
    # Search for a parameter a(2) so that rnodiff(U, a(2)) >= 0
    # expanding a(2) by doubling until rnodiff(...) >= 0
    # ------------------------------------------------------------------------
    a = [0.0, 1.0]
    while rnodiff(U, a[1]) < 0:
        a[1] *= 2.0

    # Now do a bisection (interval halving) until convergence
    d = [rnodiff(U, a[0]), rnodiff(U, a[1])]

    while abs(d[0]) + abs(d[1]) > 1e-6:
        auj = (a[0] + a[1]) / 2.0
        if rnodiff(U, auj) > 0:
            a[1] = auj
        else:
            a[0] = auj
        d = [rnodiff(U, a[0]), rnodiff(U, a[1])]

    # The final parameter is the midpoint
    auj = (a[0] + a[1]) / 2.0

    # ------------------------------------------------------------------------
    # Next, the code re-applies a transformation on U with this final auj
    # ------------------------------------------------------------------------
    U00 = U[:, :-1]            # everything except last col
    s   = np.sum(U00, axis=1)  # row sums
    
    # (U00 + ((auj-1)/i)*s * ones(1,i))
    U1  = U00 + ((auj - 1) / i) * s[:, None] * np.ones((1, i))
    # shift columns so min() is 0
    U2  = U1 - np.ones((n, 1)) * np.min(U1, axis=0)
    # scale columns to [0,1]
    U25 = np.ones((n, 1)) * np.max(U2, axis=0)
    U3  = U2 / U25
    # scale so max row-sum is 1
    row_sums = np.sum(U3, axis=1)  # shape (n,)
    U4  = U3 / np.max(row_sums)

    # Append the new last column again
    U = np.hstack([U4, 1.0 - np.sum(U4, axis=1, keepdims=True)])

    return U

def decomp_cno(U, h, hh, nveletlen, nzavar, nlokalis):
    """
    Perform the CNO decomposition of a matrix U with orthogonal columns.

    Parameters:
    U : ndarray
        Input matrix with shape (n, r).
    ...

    Returns:
    W : ndarray
        Transformed matrix with close-to-normal structure
    V : ndarray
        Matrix satisfying W @ V = U.
    """

    W1, _ = decomp_snnn(U)
    n, r = W1.shape

    if r > n:
        raise ValueError('If the input matrix is nxr, r cannot be greater than n')
    
    if r == 2:
        min_vals = np.min(W1, axis=0)
        V2 = np.vstack((W1[np.argmin(W1[:, 0]), :], W1[np.argmin(W1[:, 1]), :]))
        W = np.linalg.solve(V2.T, W1.T).T
        
    else:
        # Inertia calculation
        center = np.mean(W1[:, :r-1], axis=0)
        U11 = W1[:, :r-1] - center
        inertia = U11.T @ U11
    
        # Transformation to unit inertia
        sv, se = np.linalg.eigh(inertia)
        sv_inv_sqrt = np.diag(1 / np.sqrt(np.maximum(sv, 1e-10)))

        U111 = U11 @ se @ sv_inv_sqrt
        W1 = U111
        W1 = np.hstack((W1, 1 - np.sum(W1, axis=1, keepdims=True)))
    
        # Ideal decomposition via numerical methods
        ertekmin = float('inf')
        rng = np.random.default_rng(int(time.time()))

        # Random searches
        for _ in range(nveletlen):
            veletlenhely = (rng.random((r, r-2)) - 0.5) * np.pi
            veletlenhely[:, -1] *= 2

            # Flatten veletlenhely for minimize
            veletlenhely_flat = veletlenhely.ravel()

            res = scipy.optimize.minimize(closeness, veletlenhely_flat, args=(W1, h, hh), method='Nelder-Mead')

            if res.fun < ertekmin:
                ertekmin = res.fun
                polarmin = res.x.reshape(r, r-2)
    
        # Local optimization
        szamlalo = 0
        while szamlalo < nlokalis:
            polarmin_flat = polarmin.ravel()
            res = scipy.optimize.minimize(closeness, polarmin_flat, args=(W1, h, hh), method='Nelder-Mead')
            if res.fun < ertekmin - 0.001:
                szamlalo = 0
                ertekmin = res.fun
                polarmin = res.x.reshape(r, r-2)
            else:
                szamlalo += 1
    
        # Perturbation and improvement
        for _ in range(nzavar):
            zavar = (rng.random((r, r-2)) - 0.5) * np.pi
            zavar[:, -1] *= 2

            polarmin_plus_perturb = polarmin + 0.1 * rng.random() * zavar
            polarmin_plus_perturb_flat = polarmin_plus_perturb.ravel()

            res = scipy.optimize.minimize(closeness, polarmin_plus_perturb_flat, args=(W1, h, hh), method='Nelder-Mead')
            polar = res.x.reshape(r, r-2)
            ertek = res.fun
        
            szamlalo = 0
            while szamlalo < nlokalis:

                polar_flat = polar.ravel()
                res_local = scipy.optimize.minimize(closeness, polar_flat, args=(W1, h, hh), method='Nelder-Mead')
                if res_local.fun < ertek - 0.001:
                    szamlalo = 0
                    ertek = res_local.fun
                    polar = res_local.x.reshape(r, r-2)
                else:
                    szamlalo += 1
        
            if ertek < ertekmin:
                ertekmin = ertek
                polarmin = polar
    
        # Normalize the result
        if r > 3:
            ### have not tested this if clause yet...
            for i in range(r):
                for j in range(r - 3):
                    if (polarmin[i, j] + np.pi / 2) % (2 * np.pi) > np.pi:
                        polarmin[i, j+1] += np.pi
                        polarmin[i, j] = np.pi - polarmin[i, j]
            polarmin[:, -1] %= 2 * np.pi
        polarmin[:, :r-2] = (polarmin[:, :r-2] + np.pi / 2) % (2 * np.pi) - np.pi / 2
        polarmin = polarmin[np.argsort(polarmin[:, 0]), :]
    
        print("Distance from NO:", ertekmin)

        W, _ = polarorto(W1, polarmin)

    return W


def to_cno(S, Us, h=1, hh=5, nveletlen=50, nzavar=15, nlokalis=10):
    """
    Implementation of Close-to-Normal (CNO) transformation, i.e.
    Based on a TP model consisting of a core tensor S and weighting matrices Us,
    returns an equivalent TP model consisting of:
    - core tensor new_S and
    - weighting matrices new_Us
    such that in each dimension, each weighting function reaches a value as close to 1 as possible for one of the gridpoints
    (correspondingly, all other weighting function values will be close to 0 at the same gridpoint) 
    """

    originalData = reconstruct(S, Us)
    inv_Ws = [None for U in Us]
    new_Us = [None for U in Us]

    for dim in range(len(S.shape)):
        W = decomp_cno(Us[dim], h, hh, nveletlen, nzavar, nlokalis)
        new_Us[dim] = W
        inv_Ws[dim] = np.linalg.pinv(W)

    new_S = reconstruct(originalData, inv_Ws)
        
    return new_S, new_Us

def to_irno(S, Us):
    """
    Implementation of IRNO transformation, i.e.
    Based on a TP model consisting of a core tensor S and weighting matrices Us,
    returns an equivalent TP model consisting of:
    - core tensor new_S and
    - weighting matrices new_Us
    such that in each dimension, each weighting function reaches the same maximum value that is <= 1 at some point
    and the weighting functions are normalized and >= 0
    """

    originalData = reconstruct(S, Us)
    inv_Ws = [None for U in Us]
    new_Us = [None for U in Us]

    for dim in range(len(S.shape)):
        W = decomp_irno(Us[dim])
        new_Us[dim] = W
        inv_Ws[dim] = np.linalg.pinv(W)

    new_S = reconstruct(originalData, inv_Ws)
        
    return new_S, new_Us