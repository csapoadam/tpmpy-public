# Unit tests
# Copyright (C) 2025 Corvinus University of Budapest <adambalazs.csapo@uni-corvinus.hu>

import unittest

import numpy as np
from numpy.linalg import pinv
from tensorlib.tucker import hosvd
from tensorlib import dtensor, unfolded_dtensor
from tensorlib.tpconvex import to_cno, decomp_cno, decomp_snnn, closeness, polarorto
from tensorlib.grid import coordinate_grid
from tensorlib.draw import draw_weighting_system
from tensorlib.inference import reconstruct, infer_ats
from functools import reduce

import csv

def calc_R2(nparr_true, nparr_pred):
    true_mean = np.mean(nparr_true)
    ss_tot = np.sum((nparr_true - true_mean)**2)
    ss_res = np.sum((nparr_true - nparr_pred)**2)
    return 1 - (ss_res / ss_tot) 

class MatlabEquivalencesDecomposition(unittest.TestCase):

	def setUp(self):
		"""Call before every test case."""
		pass

	def tearDown(self):
		"""Call after every test case."""
		pass

	def testCloseness1(self):
		# Case 1: Basic 3x3 matrix with r = 3
	    polarv = np.array([[0.5], [0.7], [0.9]])  # Example polar coordinates (3x1 for r = 3)
	    U1 = np.array([[1.0, 0.5, 0.2],
	                   [0.3, 0.8, 0.6],
	                   [0.4, 0.1, 0.9]])
	    h = 2  # Use Euclidean norm
	    hh = 0  # No additional weight
	    result = closeness(polarv, U1, h, hh)
	    # Assert the result is finite and non-negative
	    assert np.isfinite(result), "Result should be a finite number."
	    assert result >= 0, "Result should be non-negative."
	    assert np.isclose(result, 2.86, atol=1e-2), f"Result is {result}, not 2.86..."

	def testCloseness2(self):
	    # Case 2: Larger matrix with r = 4
	    polarv = np.random.rand(4, 2)  # Random polar coordinates (4x2 for r = 4)
	    U1 = np.random.rand(6, 4) - 0.5      # Random 6x4 matrix
	    h = 1                          # Use Manhattan norm
	    hh = 0.5                       # Additional weight
	    result = closeness(polarv, U1, h, hh)
	    # Assert the result is finite and non-negative
	    assert np.isfinite(result), "Result should be a finite number."
	    assert result >= 0, "Result should be non-negative."

	def testCloseness3(self):
	    # Case 4: Invalid input shapes (expect exception)
	    try:
	        polarv = np.array([[0.1, 0.2], [0.3, 0.4]])  # Wrong shape
	        U1 = np.array([[1.0, 0.5, 0.2], [0.3, 0.8, 0.6]])
	        h, hh = 2, 0.5
	        result = closeness(polarv, U1, h, hh)
	        assert False, "An exception should have been raised for invalid input shapes."
	    except ValueError as e:
	        assert True

	def testCloseness4(self):
	    # Case 5: Degenerate simplex with all zero rows in U1
	    polarv = np.random.rand(4, 2)  # Random polar coordinates for r = 4
	    U1 = np.zeros((6, 4)) + 0.5    # Degenerate input matrix
	    h = 2                          # Use Euclidean norm
	    hh = 1                         # Additional weight
	    result = closeness(polarv, U1, h, hh)
	    assert np.isfinite(result), "Result should be a finite number."
	    assert result >= 0, "Result should be non-negative."

	def testCloseness5(self):
	    # Specific input values
	    polarv = np.array([[0.5], [0.8], [1.0]])  # Polar coordinates for r = 3
	    U1 = np.array([[1.0, 0.5, 0.2],
	                   [0.3, -0.8, 0.6],
	                   [-0.4, 0.1, 0.9]])  # Input matrix (3x3)
	    h = 2  # Euclidean norm
	    hh = 0.5  # Additional weight

	    result = closeness(polarv, U1, h, hh)
	    # Validate the result
	    assert np.isclose(result, 0.8936, atol=1e-2), "Test Case 5 failed: Mismatch in results."

	def testCloseness6(self):
	    # Specific input values
	    polarv = np.array([[0.23926078, 0.01654434],
 			[0.50879577, 0.54084612],
 			[0.51901849, 0.60160894],
 			[0.39104676, 0.20245403]
 		])
	    U1 = np.array([[ 0.3117675, -0.42788082, -0.1891167, 0.3234236 ],
 			[-0.45504973,  0.35208045,  0.08503188, -0.05948601],
 			[-0.24878942, -0.36364312,  0.1147112,   0.48644995],
 			[ 0.32893661, -0.19711838,  0.31443551, -0.42111262],
 			[-0.37112393, -0.17540027, -0.29428755, -0.42986508],
 			[-0.36398655, -0.10687447, -0.21658799, -0.43436475]])
	    h = 1                          # Use Manhattan norm
	    hh = 0.5                       # Additional weight

	    result = closeness(polarv, U1, h, hh)
	    # Validate the result
	    assert np.isclose(result, 16.0126, atol=1e-2), "Test Case 5 failed: Mismatch in results."


	def testPolarorto1(self):
		polarv = np.array([[0.23926078, 0.01654434],
			[0.50879577, 0.54084612],
			[0.51901849, 0.60160894],
			[0.39104676, 0.20245403]
		])
		U1 = np.array([[ 0.3117675, -0.42788082, -0.1891167, 0.3234236 ],
			[-0.45504973,  0.35208045,  0.08503188, -0.05948601],
			[-0.24878942, -0.36364312,  0.1147112,   0.48644995],
			[ 0.32893661, -0.19711838,  0.31443551, -0.42111262],
			[-0.37112393, -0.17540027, -0.29428755, -0.42986508],
			[-0.36398655, -0.10687447, -0.21658799, -0.43436475]])
		U2, v = polarorto(U1, polarv)

		assert np.allclose(U2, np.array([[0.0033, -0.0894, 0.0741, 0.0302],
			[-0.0495, 0.0619, -0.0624, -0.0274],
			[-0.0087, -0.0104, 0.0062, 0.0016],
			[0.0170, 0.0052, -0.0012, 0.0042],
			[-0.9835, -0.3496, 0.0497, 0.0127],
			[-0.8668, -0.2950, 0.0330, 0.0070]]
		), atol=0.01)

		assert np.allclose(v, np.array([
			[-119.4079, -206.8589, 106.0718, 221.1950],
			[403.6245, 699.7185, -359.5746, -742.7683],
			[422.2385, 787.0995, -428.2195, -780.1185],
			[182.0478, 148.2312, -31.4122, -297.8667],
		]), atol=0.01)




	
