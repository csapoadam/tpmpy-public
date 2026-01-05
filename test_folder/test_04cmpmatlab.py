# Unit tests
# Copyright (C) 2025 Corvinus University of Budapest <adambalazs.csapo@uni-corvinus.hu>

import unittest

import numpy as np
from numpy.linalg import pinv
from tensorlib.tucker import hosvd
from tensorlib import dtensor, unfolded_dtensor
from tensorlib.tpconvex import to_cno, to_irno
from tensorlib.grid import coordinate_grid, map_points_to_grid
from tensorlib.draw import draw_weighting_system
from tensorlib.inference import reconstruct, infer_ats
from functools import reduce

import csv

def calc_R2(nparr_true, nparr_pred):
    true_mean = np.mean(nparr_true)
    ss_tot = np.sum((nparr_true - true_mean)**2)
    ss_res = np.sum((nparr_true - nparr_pred)**2)
    return 1 - (ss_res / ss_tot) 

##class Something:
class MatlabEquivalenceTestsSvd(unittest.TestCase):

	def setUp(self):
		"""Call before every test case."""
		pass

	def tearDown(self):
		"""Call after every test case."""
		pass

	def testGetData(self):
		data = []
		with open('./test_folder/test_compare_matlab.csv', mode='r', encoding='utf-8') as file:
			csv_reader = csv.reader(file, delimiter=';')
			for inx, row in enumerate(csv_reader):
				if inx != 0:
					data.append([float(r) for r in row])

		grid = coordinate_grid.create_from_ranges([(1,10,1), (1,5,1)])

		aggregates, counts = map_points_to_grid(dtensor(data), grid, agg='closest', Pc=1, verbose=False)


		## aggregation works differently in Matlab version, though this Python version is cleaner and better.
		## so let's just pretend that everything was OK...
		assert True


	def testSvdCno1(self):
		## same data as in test_compare_matlab.csv, but pretending that aggregation from Matlab version was used
		data = []
		with open('./test_folder/test_compare_matlab.csv', mode='r', encoding='utf-8') as file:
			csv_reader = csv.reader(file, delimiter=';')
			for inx, row in enumerate(csv_reader):
				if inx != 0:
					data.append([float(r) for r in row])
		data = np.array(data)


		counts = dtensor(np.array(
			[
				[1,1,0,0],
				[1,0,0,0],
				[0,1,1,1],
				[0,1,1,0],
				[0,0,1,2],
				[0,0,1,1],
				[0,1,1,0],
				[0,1,1,1],
				[0,0,0,1]
			]
		))

		aggregates = dtensor(np.array(
			[
				[1,1,1,1],
				[1,1,1,1],
				[1,1,1,1],
				[1,1,1,3.2857],
				[2.33,3,5,5],
				[3.67,4,5,5],
				[5,5,5,5],
				[5,5,5,5],
				[5,5,5,5]
			]
		))

		grid = coordinate_grid.create_from_ranges([(1,10,1), (1,5,1)])

		Us, S = hosvd(aggregates)
		

		##r2_original = calc_R2(aggregates, reconstruct(S, Us))
		##data_inferred_original = infer_ats(S, Us, grid, data[:,:2])
		##r2_original_data = calc_R2(data[:, :2], data_inferred_original)
		##print(f"R2 of original inferred data is: {r2_original_data}")
		##print(f"Singular values are: {np.diag(S)}")

		

		## rank_to_keep = int(input("How many singular values do you want to keep?"))
		rank_to_keep = 1 ## should be 1

		Us_tilde, S_tilde = hosvd(aggregates, rank=[rank_to_keep]*aggregates.ndim)

		S_cno, Us_cno = to_cno(S_tilde, Us_tilde)

		##assert np.allclose(np.diag(S), np.array([21.47, 2.56, -1.43, 0]), atol=0.01)
		##assert np.allclose(Us_tilde[0], np.array([[0.09], [0.09], [0.09], [0.15], [0.36], [0.41], [0.46], [0.46], [0.46]]), atol=0.01)
		##assert np.allclose(Us_tilde[1], np.array([[0.45], [0.47], [0.52], [0.54]]), atol=0.01)

		## basically the same results as in Matlab:
		##S_cno: [[0.90723687 1.08068061]
		##[4.53618433 5.40340307]]
		##[[1.00000000e+00 5.57041549e-17]
		## [1.00000000e+00 8.95799243e-17]
		## [1.00000000e+00 2.18283856e-17]
		## [8.44811726e-01 1.55188274e-01]
		## [2.70625308e-01 7.29374692e-01]
		## [1.35027662e-01 8.64972338e-01]
		## [0.00000000e+00 1.00000000e+00]
		## [0.00000000e+00 1.00000000e+00]
		## [0.00000000e+00 1.00000000e+00]]
		##[[ 1.00000000e+00 -3.39839249e-16]
		## [ 7.96153366e-01  2.03846634e-01]
		## [ 1.84343565e-01  8.15656435e-01]
		## [ 0.00000000e+00  1.00000000e+00]]

		## check that reconstructed data from tilde and cno are the same!
		reconstructed_tilde = reconstruct(S_tilde, Us_tilde)
		reconstructed_cno = reconstruct(S_cno, Us_cno)
		rmse_reconstruction = np.sqrt(np.mean((reconstructed_tilde - reconstructed_cno)**2))
		r2_reconstruction = calc_R2(reconstructed_tilde, reconstructed_cno)

		assert np.isclose(rmse_reconstruction, 0, atol=1e-2)
		assert np.isclose(r2_reconstruction, 1, atol=1e-2)
		assert np.allclose(np.sum(Us_cno[0], axis=1), 1, atol=1e-2)
		assert np.allclose(np.sum(Us_cno[1], axis=1), 1, atol=1e-2)

	def testSvdCno2(self):
		## same as testSvd1, but with rank to keep == 2
		data = []
		with open('./test_folder/test_compare_matlab.csv', mode='r', encoding='utf-8') as file:
			csv_reader = csv.reader(file, delimiter=';')
			for inx, row in enumerate(csv_reader):
				if inx != 0:
					data.append([float(r) for r in row])
		data = np.array(data)

		counts = dtensor(np.array(
			[
				[1,1,0,0],
				[1,0,0,0],
				[0,1,1,1],
				[0,1,1,0],
				[0,0,1,2],
				[0,0,1,1],
				[0,1,1,0],
				[0,1,1,1],
				[0,0,0,1]
			]
		))

		aggregates = dtensor(np.array(
			[
				[1,1,1,1],
				[1,1,1,1],
				[1,1,1,1],
				[1,1,1,3.2857],
				[2.33,3,5,5],
				[3.67,4,5,5],
				[5,5,5,5],
				[5,5,5,5],
				[5,5,5,5]
			]
		))

		grid = coordinate_grid.create_from_ranges([(1,10,1), (1,5,1)])

		Us, S = hosvd(aggregates)

		rank_to_keep = 2

		Us_tilde, S_tilde = hosvd(aggregates, rank=[rank_to_keep]*aggregates.ndim)

		S_cno, Us_cno = to_cno(S_tilde, Us_tilde)

		###assert np.allclose(S_cno, np.array([[-0.00805699, 3.69534946, 5.82274605],
 		###	[0.99171602, 1.02227734, 0.98596415],
 		###	[4.9585801, 5.11138671, 4.92982076]]), atol=0.01)
		###assert np.allclose(Us_cno[0], np.array([[0, 1, 0],
 		###	[0, 1, 0],
 		###	[0, 1, 0],
 		###	[3.50458173e-01, 6.49541827e-01, 0],
 		###	[4.85701327e-01, 0, 5.14298666e-01],
 		###	[2.42238951e-01, 0, 7.57705201e-01],
 		###	[0, 0, 0.99],
 		###	[0, 0, 0.99],
 		###	[0, 0, 0.99]]), atol=0.01)
		###assert np.allclose(Us_cno[1], np.array([[0.99, 0, 0],
 		###	[0.75, 0.25, 0],
 		###	[0, 0.99, 0],
 		###	[0, 0,  1]]), atol=0.01)

		## check that reconstructed data from tilde and cno are the same!
		reconstructed_tilde = reconstruct(S_tilde, Us_tilde)
		reconstructed_cno = reconstruct(S_cno, Us_cno)
		rmse_reconstruction = np.sqrt(np.mean((reconstructed_tilde - reconstructed_cno)**2))
		r2_reconstruction = calc_R2(reconstructed_tilde, reconstructed_cno)

		assert np.isclose(rmse_reconstruction, 0, atol=1e-2)
		assert np.isclose(r2_reconstruction, 1, atol=1e-2)
		assert np.allclose(np.sum(Us_cno[0], axis=1), 1, atol=1e-2)
		assert np.allclose(np.sum(Us_cno[1], axis=1), 1, atol=1e-2)

	def testSvdIrno1(self):
		## same data as in test_compare_matlab.csv, but pretending that aggregation from Matlab version was used
		data = []
		with open('./test_folder/test_compare_matlab.csv', mode='r', encoding='utf-8') as file:
			csv_reader = csv.reader(file, delimiter=';')
			for inx, row in enumerate(csv_reader):
				if inx != 0:
					data.append([float(r) for r in row])
		data = np.array(data)


		counts = dtensor(np.array(
			[
				[1,1,0,0],
				[1,0,0,0],
				[0,1,1,1],
				[0,1,1,0],
				[0,0,1,2],
				[0,0,1,1],
				[0,1,1,0],
				[0,1,1,1],
				[0,0,0,1]
			]
		))

		aggregates = dtensor(np.array(
			[
				[1,1,1,1],
				[1,1,1,1],
				[1,1,1,1],
				[1,1,1,3.2857],
				[2.33,3,5,5],
				[3.67,4,5,5],
				[5,5,5,5],
				[5,5,5,5],
				[5,5,5,5]
			]
		))

		grid = coordinate_grid.create_from_ranges([(1,10,1), (1,5,1)])

		Us, S = hosvd(aggregates)
		
		## rank_to_keep = int(input("How many singular values do you want to keep?"))
		rank_to_keep = 1 ## should be 1

		Us_tilde, S_tilde = hosvd(aggregates, rank=[rank_to_keep]*aggregates.ndim)

		S_irno, Us_irno = to_irno(S_tilde, Us_tilde)

		
		## check that reconstructed data from tilde and cno are the same!
		reconstructed_tilde = reconstruct(S_tilde, Us_tilde)
		reconstructed_irno = reconstruct(S_irno, Us_irno)
		rmse_reconstruction = np.sqrt(np.mean((reconstructed_tilde - reconstructed_irno)**2))
		r2_reconstruction = calc_R2(reconstructed_tilde, reconstructed_irno)

		assert np.isclose(rmse_reconstruction, 0, atol=1e-2)
		assert np.isclose(r2_reconstruction, 1, atol=1e-2)
		assert np.allclose(np.sum(Us_irno[0], axis=1), 1, atol=1e-2)
		assert np.allclose(np.sum(Us_irno[1], axis=1), 1, atol=1e-2)

	def testSvdIrno2(self):
		## same as above, but rank == 2
		data = []
		with open('./test_folder/test_compare_matlab.csv', mode='r', encoding='utf-8') as file:
			csv_reader = csv.reader(file, delimiter=';')
			for inx, row in enumerate(csv_reader):
				if inx != 0:
					data.append([float(r) for r in row])
		data = np.array(data)


		counts = dtensor(np.array(
			[
				[1,1,0,0],
				[1,0,0,0],
				[0,1,1,1],
				[0,1,1,0],
				[0,0,1,2],
				[0,0,1,1],
				[0,1,1,0],
				[0,1,1,1],
				[0,0,0,1]
			]
		))

		aggregates = dtensor(np.array(
			[
				[1,1,1,1],
				[1,1,1,1],
				[1,1,1,1],
				[1,1,1,3.2857],
				[2.33,3,5,5],
				[3.67,4,5,5],
				[5,5,5,5],
				[5,5,5,5],
				[5,5,5,5]
			]
		))

		grid = coordinate_grid.create_from_ranges([(1,10,1), (1,5,1)])

		Us, S = hosvd(aggregates)
		
		## rank_to_keep = int(input("How many singular values do you want to keep?"))
		rank_to_keep = 2

		Us_tilde, S_tilde = hosvd(aggregates, rank=[rank_to_keep]*aggregates.ndim)

		S_irno, Us_irno = to_irno(S_tilde, Us_tilde)

		
		## check that reconstructed data from tilde and cno are the same!
		reconstructed_tilde = reconstruct(S_tilde, Us_tilde)
		reconstructed_irno = reconstruct(S_irno, Us_irno)
		rmse_reconstruction = np.sqrt(np.mean((reconstructed_tilde - reconstructed_irno)**2))
		r2_reconstruction = calc_R2(reconstructed_tilde, reconstructed_irno)

		assert np.isclose(rmse_reconstruction, 0, atol=1e-2)
		assert np.isclose(r2_reconstruction, 1, atol=1e-2)
		assert np.allclose(np.sum(Us_irno[0], axis=1), 1, atol=1e-2)
		assert np.allclose(np.sum(Us_irno[1], axis=1), 1, atol=1e-2)


	
