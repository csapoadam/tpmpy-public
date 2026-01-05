# Unit tests
# Copyright (C) 2025 Corvinus University of Budapest <adambalazs.csapo@uni-corvinus.hu>

import unittest

import numpy as np
from numpy.linalg import pinv
from tensorlib.tucker import hosvd
from tensorlib import dtensor, unfolded_dtensor
from tensorlib.tpconvex import to_cno
from tensorlib.grid import coordinate_grid, map_points_to_grid
from tensorlib.draw import draw_weighting_system
from tensorlib.inference import reconstruct, infer_ats, infer_from_ruleset
from functools import reduce


class InferenceTests(unittest.TestCase):

	def setUp(self):
		"""Call before every test case."""
		pass

	def tearDown(self):
		"""Call after every test case."""
		pass

	def testReconstruct1(self):
		inputs = [[8,5], [12,4], [2,4]]
		outputs = []
		for v1,v2 in inputs:
			o = v1 + v2
			outputs.append([v1,v2,o])

		grid = coordinate_grid.create_from_ranges([(0,14,2), (3.5,8,1)])

		aggregates, counts = map_points_to_grid(dtensor(outputs), grid, agg='closest', Pc=1, verbose=False)

		Us, S = hosvd(dtensor(aggregates))

		reconstructed = reconstruct(S, [None, None])
		assert np.allclose(S, reconstructed, atol=1e-2)


	def testInference1(self):
		inputs = [[8,5], [12,4], [2,4]]
		outputs = []
		for v1,v2 in inputs:
			o = v1 + v2
			outputs.append([v1,v2,o])

		grid = coordinate_grid.create_from_ranges([(0,14,2), (3.5,8,1)])

		aggregates, counts = map_points_to_grid(dtensor(outputs), grid, agg='closest', Pc=1, verbose=False)

		Us, S = hosvd(dtensor(aggregates))

		results = infer_ats(S, Us, grid, np.array([[9,5], [12, 3]]))

		assert(np.allclose(results, np.array([[13], [16]])))

	def testInference2(self):
		inputs = [[8,5], [12,4], [2,4]]
		outputs = []
		for v1,v2 in inputs:
			o = v1 + v2
			outputs.append([v1,v2,o])

		grid = coordinate_grid.create_from_ranges([(0,14,2), (3.5,8,1)])

		aggregates, counts = map_points_to_grid(dtensor(outputs), grid, agg='mean', Pc=2, verbose=False)

		Us, S = hosvd(dtensor(aggregates))

		results = infer_ats(S, Us, grid, np.array([[9,5], [12, 3]]))

		assert(np.allclose(results, np.array([[14.5], [14.5]])))

	def testInferenceFromRuleset(self):



	
