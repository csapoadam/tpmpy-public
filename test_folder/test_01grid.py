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
from tensorlib.inference import reconstruct, infer_ats
from functools import reduce


class GridTests(unittest.TestCase):

	def setUp(self):
		"""Call before every test case."""
		pass

	def tearDown(self):
		"""Call after every test case."""
		pass

	def testGridCreation1(self):
		grid = coordinate_grid.create_from_ranges([(0,10,4), (5,8,1)], dim_names=['var1', 'var2'])
		assert grid.get_coords_per_dim() == [[2, 6, 10], [5.5, 6.5, 7.5]]

	def testGridCreation2(self):
		grid = coordinate_grid.create_from_polyranges([[(0,8,4), (7.5,10,1)], [(5,8,1)]], dim_names=['var1', 'var2'])
		assert grid.get_coords_per_dim() == [[2, 6, 8, 9, 10], [5.5, 6.5, 7.5]]

	def testGridCreationManual(self):
		## if you ever wanted to create a set of coordinates manually, just put it in the max value and select a stepsz
		## that is double the range...
		grid = coordinate_grid.create_from_polyranges([[(0, 3, 6), (18, 19, 2)], [(5,8,1)]], dim_names=['var1', 'var2'])
		assert grid.get_coords_per_dim() == [[3, 19], [5.5, 6.5, 7.5]]

	def testGridCreationManual2(self):
		grid = coordinate_grid([(1,3,7), (10,12,15,16)])
		assert np.allclose(grid.get_grid_with_coords(), [
    	    [
    	        [1,10], [1,12], [1,15], [1,16]
    	    ],
    	    [
    	        [3,10], [3,12], [3,15], [3,16]
    	    ],
    	    [
    	        [7,10], [7,12], [7,15], [7,16]
    	    ]
    	])

	def testAggregateClosest(self):
		inputs = [[8,5], [12,4], [2,4]]
		outputs = []
		for v1,v2 in inputs:
			o = v1 + v2
			outputs.append([v1,v2,o])

		grid = coordinate_grid.create_from_ranges([(0,14,2), (3.5,8,1)])

		aggregates, counts = map_points_to_grid(dtensor(outputs), grid, agg='closest', Pc=1, chunk_size=2, verbose=False)

		assert(np.allclose(counts, np.array([[1,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,1,0,0,0], [0,0,0,0,0], [1,0,0,0,0], [0,0,0,0,0]])))
		assert(np.allclose(aggregates, np.array([[6,6,6,6,6], [6,6,6,6,6], [6,13,13,13,13], [13,13,13,13,13], [13,13,13,13,13], [16,16,16,16,16], [16,16,16,16,16]])))

	def testAggregateMean(self):
		inputs = [[8,5], [12,4], [2,4]]
		outputs = []
		for v1,v2 in inputs:
			o = v1 + v2
			outputs.append([v1,v2,o])

		grid = coordinate_grid.create_from_ranges([(0,14,2), (3.5,8,1)])
		aggregates, counts = map_points_to_grid(dtensor(outputs), grid, agg='mean', Pc=2, chunk_size=2, verbose=False)

		assert(np.allclose(counts, np.array([[1,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,1,0,0,0], [0,0,0,0,0], [1,0,0,0,0], [0,0,0,0,0]])))
		assert(np.allclose(aggregates, np.array([[9.5,9.5,9.5,9.5,9.5], [9.5,9.5,9.5,9.5,9.5], [9.5,9.5,9.5,9.5,9.5], [14.5,14.5,14.5,14.5,14.5], [14.5,14.5,14.5,14.5,14.5], [14.5,14.5,14.5,14.5,14.5], [14.5,14.5,14.5,14.5,14.5]])))

	def testAggregateMedian(self):
		inputs = [[8,5], [12,4], [2,4]]
		outputs = []
		for v1,v2 in inputs:
			o = v1 + v2
			outputs.append([v1,v2,o])

		grid = coordinate_grid.create_from_ranges([(0,14,2), (3.5,8,1)])

		aggregates, counts = map_points_to_grid(dtensor(outputs), grid, agg='median', Pc=2, verbose=False)

		assert(np.allclose(counts, np.array([[1,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,1,0,0,0], [0,0,0,0,0], [1,0,0,0,0], [0,0,0,0,0]])))
		assert(np.allclose(aggregates, np.array([[9.5,9.5,9.5,9.5,9.5], [9.5,9.5,9.5,9.5,9.5], [9.5,9.5,9.5,9.5,9.5], [14.5,14.5,14.5,14.5,14.5], [14.5,14.5,14.5,14.5,14.5], [14.5,14.5,14.5,14.5,14.5], [14.5,14.5,14.5,14.5,14.5]])))

	def testAggregateWeightedsum(self):
		inputs = [[8,5], [12,4], [2,4]]
		outputs = []
		for v1,v2 in inputs:
			o = v1 + v2
			outputs.append([v1,v2,o])

		grid = coordinate_grid.create_from_ranges([(0,14,2), (3.5,8,1)])

		aggregates, counts = map_points_to_grid(dtensor(outputs), grid, agg='wsum', Pc=2, verbose=False)

		print(counts)

		print(aggregates)

		assert(np.allclose(counts, np.array([[1,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,1,0,0,0], [0,0,0,0,0], [1,0,0,0,0], [0,0,0,0,0]])))
		assert(np.allclose(aggregates, np.array([[ 6.8672953, 7.1765203, 7.6817713, 8.119817, 8.458645],
			[ 7.1477256, 7.5433683, 8.133918, 8.589774, 8.899495 ],
			[ 9.407831, 9.592169, 9.72924, 9.784118, 9.786797 ],
			[13.661444, 13.491882, 13.623975, 13.831559, 13.991786 ],
			[13.961132, 13.720759, 13.845187, 14.035423, 14.162277 ],
			[15.279241, 15.038868, 14.7573595, 14.598246, 14.521433 ],
			[15.508118, 15.338556, 15.085464, 14.890097, 14.7573595]]))
		)
	
