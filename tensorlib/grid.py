# Coordinate grid class
# Copyright (C) 2025 Corvinus University of Budapest <adambalazs.csapo@uni-corvinus.hu>

import numpy as np
from math import exp
from functools import reduce
from .dtensor import dtensor, unfolded_dtensor
from .metrics import calc_R2

from itertools import product
from collections import defaultdict

from fp_types import progress_bar

from fp_types import AssemblerReducer
from genetic_algo import GeneticAlgorithm
from random import random

from .inference import infer_ats
from .tucker import hosvd


__all__ = [
    'coordinate_grid',
    'create_coordinate_grid',
    'aggregate_over_coordinate_grid'
]

class coordinate_grid:
    """
    A coordinate_grid is an N-dimensional structure that has an ordered set of coordinate values along each of its N dimensions.

    ----------
    Such a grid can have multiple views:
    coords_per_dim          : a list of tuples, such that each tuple corresponds to a different dimension n, listing all of the coordinate values
                            along that dimension
    grid_with_coords        : an N+1-dimensional tensor, such that the coordinates in the first N dimensions correspond to the ordinal values of
                            the coordinates along that dimension, while the last dimension has a length of N and lists the coordinate values
                            (see example below).
    value_tensor_over_grid  : same as grid_with_coords, with the difference that the N+1-th dimension contains some value or set of values
                            either instead of, or besides the coordinate values

    For example, if we had a 2D coordinate grid with coordinate values 1, 3 and 7 in dimension 1, and coordinate values 10, 12, 15 and 16 in
    dimension 2, then:
    - coords_per_dim would be [(1,3,7), (10,12,15,16)]
    - grid_with_coords would be a 3-dimensional tensor of shape (3,4,2) and values:
    [
        [
            [1,10], [1,12], [1,15], [1,16]
        ],
        [
            [3,10], [3,12], [3,15], [3,16]
        ],
        [
            [7,10], [7,12], [7,15], [7,16]
        ]
    ]
    """

    def __init__(self, coords_per_dim, dim_names=None):
        """
        Initializes a coordinate_grid object based on the coords_per_dim parameter.

        Parameters
        ----------
        coords_per_dim      : a list of tuples, such that each tuple corresponds to a different dimension n, listing all of the coordinate values
                            along that dimension

        """
        self.__coords_per_dim = coords_per_dim
        self.__dim_names = dim_names if dim_names is not None else [f"dimension{i}" for i in range(len(coords_per_dim))]

        dimszes = [len(cpd) for cpd in coords_per_dim] + [len(coords_per_dim)]

        grid_w_coords = dtensor(np.zeros(dimszes))

        for combo in product(*coords_per_dim):
            coords = [coords_per_dim[i].index(combo[i]) for i in range(len(combo))]
            grid_w_coords[*coords] = combo

        self.__grid_w_coords = grid_w_coords

    @staticmethod
    def __coordinates_per_dim_for_range(one_range):
        """
        Helper function for classmethods used to create coordinate_grid objects based on ranges.

        Parameters
        ----------
        one_range       : a range of form (min, max, stepsz), this function returns a list of equidistant coordinates
                        of form [c1, c2, ..., cX], where X = (max - min - (stepsz / 2)) / stepsz + 1, such that:        
                        c1 = min + stepsz / 2
                        ...
                        cX = min + stepsz / 2 + (X-1)*stepsz
        """
        (mi, ma, sz) = one_range
        dimsize = int((ma - mi - sz/2) / sz) + 1
        coords = [mi + sz / 2 + i*sz for i in range(dimsize)]
        return coords

    @classmethod
    def create_from_ranges(cls, ranges, dim_names=None):
        """
        Classmethod for creating coordinate_grid objects based on ranges.

        Parameters
        ----------
        ranges      : a list of ranges of form [(min1, max1, stepsz1), ..., (minN, maxN, stepszN)]


        Returns
        -------
        A coordinate_grid object such that:
            - the length of each dimension n (from 1..N) will be (maxn - (minn + stepszn / 2)) / stepszn + 1
            - the length of dimension N+1 will be N
            - the value in each element (e1, e2, ..., eN) will have the form (a1, a2, ..., aN) such that:
                - each an is the value of a coordinate in dimension n.
                - an will range from (minn + stepsz / 2) ... (maxn - stepsz / 2)

        For example, if ranges = [(1, 3, 1), (2, 6, 2)], then the coordinates will be of the form:
        (1.5, 3), (2.5, 3), (1.5, 5), and (2.5, 5)
        """

        coordinates_per_dim = [
            coordinate_grid.__coordinates_per_dim_for_range(one_range)
                for one_range in ranges
        ]

        return cls(coordinates_per_dim, dim_names)

    @classmethod
    def create_from_polyranges(cls, polyranges, dim_names=None):
        """
        Classmethod for creating coordinate_grid objects based on polyranges.

        Parameters
        ----------
        polyranges      : a list of polyranges of form [
                            [(min11, max11, stepsz11), ..., (min1d1, max1d1, stepsz1d1)] ...,       # polyrange 1
                            [(minN1, maxN1, stepszN1), ..., (minNdN, maxNdN, stepszNdN))]           # polyrange 2
                        ]


        Returns
        -------
        A coordinate_grid object such that:
            - the length of each dimension n (from 1..N) will be:
            (maxn1 - (minn1 + stepszn1 / 2)) / stepszn1 + 1 + ... + (maxndn - (minndn + stepszndn / 2)) / stepszndn + 1
            - the length of dimension N+1 will be N
            - the value in each element (e1, e2, ..., eN) will have the form (a1, a2, ..., aN) such that:
                - each an is the value of a coordinate in dimension n.
                - an will belong to one of the ranges within the associated n-th polyrange

        For example, if polyranges = [[(1, 3, 1), (3, 7, 2)], [(2, 6, 2)], then the coordinates will be of the form:
        (1.5, 3), (2.5, 3), (4, 3), (6, 3), (1.5, 5), (2.5, 5), (4, 5), (6, 5)
        """
        coordinates_per_dim = [
            reduce(
                lambda x, y: x + y,
                [coordinate_grid.__coordinates_per_dim_for_range(one_range) for one_range in current_range],
                []
            ) for current_range in polyranges
        ]

        return cls(coordinates_per_dim, dim_names)

    @classmethod
    def create_from_ranges_and_positions_as_percentages(cls, ranges, positions, dim_names=None):
        """
        Classmethod for creating coordinate_grid objects based on ranges and positions along them.

        Parameters
        ----------
        ranges          : a list of ranges of form [(min1, max1), ..., (minN, maxN)]
        positions       : a list of tuples with percentages of form [(pct11, ..., pct1d1), ..., (pctN1, ..., pctNdN)]


        Returns
        -------
        A coordinate_grid object such that:
            - the length of each dimension n (from 1..N) will be dn (the number of percentages in the corresponding dimension
            of positions)
            - the length of dimension N+1 will be N
            - the value in each element (e1, e2, ..., eN) will have the form (a1, a2, ..., aN) such that:
                - each an is the value of a coordinate in dimension n.
                - an will correspond to a point within the n-th range that corresponds to a percentage

        For example, if ranges = [[(1, 3), (2, 6)], and percentages are [(0.5, 0.75), (0.25, 0.75)] then the coordinates will be of the form:
        (2, 3), (2, 5), (2.5, 3), (2.5, 5)
        """
        ndims = len(ranges)

        coords_per_dim = [
            tuple([
                ranges[dim][0] + (pc * (ranges[dim][1] - ranges[dim][0])) for pc in positions[dim]
            ]) for dim in range(ndims)]

        return cls(coords_per_dim, dim_names)

    @classmethod
    def assembler_reducer_creator_r2(cls, *genotype):
        """
        Helper function for genetic_algo_search_for_coordinates_r2() below.

        An AssemberReducer consists of two functions: an assembler and a reducer.
    
        The assembler is a generator function of form: genotype -> FPS
        (where FPS stands for fitness, phenotype, string_representation)

        The reducer shall have the form: (acc, FPS) -> acc

        The point is that when the AssemblerReducer is run (i.e., .run(genotype) is called on the object), then it will:
        a.) run the assembler function on the genotype, such that for each yielded FPS result:
        b.) reducer(acc, FPS) will be called to get the new accumulator - itself an FPS
        In the end, the accumulator that has the best fitness for the genotype will be returned.

        The genetic algorithm below will be run with this function, which will enable different grids to be explored.
        """

        def generate_coordinates_as_percentages_within_ranges(num_coords_per_dim, minmaxes_per_dim):
            positions_as_perc = [[] for _ in num_coords_per_dim]
            for inx, g in enumerate(num_coords_per_dim):
                num_positions = g
                positions = []
                while num_positions > 0:
                    last_position = 0 if len(positions) == 0 else positions[-1]
                    max_position = 1.0 - (num_positions * 0.01)

                    next_position = last_position + (random() * (max_position - last_position))
                    if (next_position - last_position) < 0.01:
                        next_position = last_position + 0.01

                    positions.append(next_position)
                    num_positions -= 1

                positions_as_perc[inx] = positions
            return positions_as_perc

        def assembler(*genotype_to_assemble):
            ndims = (len(genotype_to_assemble) - 3) // 2
            
            ncoords_per_dim = list(genotype_to_assemble[:ndims])
            
            aggregation_type_as_num = genotype_to_assemble[ndims]
            aggregation_type = 'wsum'
            if aggregation_type_as_num == 1:
                aggregation_type = 'closest'
            elif aggregation_type_as_num == 2:
                aggregation_type = 'mean'
            elif aggregation_type_as_num == 3:
                aggregation_type = 'median'

            pc = genotype_to_assemble[ndims+1]
            
            minmaxs_per_dim = list(genotype_to_assemble[ndims+2:-1])
            
            data = genotype_to_assemble[-1]

            for i in range(10):
                ##progress_bar(i+1, 10, progress_msg="Going through a single genotype...")
                percs = generate_coordinates_as_percentages_within_ranges(ncoords_per_dim, minmaxs_per_dim)

                grid = coordinate_grid.create_from_ranges_and_positions_as_percentages(minmaxs_per_dim, percs)

                aggs, counts = map_points_to_grid(data, grid, agg=aggregation_type, Pc=pc, verbose=False) ## can be closest, mean, median or wsum
                aggs_filled = np.nan_to_num(aggs, nan=0) ## now, because the frequency of some gridpoints is 0, we need to fill the nans with zeros
                aggregates = dtensor(aggs_filled)

                Us, S, eigvals = hosvd(aggregates, with_eigvals=True)

                data_reconstructed = infer_ats(S, Us, grid, data[:,:ndims])

                r2_dataset = calc_R2(data[:, -1].reshape(-1,1), data_reconstructed)

                fitness = r2_dataset

                phenotype = percs + minmaxs_per_dim + [aggregation_type] + [pc]

                yield fitness, phenotype, str(phenotype)

        def reducer(acc, cur):
            return cur if acc is None or cur[0] > acc[0] else acc

        ar = AssemblerReducer(assembler, reducer)
        return ar.run(
            genotype, initval=None, show_progress=False
        )

    @classmethod
    def genetic_algo_search_for_coordinates_r2(cls, data, min_and_max_per_dim, population_sz, num_generations, verbose=False, show_progress=True):
        """
        Parameters
        ----------
        data                : numpy array such that each row is a vector of original inputs (with N dimensions) and an output (of 1 dimension).
                            The GA will test for the precision with which the dataset can be reconstructed using aggregation over the generated
                            coordinate grid and aggregation method
        min_and_max_per_dim : a list of tuples containing the minimum and maximum value on each dimension of the problem space. Grid points will be
                            generated over the span of these values in each generation
        population_sz       : population size in GA
        num_generations     : number of generations in GA
        verbose             : whether or not the GA should run in verbose mode
        show_progress       : whether or not the GA should display a progress bar over generations

        This function evolves a population within a GA to find the best coordinate grid and aggregation method possible. Using map_points_to_grid(),
        it evaluates for each genotype what R2 score the data can be reconstructed with.

        The genotype consists of:
        - N tuples of form (min_n, max_n) representing the smallest and largest possible number of coordinates in each dimension. The GA is run with
        values (2,5) for each dimension.
        - an integer value between 1 and 4 corresponding to closest, mean, median or weightedsum aggregation
        - a Pc value ranging between 2 and 10, corresponding to the minimum required number of data points for each gridpoint
        - N tuples of form (min_n, max_n) representing the smallest and largest possible coordinate value in each dimension
        - data (the data parameter from this function)

        The reward is the R2 score obtained when trying to reconstruct the data using the derived rule set


        Returns
        -------
        GeneticAlgorithm object that can be queried for genotype and phenotype of best entity that was found.
        """

        num_dims = data.shape[1] - 1
        assert num_dims == len(min_and_max_per_dim), "Number of dimensions are off in coordinate_grid.create_with_genetic_algorithm()"

        ga = GeneticAlgorithm(
            population_sz=population_sz,
            survival_rate=0.5,
            mutation_base_prob=0.1,
            mutation_incr_fitness_limit=0.25, # if fitness lower than this (as well as std dev lower than next parameter), increase mutation probability
            mutation_incr_stdev_limit=0.1,
            mutation_max=0.15, # maximum mutation probability
            num_generations=num_generations,
            max_admissible_fitness=1.01, # better fitnesses will not be accepted for fear of overfitting
            fitness_when_max_exceeded=lambda x: -100 # instead, set the fitness for such entities to -100
        )
        
        ga.run(
            coordinate_grid.assembler_reducer_creator_r2, ## function that creates the assembler-reducer
            genotype_param_types = [
                'int' for _ in range(num_dims)] +
                ['int'] + # closest, mean, median or wsum, i.e. 1-4
                ['int'] + # Pc = 2 to 10
                ['fixed' for _ in range(num_dims)] +
                ['fixed'
            ], # types of variables in genotype
            genotype_param_specs = [
                (2, 5) for _ in range(num_dims)] +
                [(1, 4)] +
                [(2, 10)] + 
                min_and_max_per_dim +
                [data], # parameters for generating those variables
            verbose=verbose,
            show_progress=show_progress
        )
        
        print("top top entity:")
        print(ga.top_top_entity)

        return ga


    def get_grid_with_coords(self, copy=True):
        """
        Getter function for grid_with_coords, specified as an N+1-dimensional tensor, such that
        the coordinates in the first N dimensions correspond to the ordinate values of the coordinates
        along that dimension, while the last dimension has a length of N and lists the coordinate values.

        copy    : if True, returns a deep copy of the tensor; otherwise returns a reference


        Returns
        -------
        grid_w_coords  : self.__grid_w_coords
        """
        if copy:
            return self.__grid_w_coords.copy()
        else:
            return self.__grid_w_coords

    def get_coords_per_dim(self):
        """
        Getter function for coords_per_dim, specified as a list of tuples, such that each tuple
        corresponds to a different dimension n, listing all of the coordinate values along that dimension


        Returns
        -------
        coords_per_dim  : self.__coords_per_dim
        """
        return self.__coords_per_dim

    def get_dim_names(self):
        """
        Getter function for self.__dim_names
        """
        return self.__dim_names


def __closest_grid_indices(coords, grid_pts,
                         row_chunk=5_000, grid_chunk=20_000,
                         dtype=np.float64, verbose=True):
    """
    Return for every data row its closest grid-point index.

    Parameters
    ----------
    coords      : (P, D)   data coordinates
    grid_pts    : (G, D)   grid coordinates
    row_chunk   : rows processed in one pass   (tune for RAM)
    grid_chunk  : grid pts processed in one pass
    dtype       : float32 keeps RAM small, enough for distances


    Returns
    -------
    A vector of length P containing the closest indices
    """
    P, G = coords.shape[0], grid_pts.shape[0]

    # pre-compute squared norms once
    row_norm2  = np.sum(coords.astype(dtype)**2,     axis=1)       # (P,)
    grid_norm2 = np.sum(grid_pts.astype(dtype)**2,   axis=1)       # (G,)

    best_dist2 = np.full(P, np.inf,   dtype=dtype)
    best_idx   = np.empty(P,          dtype=np.int32)

    for g0 in range(0, G, grid_chunk):
        g1      = min(g0 + grid_chunk, G)
        gp_chunk   = grid_pts[g0:g1].astype(dtype)                 # (Gc, D)
        gp_n2      = grid_norm2[g0:g1]                             # (Gc,)

        # (coords · gp_chunk.T) is the only large matrix, shape (row_chunk, Gc)
        for r0 in range(0, P, row_chunk):
            if verbose:
                progress_bar(g0*P+r0+1, G*P, progress_msg="Getting closest gridpoint for each data point")

            r1      = min(r0 + row_chunk, P)
            rows    = coords[r0:r1].astype(dtype)                  # (Rc,D)
            dot     = rows @ gp_chunk.T                            # (Rc,Gc)
            # use broadcasting to get squared distances
            dist2   = row_norm2[r0:r1, None] + gp_n2[None, :] - 2.0 * dot

            # local minima inside this (Rc,Gc) slice
            idx_local  = np.argmin(dist2, axis=1)                  # (Rc,)
            dist_local = dist2[np.arange(r1-r0), idx_local]

            # keep global minima per row
            better = dist_local < best_dist2[r0:r1]
            best_dist2[r0:r1][better] = dist_local[better]
            best_idx[r0:r1][better]   = idx_local[better] + g0     # offset

    return best_idx            # shape (P,) dtype int32


def __compute_point_to_grid_mapping(X, grid_points, agg='mean', Pc=1,
                                  chunk_size=1_000,
                                  primary_counts_only=False,
                                  verbose=True):
    """
    Parameters
    ----------
    X           : (P, D+1) ndarray  – D coords + 1 value
    grid_points : (G, D)  ndarray  – full list of grid coordinates
    agg         : 'closest' | 'mean' | 'median' | 'wsum'
    Pc          : minimum rows per grid point for aggregation (does NOT affect counts)
    chunk_size  : rows processed in one distance batch
    primary_counts_only: if True, just returns None, primary_counts (no aggregation is performed and Pc is disregarded)


    Returns
    -------
    aggregates  : flat length-G ndarray
    counts      : flat length-G ndarray  (sum == P, independent of Pc)
    """
    P, G = X.shape[0], grid_points.shape[0]
    coords   = X[:, :-1]
    yvalues  = X[:, -1]

    # --- 1. closest grid-point for every data row (primary associations) ---
    closest_grid = __closest_grid_indices(coords, grid_points,
                                    row_chunk=chunk_size,
                                    grid_chunk=chunk_size,
                                    dtype=np.float64,
                                    verbose=verbose)

    # --- 2. bucket rows by grid-point --------------------------------------
    assignments = [[] for _ in range(G)]
    for row, g in enumerate(closest_grid):
        if verbose:
            progress_bar(row+1, P, progress_msg="Bucketing all datapoints by gridpoint based on primary associations")
        assignments[g].append(row)

    # >>> grab primary counts before any Pc top-up  <<<
    primary_counts = np.fromiter((len(a) for a in assignments),
                                 dtype=np.int32, count=G)

    if primary_counts_only:
        return None, primary_counts

    # --- 3. ensure at least Pc rows (does NOT change counts) ---------------
    if Pc > 0:
        for g in range(G):
            if verbose:
                progress_bar(g+1, G, progress_msg="Bucket further datapoints by gridpoint based on secondary associations (where necessary)")
            if len(assignments[g]) < Pc:
                d = np.linalg.norm(coords - grid_points[g], axis=1)
                for idx in np.argsort(d):
                    if idx not in assignments[g]:
                        assignments[g].append(idx)
                        if len(assignments[g]) == Pc:
                            break

    # --- 4. aggregate -------------------------------------------------------
    aggregates = np.zeros(G, dtype=np.float32)

    if agg == 'closest':
        for g, rows in enumerate(assignments):
            if verbose:
                progress_bar(g+1, G, progress_msg="Aggregate")
            if rows:                       # might still be empty if Pc==0
                # choose the one truly closest (first row in assignments is ok too)
                d = np.linalg.norm(coords[rows] - grid_points[g], axis=1)
                aggregates[g] = yvalues[rows[np.argmin(d)]]
            else:
                aggregates[g] = np.nan
    elif agg == 'mean':
        for g, rows in enumerate(assignments):
            if verbose:
                progress_bar(g+1, G, progress_msg="Aggregate")
            if rows:
                aggregates[g] = yvalues[rows].mean()
            else:
                aggregates[g] = np.nan
    elif agg == 'median':
        for g, rows in enumerate(assignments):
            if verbose:
                progress_bar(g+1, G, progress_msg="Aggregate")
            if rows:
                aggregates[g] = np.median(yvalues[rows])
            else:
                aggregates[g] = np.nan
    elif agg == 'wsum':                             # weighted mean
        eps = 1e-12                                  # small constant for safety
        for g, rows in enumerate(assignments):
            if verbose:
                progress_bar(g + 1, G, progress_msg="Aggregate (wsum)")
            if rows:
                # distances of the selected rows to grid-point g
                d = np.linalg.norm(coords[rows] - grid_points[g], axis=1)
                w = 1.0 / (d + eps)                  # larger weight → closer point
                aggregates[g] = np.average(yvalues[rows], weights=w)
            else:
                aggregates[g] = np.nan
    else:
        raise ValueError(f"unknown agg='{agg}'")

    # --- return aggregates & the ORIGINAL unique counts --------------------
    return aggregates, primary_counts


def map_points_to_grid(X, grid, agg='mean', Pc=1, chunk_size=10000, primary_counts_only=False, verbose=True):
    """
    Parameters
    ----------
    X           : (P, D+1) ndarray  – P data points, each represented by D coordinates (features) and 1 value
    grid        : grid object with D dimensions
    agg         : 'closest' | 'mean' | 'median' | 'wsum'
    Pc          : minimum number of data points to be associated with each gridpoint
    chunk_size  : maximum number of data points processed in one batch (when calculating distances from gridpoints)
    primary_counts_only: if True, just returns None, primary_counts (no aggregation is performed and Pc is disregarded)

    This function associates each data point in X with 1 or more gridpoints. Then, at each gridpoint, the values of the
    data points associated with it are aggregated into a single value.

    By default, each data point is associated with 1 and only 1 gridpoint (we refer to this as the data point's primary association).
    However, if for any gridpoint the number of data points in primary association with it is less than Pc, further data points
    (already primarily associated with another gridpoint) can become associated with it (we refer to thsi as the data point's secondary association).


    Returns
    -------
    aggregates  : numpy array with D dimensions, with an aggregated value per grid coordinate
    counts      : numpy array with D dimensions, with a count per grid coordinate. This count refers only to primary associations
    """
    coords_per_dim = grid.get_coords_per_dim()
    grid_shape = tuple(len(c) for c in coords_per_dim)
    grid_points = np.array(list(product(*coords_per_dim)), dtype=X.dtype)

    aggregates, counts = __compute_point_to_grid_mapping(X, grid_points, agg, Pc, chunk_size, primary_counts_only, verbose)
    if aggregates is not None:
        aggregates = aggregates.reshape(grid_shape)
    counts = counts.reshape(grid_shape)
    return aggregates, counts



