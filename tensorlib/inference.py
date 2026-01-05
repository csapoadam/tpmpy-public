# Tools for TP system reconstruction, rule generation and inference from TP system and rules
# Copyright (C) 2025 Corvinus University of Budapest <adambalazs.csapo@uni-corvinus.hu>

import numpy as np
from .dtensor import dtensor
from .ruleset import RuleSet
from .metrics import calc_R2
from fp_types import progress_bar, AssemblerReducer
from genetic_algo import GeneticAlgorithm
from functools import reduce
from itertools import product, groupby
from operator import itemgetter
import random

# Copyright (C) 2025 Adam Csapo <adambalazs.csapo@uni-corvinus.hu>

__all__ = [
    'reconstruct',
    'infer_ats',
    'rulify',
    'infer_from_ruleset'
]


def reconstruct(S, Us):
    """
    Reconstructs a TP model based on core tensor and weighting functions, returning the full result.

    Parameters
    ----------
    - S         : a dtensor with N dimensions
    - Us        : a list containing at least N values, such that each value is either a numpy matrix, or a None


    Returns
    -------
    - X = S tprod_1toN Us
    """
    if len(Us) < S.ndim:
        raise ValueError("inference.reconstruct(): Us must have at least S.ndim elements")

    res = S.copy()
    for dim in range(len(S.shape)):
        if Us[dim] is not None:
            res = res.ttm(Us[dim], dim)
    return res


def __interpolate_input_over_coords_and_values(inputt, over_coordinates, over_values):
    """
    Helper function for __get_weights_per_dim_for_inputvec.

    Parameters
    ----------
    - inputt                : a single coordinate (scalar value, int or float)
    - over_coordinates      : a tuple or list of coordinates in a given dimension
    - over_values           : a matrix with as many rows as the length of over_coordinates,
                              containing a list of weights at each coordinate


    Returns
    -------
    - result                : a row vector that arises as an interpolation within over_values, with length R (rank of weighting matrix)
    """
    if inputt < over_coordinates[0]:
        a = over_values[1, :]
        b = over_values[0, :]
        result = ((b - a) / (over_coordinates[1] - over_coordinates[0])) * (over_coordinates[0] - inputt) + b

    elif inputt > over_coordinates[-1]:
        a = over_values[-2, :]
        b = over_values[-1, :]
        result = ((b - a) / (over_coordinates[-1] - over_coordinates[-2])) * (inputt - over_coordinates[-1]) + b
    else:
        sentinel = False
        for coordinx in range(len(over_coordinates) - 1):
            if inputt >= over_coordinates[coordinx] and inputt < over_coordinates[coordinx+1]:
                a = over_values[coordinx+1, :]
                b = over_values[coordinx, :]
                result = ((b - a) / (over_coordinates[coordinx+1] - over_coordinates[coordinx])) * (over_coordinates[coordinx+1] - inputt) + a
                sentinel=True
                break
        if not sentinel:
            ### I don't think this is possible(?)
            result = over_values[len(over_coordinates)-1, :]
    return result.reshape(1,-1)


def __get_weights_per_dim_for_inputvec(S, Us, inputvec, coords_per_dim):
    """
    Given a TP model (based on core tensor and weighting functions), an input vector and a list of tuples containing
    coordinates for each input dimension, interpolates between weights at the closest coordinates bounding the input
    vector to get the weights for the given input vector.

    Parameters
    ----------
    - S                 : a dtensor with N dimensions
    - Us                : a list containing at least N values, such that each value is either a numpy matrix, or a None
    - inputvec          : an array-like containing N values representing an input along each dimension
    - coords_per_dim    : a list of N tuples, with each tuple containing coordinate values along the given dimension.
                          This input is usually obtained through coordinate_grid.get_coords_per_dim()


    Returns
    -------
    - weights_per_dim   : a list of N numpy vectors (each 1-by-Rank_n-dimensional)
    """
    weights_per_dim = []
    n_dims = len(coords_per_dim)

    for dim in range(n_dims):
        if (len(coords_per_dim[dim])) < 2:
            raise Exception("infer_ats: cannot use a grid with a dimension that has less than 2 coordinates")

    weights_per_dim = [
        __interpolate_input_over_coords_and_values(inputvec[dim], coords_per_dim[dim], Us[dim])
        for dim in range(n_dims)
    ]

    return weights_per_dim


def infer_ats(S, Us, grid, inputs):
    """
    Infer outputs from a TP model over a coordinate grid based on a set of input vectors.

    Parameters
    ----------
    - S                 : a dtensor with N dimensions
    - Us                : a list containing at least N values, such that each value is either a numpy matrix, or a None
    - grid              : a coordinate_grid over which the weighting functions are defined
    - inputs            : a matrix of size P-by-N, if S is N-dimensional

    
    Returns
    -------
    results             : a column vector with inferred outputs
    """
    results = np.zeros((inputs.shape[0], 1))

    coords_per_dim = grid.get_coords_per_dim()

    for p in range(inputs.shape[0]):
        inputvec = inputs[p,:]

        weights_per_dim = __get_weights_per_dim_for_inputvec(S, Us, inputvec, coords_per_dim)

        inference = reconstruct(S, weights_per_dim).flatten()
        results[p,:] = inference
    return results


def infer_from_ruleset(rule_set, inputs, cutoff_weight=None):
    """
    Infer outputs from a rule set based on a set of input vectors.

    Parameters
    ----------
    - rule_set          : an object of type RuleSet (such as that returned by rulify())
    - inputs            : a matrix of size P-by-N, if S is N-dimensional
    - cutoff_weight     : if specified, then only rules are taken into consideration whose weight >= cutoff_weight

    
    Returns
    -------
    results             : a column vector with inferred outputs
    """
    results = np.zeros((inputs.shape[0], 1))

    for p in range(inputs.shape[0]):
        inputvec = np.asarray(inputs[p,:], dtype=float)

        # Midpoints per dimension
        mids_of_antranges_per_dim = [np.array([(lo + hi) / 2.0 for (lo, hi) in dim], dtype=float)
                    for dim in rule_set.antecedent_ranges]

        # Build Cartesian product of midpoints (combos) and the corresponding index grid
        grids = np.meshgrid(*mids_of_antranges_per_dim, indexing="ij")
        combo_points = np.stack([g.reshape(-1) for g in grids], axis=1)  # (N, D)

        index_grids = np.meshgrid(*[np.arange(len(m)) for m in mids_of_antranges_per_dim], indexing="ij")
        combo_indices = np.stack([g.reshape(-1) for g in index_grids], axis=1)  # (N, D)

        # Distances
        diff = combo_points - inputvec  # (num_combos, N)
        dists = np.linalg.norm(diff, axis=1)
        

        # Sort midpoint combos and their indices by distance
        order = np.argsort(dists)
        distances = dists[order]
        combos = combo_points[order]
        combo_indices = combo_indices[order]

        is_done = False
        for cinx, combo_inx in enumerate(combo_indices):
            if is_done:
                break
            
            for rinx,(antecedent, consequent) in enumerate(rule_set.rules):
                weight = rule_set.weights[rinx]
                if cutoff_weight is None or weight >= cutoff_weight:
                    if np.array_equal(combo_inx, antecedent):
                        results[p, :] = consequent
                        is_done = True
                        break

        if not is_done:
            raise ValueError(f"could not find consequent for {inputvec} in row {p}")
        
    return results



def rulify(S, Us, grid, antecedent_ranges, counts, n_samples=1):
    """
    Parameters
    ----------
    S                   : core tensor of TP model
    Us                  : list of weighting matrices 
    grid                : coordinate_grid with N dimensions (input dimensions only)
    antecedent_ranges   : list of N sublists, such that each sublist contains any number of (min,max) tuples defining antecedent ranges along dimension n
    counts              : numpy array with N dimensions, with a count per grid coordinate (see 2nd return value of map_points_to_grid() in grid.py)
    n_samples           : determines how many samples are taken per antecedent dimension within each antecedent range

    This function generates a RuleSet (see ruleset.py) derived from the TP model (S, Us) over antecedent ranges (antecedent_ranges) in the grid (grid),
    with weights determined by counts


    Returns
    -------
    RuleSet     : the resulting RuleSet
    """
    indices_and_samples_per_dim = [
        [(inx, [mi + (i / n_samples+1) * (ma - mi) for i in range(1,n_samples+1)])
            for inx, (mi, ma) in enumerate(mimas)] for mimas in antecedent_ranges
    ]

    rules = []
    total_num_datapoints = int(sum(counts.flatten()))

    for index_samples_combo in product(*indices_and_samples_per_dim):
        antecedent = []
        consequents = []
        mean_consequent = 0.0
        weight = 0.0

        weightinxes = []
        samples = []

        for dimension, wfun_index_and_samples in enumerate(index_samples_combo):
            antecedent.append(wfun_index_and_samples[0])
            samples.append(wfun_index_and_samples[1])
            
            coordinates = grid.get_coords_per_dim()[dimension]
            distances = [(wfun_index_and_samples[1][n_samples//2] - c)**2 for c in coordinates]
            closestinx = np.argmin(distances)
            weightinxes.append(closestinx)

        for prod in product(*samples):
            consequents.append(
                float(
                    np.ravel(
                        infer_ats(S, Us, grid, np.array([prod]))
                    )[0]
                )
            )

        consequent = float(np.mean(consequents))
        
        weight = float(counts[*weightinxes] / total_num_datapoints)
        rules.append((antecedent, consequent, weight))
        
    rules = sorted(rules, key=lambda x: x[2], reverse=True)

    return RuleSet(
        rules = [(antecedent, consequent) for (antecedent, consequent, _) in rules],
        antecedent_ranges = antecedent_ranges,
        weights = [w for (_, _, w) in rules],
        counts = counts
    )



def __assembler_reducer_creator_runner(*genotype):
    """
    Helper function for genetic_algo_rulify() below.

    An AssemberReducer consists of two functions: an assembler and a reducer.
    
    The assembler is a generator function of form: genotype -> FPS
    (where FPS stands for fitness, phenotype, string_representation)

    The reducer shall have the form: (acc, FPS) -> acc

    The point is that when the AssemblerReducer is run (i.e., .run(genotype) is called on the object), then it will:
    a.) run the assembler function on the genotype, such that for each yielded FPS result:
    b.) reducer(acc, FPS) will be called to get the new accumulator - itself an FPS
    In the end, the accumulator that has the best fitness for the genotype will be returned.

    The genetic algorithm below will be run with this function, which will enable different rulesets to be explored.
    """
    def assembler(*genotype_to_assemble):
        min_weighting_func_val = genotype_to_assemble[0]
        min_rule_weight = genotype_to_assemble[1]
        samples_consequents = genotype_to_assemble[2]

        data = genotype_to_assemble[-5]

        n_rows = data.shape[0]
        n_dims = data.shape[1]-1

        max_antecedent_nums = [genotype_to_assemble[i] for i in range(3, 3+n_dims)]

        num_trials_to_aggregate = genotype_to_assemble[-7]
        pct_rows = genotype_to_assemble[-6]
        S = genotype_to_assemble[-4]
        Us = genotype_to_assemble[-3]
        grid = genotype_to_assemble[-2]
        counts = genotype_to_assemble[-1]

        coords_per_dim = grid.get_coords_per_dim()

        k = int(pct_rows * n_rows)  # how many random rows you want

        # In each iteration, choose k random row indices (without replacement by default)
        # Then generate the antecedents based on those data points as samples
        for i in range(num_trials_to_aggregate):

            antecedent_wfuninx_ranges = [[] for _ in range(len(coords_per_dim))]
            
            indices = np.random.choice(n_rows, size=k, replace=False)

            for p in indices:
                inputvec = data[p,:-1]
                weights_per_dim = __get_weights_per_dim_for_inputvec(S, Us, inputvec, coords_per_dim)

                for diminx, dimweights in enumerate(weights_per_dim):
                    
                    if np.max(dimweights.flatten()) > min_weighting_func_val:
                        admissible_wfun_inx = np.argmax(dimweights.flatten())

                        awrs = antecedent_wfuninx_ranges[diminx].copy()

                        awrs.append((inputvec[diminx]-0.1, inputvec[diminx]+0.1, admissible_wfun_inx))

                        uniques = list(set(awrs))

                        if len(uniques) > len(antecedent_wfuninx_ranges[diminx]):
                            uniques_sorted_by_awfun_inx = sorted(uniques, key=lambda x: x[2])
                            uniques_sorted_by_awfun_inx_and_antecedent_a = sorted(uniques_sorted_by_awfun_inx, key=lambda x: x[0])

                            result = [
                                (g[0][0], max(g[-1][1], g[0][1]), k)
                                    for k, grp in groupby(
                                        uniques_sorted_by_awfun_inx_and_antecedent_a,
                                        key=itemgetter(2)
                                    ) for g in [list(grp)]
                            ]

                            if len(result) < max_antecedent_nums[diminx] + 1:
                                antecedent_wfuninx_ranges[diminx] = result

            antecedent_ranges = [
                [
                    (item[0], item[1]) for item in antecedent_wfuninx_ranges[diminx]
                ] for diminx in range(len(antecedent_wfuninx_ranges))
            ]

            rule_set = rulify(S, Us, grid, antecedent_ranges, counts, n_samples=samples_consequents)

            try:
                data_reconstructed = infer_from_ruleset(rule_set, data[:,:-1], cutoff_weight=min_rule_weight)
                r2_dataset = calc_R2(data[:, -1].reshape(-1,1), data_reconstructed)
                fitness = r2_dataset
            except ValueError as e: ## if inference from rule set not possible
                fitness = -1

            phenotype = [min_weighting_func_val, min_rule_weight, samples_consequents, max_antecedent_nums, indices, antecedent_ranges, rule_set]

            yield (
                fitness,
                phenotype,
                f"min wf val: {min_weighting_func_val}; min rule w: {min_rule_weight}; max num of antecedents: {max_antecedent_nums}; samples for creating rules: {samples_consequents}; pct of datapts: {pct_rows}; datapts considered: {indices}; antecedent ranges: {antecedent_ranges}"
            )

    def reducer(acc, cur):
        return cur if acc is None or cur[0] > acc[0] else acc

    ar = AssemblerReducer(assembler, reducer)
    return ar.run(
        genotype, initval=None #, show_progress=True, num_steps=20
    )


def genetic_algo_rulify(S, Us, grid, counts, data, population_sz, num_generations, dict_of_params):
    """
    Parameters
    ----------
    S                   : core tensor of TP model
    Us                  : list of weighting matrices 
    grid                : coordinate_grid with N dimensions (input dimensions only)
    counts              : numpy array with N dimensions, with a count per grid coordinate (see 2nd return value of map_points_to_grid() in grid.py)
    data                : numpy array such that each row is a vector of original inputs and an output. The GA will test for the precision with which the dataset
                        can be reconstructed using the generated rules
    population_sz       : population size in GA
    num_generations     : number of generations in GA

    This function evolves a population within a GA to find the best rules possible.
    The genotype consists of:
    - minwfval (between 0 and 1): minimum value for weighting function (ideally Us has matrices of CNO type, so if any weighting function value exceeds this minimum,
    that region of the input domain is used as an antecedent). Note, however, that the weighting function values are only checked for the input data points in data.
    - minrulew (between 0 and 1): minimum weight with which a rule needs to be represented in the rule set in order for it to be kept

    The reward is the R2 score obtained when trying to reconstruct the data using the derived rule set


    Returns
    -------
    GeneticAlgorithm object that can be queried for genotype and phenotype of best entity that was found.
    """
    ga = GeneticAlgorithm(
        population_sz=population_sz,
        survival_rate=0.5,
        mutation_base_prob=0.1,
        mutation_incr_fitness_limit=0.1, # if fitness lower than this (as well as std dev lower than next parameter), increase mutation probability
        mutation_incr_stdev_limit=0.1,
        mutation_max=0.15, # maximum mutation probability
        num_generations=num_generations,
        max_admissible_fitness=1.00, # better fitnesses will not be accepted for fear of overfitting
        fitness_when_max_exceeded=lambda x: -100 # instead, set the fitness for such entities to -100
    )
        
    ga.run(
        __assembler_reducer_creator_runner, ## function that creates the assembler-reducer
        genotype_param_types = ['float'] + ## minimum weighting function val
            ['float'] + ## minimum rule weight
            ['int'] + ## number of samples for rule creation
            ['int' for _ in range(data.shape[1]-1)] + ## maximum number of antecedent ranges per dimension
            ['fixed'] + ## number of trials when sampling for rule creation
            ['float'] + #percentage of data points considered
            ['fixed', 'fixed', 'fixed', 'fixed', 'fixed'], # types of variables in genotype
        genotype_param_specs = [dict_of_params['min_wfun_val_ab']] +
            [dict_of_params['min_rule_weight_ab']] +
            [dict_of_params['num_samples_ab']] +
            dict_of_params['max_antecedent_nums_ab_per_dim'] +
            [dict_of_params['num_trials_to_aggregate'] if 'num_trials_to_aggregate' in dict_of_params else 20] +
            [dict_of_params['pct_data_points_ab']] +
            [data, S, Us, grid, counts],
        verbose = False,
        show_progress = True
    )
        
    print("top top entity:")
    print(ga.top_top_entity)

    return ga

