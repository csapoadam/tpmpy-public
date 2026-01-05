# Generic implementation of genetic algorithm - to be used for optimal gridding and rule set generation
# Copyright (C) 2025 Corvinus University of Budapest <adambalazs.csapo@uni-corvinus.hu>

import random
import statistics
from fp_types.progress_bar import progress_bar


class Entity:
    def __init__(self, genotype, phenotype=None, fitness=None, str_repr=None, max_admissible_fitness=0.1, fitness_when_max_exceeded=lambda x: x):
        """
        Parameters
        ----------
        genotype                   : parameters used to generate one or more phenotypes
        phenotype                  : phenotype with the best fitness generated based on the genotype
        fitness                    : fitness of the best phenotype
        str_repr                   : a string representation of the genotype, phenotype and fitness used to display info in GA.run()
        max_admissible_fitness     : if the fitness is greater than this, do not accept it (for fear of overfitting)
        fitness_when_max_exceeded  : if fitness > max_admissible_fitness, use this function to get 'real' fitness value

        Helper class for GeneticAlgorithm (GA) class below, representing entities in the population.
        """
        self.__data = [genotype, (fitness, phenotype, str_repr)]
        self.max_admissible_fitness = max_admissible_fitness
        self.fitness_when_max_exceeded = fitness_when_max_exceeded

    def update_thru_aggregator(self, fitness_phenotype_str):
        self.__data[1] = fitness_phenotype_str

    def get_fitness(self):
        """
        Returns the fitness of the entity
        """
        fitness_base = self.__data[1][0]
        if fitness_base is not None and fitness_base > self.max_admissible_fitness:
            return self.fitness_when_max_exceeded(fitness_base)
        
        return fitness_base

    def get_genotype(self):
        return self.__data[0]

    def get_phenotype(self):
        return self.__data[1][1]

    def __str__(self):
        if self.__data[1][2] is not None:
            return self.__data[1][2]
        else:
            return "None"


class GeneticAlgorithm:
    """
    A generic class that runs a Genetic Algorithm based on an injected 'Assembler' and 'Reducer'.

    Key operating parameters
    ----------
    - Assembler             : takes a genotype, creates P phenotypes based on it and yields the corresponding FPS values one by one
                              (where FPS stands for fitness, phenotype, string_representation)
    - Reducer               : reduces all of the P FPS values yielded by the Assembler to get the one with the highest fitness
    - ar_creator_runner     : a function that creates an AssemblerReducer object, runs it and returns an FPS; that is, a 3-tuple containing
                            - a fitness value
                            - a phenotype
                            - a string representation of some aspects of the genotype, phenotype and fitness based on which the solution can
                              be reconstructed
    """

    def __init__(
            self, population_sz, survival_rate, mutation_base_prob, mutation_incr_fitness_limit,
            mutation_incr_stdev_limit, mutation_max, num_generations=10,
            max_admissible_fitness=0.1, fitness_when_max_exceeded=lambda x: x):
        """
        Parameters
        ----------
        - population_sz                 : population size
        - survival_rate                 : float between 0 and 1
        - mutation_base_prob            : float between 0 and 1 that serves as a base probability for mutation of each gene
        - mutation_incr_fitness_limit   : fitness value below which mutation rate will be increased by 10% (unless we have reached mutation_max)
        - mutation_incr_stdev_limit     : stdev of fitness values below which mutation rate will be increased by 10% (unless we have reached mutation_max)
        - num_generations               : how many generations to model
        - max_admissible_fitness        : if the fitness value is greater than this, do not accept it (for fear of overfitting)
        - fitness_when_max_exceeded     : if fitness value is inadmissible (see max_admissible_fitness), use this function to get the 'real' fitness
        """
        self.population_sz = population_sz
        self.survival_rate = survival_rate
        self.mutation_base_prob = mutation_base_prob
        self.mutation_prob = mutation_base_prob
        self.mutation_incr_fitness_limit = mutation_incr_fitness_limit
        self.mutation_incr_stdev_limit = mutation_incr_stdev_limit
        self.mutation_max = mutation_max
        self.max_admissible_fitness = max_admissible_fitness
        self.fitness_when_max_exceeded = fitness_when_max_exceeded

        self.n_generations = num_generations
        self.population = None
        self.top_top_entity = None
        self.simulation_complete = False

    def __generate_parameter_value(self, p_type, p_spec):
        """
        Generates a value for parameter of type p_type constrained by p_spec.
        
        Parameters
        ----------
        - p_type                : type of value to be generated. p_type can be any one of the following:
                - 'fixed'       : p_spec is 1 specific value which is always chosen
                - 'int'         : p_spec is a tuple of a minimum and a maximum (integer) value (uniform sampling is performed)
                - 'float'       : p_spec is a tuple of a minimum and a maximum (float) value (uniform sampling is performed)
                - 'float2incr'  : generates a tuple of floats such that the second is >= the first, using uniform sampling
                                  between the first and second values in p_spec
                - 'float3incr'  : generates a tuple of floats such that the third is >= the second, and the second >= the first, using
                                  uniform sampling between the first and second values in p_spec for the first two values, and between
                                  the second and third values in p_spec for the third value
        - p_spec                : hyperparameters of some format that corresponds to p_type.
        """
        p_value = None
        if p_type == 'fixed':
            p_value = p_spec
        elif p_type == 'int':
            minv = p_spec[0]
            maxv = p_spec[1]
            p_value = random.randint(minv, maxv)
        elif p_type == 'float':
            minv = p_spec[0]
            maxv = p_spec[1]
            p_value = random.random() * (maxv - minv) + minv
        elif p_type == 'float2incr':
            minv = p_spec[0]
            maxv = p_spec[1]
            p_value1 = random.random() * (maxv - minv) + minv
            p_value2 = random.random() * (maxv - minv) + minv
            if p_value1 > p_value2:
                p_value = (p_value2, p_value1)
            else:
                p_value = (p_value1, p_value2)
        elif p_type == 'float3incr':
            minv = p_spec[0]
            maxv = p_spec[1]
            p_value1 = random.random() * (maxv - minv) + minv
            p_value2 = random.random() * (maxv - minv) + minv
            p_value3 = random.random() * (p_spec[2] - p_spec[1]) + p_spec[1]
            if p_value1 > p_value2:
                p_value = (p_value2, p_value1, p_value3)
            else:
                p_value = (p_value1, p_value2, p_value3)

        return p_value

    def __recombine_and_mutate(self, genes_1, genes_2, param_types, param_specs):
        """
        Recombines 2 genotypes in a random position (from the second to the penultimate) to obtain 2 children.
        Each chromosome in each child is mutated with some mutation probability (self.mutation_prob).
        In this case, a new parameter value is generated based on param_types and param_specs.
        """
        splitpos = random.randint(1, len(genes_1) - 1)
        child_1 = genes_1[:splitpos] + genes_2[splitpos:]
        child_2 = genes_2[:splitpos] + genes_1[splitpos:]

        for c_ix in range(len(child_1)):
            do_mutate_1 = (random.random() < self.mutation_prob)
            do_mutate_2 = (random.random() < self.mutation_prob)
            
            if do_mutate_1:
                child_1[c_ix] = self.__generate_parameter_value(param_types[c_ix], param_specs[c_ix])
            
            if do_mutate_2:
                child_2[c_ix] = self.__generate_parameter_value(param_types[c_ix], param_specs[c_ix])

        return child_1, child_2
        

    def __step_generation(self, param_types, param_specs):
        """
        Creates a new generation based on an older one by keeping only the best self.survival_rate entities,
        and generating new children in place of the entities that have died out.
        """

        ### if the simulation has just started, self.population will be None and all entities have to be
        ### generated randomly
        if self.population is None:
            self.population = [None for _ in range(self.population_sz)]
            for entity_ix in range(self.population_sz):
                genotype = [self.__generate_parameter_value(param_types[p_ix], param_specs[p_ix]) for p_ix in range(len(param_types))]
                self.population[entity_ix] = Entity(
                    genotype=genotype,
                    max_admissible_fitness=self.max_admissible_fitness,
                    fitness_when_max_exceeded=self.fitness_when_max_exceeded
                )
        ### otherwise, sort the population in reverse order based on the fitness values, keeping the first self.survival_rate percent
        ### then fill in the remaining spots in the population by sampling a 'mother' and 'father' randomly and recombining them
        else:
            self.population.sort(key=lambda entity: entity.get_fitness(), reverse=True)
            self.population = self.population[:int(self.population_sz * self.survival_rate)]

            new_population = self.population.copy()
            while True:
                if len(new_population) >= self.population_sz:
                    new_population = new_population[:self.population_sz]
                    break
                mama, papa = random.sample(self.population, 2)
                child_1, child_2 = self.__recombine_and_mutate(
                    mama.get_genotype(),
                    papa.get_genotype(),
                    param_types,
                    param_specs
                )
                new_population.append(Entity(
                    genotype=child_1,
                    max_admissible_fitness=self.max_admissible_fitness,
                    fitness_when_max_exceeded=self.fitness_when_max_exceeded
                ))
                new_population.append(Entity(
                    genotype=child_2,
                    max_admissible_fitness=self.max_admissible_fitness,
                    fitness_when_max_exceeded=self.fitness_when_max_exceeded
                ))            
            self.population = new_population.copy()
    
    def run(self, ar_creator_runner, genotype_param_types, genotype_param_specs, verbose=True, show_progress=False):
        """
        Runs the GA. Note that run() can only be called 1 time. Afterwards, self.simulation_complete will be True, and
        an exception will be thrown.
        

        Parameters:
        ----------
        - ar_creator_runner     : a function that creates an AssemblerReducer object based on a genotype, runs it and returns a 3-tuple containing
                                - a fitness value
                                - a phenotype
                                - a string representation of some aspects of the genotype, phenotype and fitness based on which the solution can
                                  be reconstructed
        - genotype_param_types  : see p_type parameter in __generate_parameter_value
        - genotype_param_specs  : see p_spec parameter in __generate_parameter_value
        - verbose               : if True, prints the fitnesses etc. of all genotypes
        - show_progress         : if True, displays progress bar for each generation
        """

        if not self.simulation_complete:
            for generation in range(self.n_generations):
                if verbose:
                    print(f"************************* Starting generation {generation+1} *************************")
                elif not show_progress:
                    print(f"************************* Starting generation {generation+1} *************************", end='\r')
                else:
                    pass

                ## sort previous population, then sample new genes in population
                self.__step_generation(genotype_param_types, genotype_param_specs)
                
                ## calculate fitnesses using ar_creator_runner...
                for entity_ix in range(len(self.population)):
                    if show_progress:
                        progress_bar(entity_ix + 1, len(self.population), f"Generation {generation+1}", end="\r")
                    if self.population[entity_ix].get_fitness() is None:
                        fitness_phenotype_str = ar_creator_runner(*self.population[entity_ix].get_genotype())
                        self.population[entity_ix].update_thru_aggregator(fitness_phenotype_str)
                    if verbose:
                        print(f"\tResults for entity {entity_ix+1}/{len(self.population)}: {self.population[entity_ix].get_fitness():.8f}")

                fs = [entity.get_fitness() for entity in self.population]
                if verbose:
                    print(f"\n\tMax fitness: {max(fs)}; Std dev of fitnesses: {statistics.stdev(fs)}")

                if max(fs) < self.mutation_incr_fitness_limit or statistics.stdev(fs) < self.mutation_incr_stdev_limit:
                    if self.mutation_prob * 1.1 > self.mutation_max:
                        self.mutation_prob = self.mutation_max
                    else:
                        self.mutation_prob = 1.1 * self.mutation_prob
                    if verbose:
                        print(f"mutation probability updated to: {self.mutation_prob}")

                sorted_pop = sorted(self.population, key=lambda entity: entity.get_fitness(), reverse=True)
                top_top_candidate = sorted_pop[0]
                if self.top_top_entity is None or self.top_top_entity.get_fitness() < top_top_candidate.get_fitness():
                    if self.top_top_entity is not None:
                        old_eval = self.top_top_entity.get_fitness()
                        self.top_top_entity = top_top_candidate
                        print(f"\n\tValidation error for top-top entity so far: {top_top_candidate.get_fitness()}, because old < new: {old_eval < self.top_top_entity.get_fitness()}")
                    else:
                        self.top_top_entity = top_top_candidate
                        print(f"\n\tValidation error for top-top entity so far: {top_top_candidate.get_fitness()}")
                    print(f"\t... with params {top_top_candidate}")

            self.simulation_complete = True
        else:
            raise Exception("Simulation of GA is already complete")
