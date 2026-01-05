# Functional abstractions for evaluating and aggregating sets of configurations
# Copyright (C) 2025 Corvinus University of Budapest <adambalazs.csapo@uni-corvinus.hu>

from fp_types.progress_bar import progress_bar

class Reducer:
    """
    Wraps a function of form (acc, x) -> acc
    When run over an iterable, along with an initial value, it uses the function to aggregate its values
    """
    def __init__(self, f):
        ## f shall have the form: (acc, x) -> acc
        self.f = f

    def run(self, li, initval = None):
        acc = initval
        for item in li:
            acc = self.f(acc, item)
        return acc

## now, here is an AssemblerReducer, which also assembles the items to reduce on-the-fly using a generator
class AssemblerReducer:
    """
    Takes as input:
    - an assembler that is a generator function of form (params) -> ys, such that in each step the function generates a single y
    - a reducer function of form (acc, y) -> acc

    When run over a set of parameters params, it runs the assembler, and aggregates each of the generated ys.

    For instance, within a genetic algorithm, the assembler can be a function that takes a genotype, generates a phenotype based on the genotype,
    then carries out T trials to evaluate its performance (in case some randomness or trial-and-error aspect is involved using a set of hyperparameters).
    In this case, the ys that are returned might be a loss value, plus some additional information (e.g. on the phenotype or other hyperparameters).
    The reducer function could then aggregate these to create a single loss value.
    """
    def __init__(self, ass, red):
        ## ass should be a generator of form: x -> y
        ## red shall have the form: (acc, y) -> acc
        self.assembler = ass
        self.reducer = red

    def run(self, params, initval = None, show_progress=False, progress_msg=None, num_steps=None):
        acc = initval
        for yinx, y in enumerate(self.assembler(*params)):
            acc = self.reducer(acc, y)
            if show_progress:
                progress_bar(yinx+1, num_steps, progress_msg)
        return acc

