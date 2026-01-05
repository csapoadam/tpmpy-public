# Class for managing TS fuzzy rule sets derived from TP systems
# Copyright (C) 2025 Corvinus University of Budapest <adambalazs.csapo@uni-corvinus.hu>

import csv

# Copyright (C) 2025 Adam Csapo <adambalazs.csapo@uni-corvinus.hu>

__all__ = [
    'RuleSet'
]


class RuleSet:
    """
    A RuleSet is a set of fuzzy rules that can be derived from e.g. a TP model (ideally w/ CNO type weighting functions).
    Each rule in the RuleSet has a weight associated with it that is in direct proportion to the number of data points that represent it.
    Further, each antecedent variable in each rule is represented by an ordinal number 0, 1, ...; such that
    each of those ordinal numbers have a range of values associated with them within the corresponding input dimension.
    """
    def __init__(self, rules, antecedent_ranges, weights, counts):
        """
        rules - a list of R tuples (ant, cons) such that ant is a list of indices in each dimension, and cons is an output value
        antecedent_ranges - a list of D lists (one sub-list per dimension), such that each sub-list contains tuples of (min,max) values defining a range of values along the corresponding dimension
        weights - a list R weights, such that each item is a weight associated with the corresponding rule in rules
        counts - numpy array with D dimensions, with a count per grid coordinate (see 2nd return value of map_points_to_grid() in grid.py)

        """
        self.rules = rules
        self.antecedent_ranges = antecedent_ranges
        self.weights = weights
        self.counts = counts

    def __str__(self):
        outs = []
        for inx,(inputs, output) in enumerate(self.rules):
            current_out = " AND ".join(
                list(map(lambda x: str(x), inputs))
            )
            current_out = current_out + " => " + f"{output:.2f}"
            outs.append(f"{current_out} (weight: {self.weights[inx]:.4f})")
        outs.append(f"\nTotal number of rules: {len(self.rules)}")
        outs.append(f"\nSum of all weights: {sum(self.weights)}")
        return "\n".join(outs)

    def to_csv(self, min_weight, filename):
        """
        Convert fuzzy rules into CSV rows with numeric values.

        Args:
            min_weight (float): Minimum weight threshold.
            filename (str): Name of file to write results
        """
        rows = []
        for inx, (inputs, output) in enumerate(self.rules):
            weight = self.weights[inx]

            if weight >= min_weight:
                nums = [str(i) for i in inputs] + [f"{output:.2f}"]
                rows.append(",".join(nums))

        csv_str = "\n".join(rows)
        # newline='' avoids extra blank lines on Windows
        with open(filename, "w", newline="") as f:
            f.write(csv_str)


