# this basically just copies some functions/classes from gatai to expose them to the user
from __future__ import annotations
import scipy
import numpy as np
import pandas as pd
from typing import Literal
from functools import partial
from setga import utils, select_subset


class Expression_data:
    """class to store the expression dataset with some precomputations"""

    @staticmethod
    def quantilerank(xs):
        """computes the quantile rank for the phylostrata

        :param xs: numpy array of values
        :type xs: np.array
        :return: quantile ranked values
        :rtype: np.array
        """
        ranks = scipy.stats.rankdata(xs, method="average")
        quantile_ranks = [
            scipy.stats.percentileofscore(ranks, rank, kind="weak") for rank in ranks
        ]
        return np.array(quantile_ranks) / 100

    @classmethod
    def from_csv(cls, path: str, delimiter="\t", **kwargs) -> Expression_data:
        arr = pd.read_csv(path, delimiter=delimiter)
        return Expression_data(arr, **kwargs)

    @property
    def p_matrix(self):
        normalised_expr = self.expressions / self.expressions_n.sum(axis=0)
        weighted_expr = normalised_expr.mul(self.full["Phylostratum"], axis=0)

        return weighted_expr

    @property
    def centered_p_matrix(self):
        p = self.p_matrix.to_numpy()
        return p - np.mean(p, axis=1)[:, None]

    @property
    def tai(self):
        avgs = self.p_matrix.sum(axis=0)

        return avgs

    def __init__(
        self,
        expression_data: pd.DataFrame,
        transformation: Literal["none", "sqrt", "log"] = "none",
    ):
        """
        :param expression_data: expression dataset
        :type expression_data: pd.DataFrame
        """
        self.raw = expression_data

        self.transformation = transformation
        # NOTE: apparently pandas iloc sometimes returns a copy instead of a view if the dtypes of columns are different
        # In our case, applying the transformation to exps wasn't also applying it to full so I changed the order trasnformations are applied
        exps = self.raw.iloc[:, 2:]
        match transformation:
            case "sqrt":
                exps = exps.map(lambda x: np.sqrt(x))
            case "log":
                exps = exps.map(lambda x: np.log(x + 1))

        self.full = pd.concat([self.raw[["Phylostratum", "GeneID"]], exps], axis=1)
        self.full["Phylostratum"] = Expression_data.quantilerank(
            self.full["Phylostratum"]
        )

        age_weighted = exps.mul(self.full["Phylostratum"], axis=0).to_numpy()
        self.age_weighted = age_weighted
        self.expressions_n = exps.to_numpy()
        self.expressions = exps
        self.weighted_sum = np.sum(
            exps.mul(self.full["Phylostratum"], axis=0).to_numpy(), axis=0
        )
        self.exp_sum = np.sum(exps.to_numpy(), axis=0)
        self.expressions_n_sc = exps.to_numpy()

    def remove_genes(self, gene_ids) -> Expression_data:
        expr_data = self.raw[~self.raw["GeneID"].isin(gene_ids)]
        return Expression_data(expr_data, transformation=self.transformation)

    @property
    def gene_variances(self) -> pd.DataFrame:
        df = self.full.copy()
        age_weighted = self.expressions.mul(self.full["Phylostratum"], axis=0)
        df["Variance"] = age_weighted.values.tolist()
        df["Variance"] = df["Variance"].apply(np.var)
        return df[["Phylostratum", "GeneID", "Variance"]]


def get_extracted_genes(
    data: Expression_data,
    permuts,
    /,
    population_size=150,
    num_generations=15000,
    num_islands=4,
    mut=0.005,
    cross=0.02,
    stop_after=200,
):

    def get_distance(solution):
        """computes variance of the TAI for the particular solution

        :param solution: binary encoded, which genes belong in the solution
        :type solution: array
        :return: variance
        :rtype: float
        """

        sol = np.array(solution)
        sol = np.logical_not(sol).astype(int)
        up = sol.dot(data.age_weighted)
        down = sol.dot(data.expressions_n)
        avgs = np.divide(up, down)
        return np.var(avgs)

    def get_skewed_reference(num_points, skew):
        y_values = np.linspace(skew, 1, num_points + 1)
        # Calculate corresponding y values such that the sum of x and y is 1
        x_values = 1 - y_values

        # Create the numpy array with two columns
        return np.column_stack((x_values, y_values))[:-1]

    def get_uniform_reference(num_points):
        y_values = np.linspace(0, 1, num_points)
        # Calculate corresponding y values such that the sum of x and y is 1
        x_values = 1 - y_values

        # Create the numpy array with two columns
        return np.column_stack((x_values, y_values))

    ref_points = get_uniform_reference(10)
    ref_points = np.append(ref_points, get_skewed_reference(4, 0.75)[:-1], axis=0)

    ind_length = data.full.shape[0]
    max_value = get_distance(np.zeros(ind_length))

    def evaluate_individual(individual, permuts, expression_data):
        """computes the overall fitness of an individual

        :param individual: binary encoded, which genes belong in the solution
        :type individual: array
        :param permuts: precomputed variances from flat-line test
        :type permuts: np.array
        :param expression_data: dataset of expression of the genes
        :type expression_data: pd.DataFrame
        """

        def get_fit(res):
            """computes empirical p-value of an individual

            :param res: variance of an individual
            :type res: np.array
            :return: empirical p-value
            :rtype: float
            """
            p = np.count_nonzero(permuts < res) / len(permuts)
            r = (res) / (max_value)
            r = r + p
            return r if p > 0.2 else 0

        sol = np.array(individual)
        sol = np.logical_not(sol).astype(int)
        distance = np.var(
            np.divide(
                sol.dot(expression_data.age_weighted),
                sol.dot(expression_data.expressions_n),
            )
        )
        fit = get_fit(distance)
        # Return the fitness values as a tuple
        return fit

    def end_evaluate_individual(individual):
        """individual fitness without the cutoff, just pure p-value

        :param individual: binary encoded, which genes belong in the solution
        :type individual: array
        :return: fitness
        :rtype: float
        """
        individual = np.array(individual)
        distance = get_distance(individual)
        fit = np.count_nonzero(permuts < distance) / len(permuts)
        # Return the fitness values as a tuple
        return np.sum(individual), fit

    eval_part = partial(evaluate_individual, permuts=permuts, expression_data=data)

    pop, _, gens, logbook, best_sols = select_subset.run_minimizer(
        data.full.shape[0],
        eval_part,
        1,
        "Variance",
        mutation_rate=mut,
        crossover_rate=cross,
        pop_size=population_size,
        num_gen=num_generations,
        num_islands=num_islands,
        mutation=["weighted", "weighted", "bit-flip", "bit-flip"],
        crossover="uniform",
        selection="NSGA3",
        frac_init_not_removed=0.2,
        ref_points=ref_points,
        stop_after=stop_after,
        weights=np.sqrt(np.var(data.expressions_n, axis=1)),
    )

    ress = np.array([end_evaluate_individual(x) for x in pop])
    min_ex = min(pop, key=lambda ind: ind.fitness.values[1]).fitness.values[0]
    pop = np.array(pop)
    genes = utils.get_results(pop, ress, data.full.GeneID)

    return genes
