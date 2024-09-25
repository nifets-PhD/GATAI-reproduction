# this basically just copies some functions/classes from gatai to expose them to the user
from __future__ import annotations
import scipy
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from typing import Literal


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
    def tai(self):
        weighted_expr = self.expressions.mul(self.full["Phylostratum"], axis=0)
        avgs = weighted_expr.sum(axis=0) / self.expressions_n_sc.sum(axis=0)

        return avgs

    def __init__(
        self,
        expression_data: pd.DataFrame,
        single_cell: bool = False,
        transformation: Literal["none", "sqrt", "log"] = "none",
    ):
        """
        :param expression_data: expression dataset
        :type expression_data: pd.DataFrame
        """
        expression_data["Phylostratum"] = Expression_data.quantilerank(
            expression_data["Phylostratum"]
        )
        self.full = expression_data
        exps = expression_data.iloc[:, 2:]
        match transformation:
            case "sqrt":
                exps = exps.map(lambda x: np.sqrt(x))
            case "log":
                exps = exps.map(lambda x: np.log(x + 1))

        age_weighted = exps.mul(expression_data["Phylostratum"], axis=0).to_numpy()
        self.age_weighted = age_weighted
        self.expressions_n = exps.to_numpy()
        self.expressions = exps
        self.weighted_sum = np.sum(
            exps.mul(expression_data["Phylostratum"], axis=0).to_numpy(), axis=0
        )
        self.exp_sum = np.sum(exps.to_numpy(), axis=0)
        self.expressions_n_sc = exps.to_numpy()
        if single_cell:
            self.expressions_n = csr_matrix(
                exps.to_numpy()
            )  # Define your sparse matrix 'a'
            self.age_weighted = csr_matrix(
                age_weighted
            )  # Define your sparse matrix 'a_w'
