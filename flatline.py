import scipy.stats as stats


def fit_gamma(permutation_vars):
    a, loc, scale = stats.gamma.fit(permutation_vars, method="MM")
    return a, loc, scale


def p_value_function(a, loc, scale):
    f = lambda x: stats.gamma.sf(x, a, loc=loc, scale=scale)
    return f
