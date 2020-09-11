
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_alpha_cont(X):
    """ gets alpha paramter for continuous power law distribution
    @param X (np.array): just holds diff
    """
    n = len(X)
    xmin = X[0]
    a = 1 + n / np.sum(np.log(X / xmin))
    return a


def get_a_range(a, N):
    """ gets 95% confidence interval on alpha parameter estimate
    @param a: alpha
    @param N: number of samples in power law distribution
    @returns a_lower, a_upper: lower and upper bound on alpha 95% CI
    """
    stdev = (a - 1) / np.sqrt(N)
    a_lower, a_upper = ((a - stdev * 2), (a + stdev * 2))
    return a_lower, a_upper


# CODE TO CHECK WHETHER POWER LAW

# for a stock, calculate xmin, alpha, KS stat and determine whether power law
def calc_emperical(df_sorted):
    """ determines whether a power law fits the dataset
    @param df_sorted: price change dataframe sorted from most frequent to least frequent
    @returns p: p-value of fit for power law. p < .1 indicates not a power law.
    p-value above .1 indicates either power law (could exponential or lognormal if N<200)
    """
    # get quantile, alpha, and KS for best fit
    q, a, KS, xmin = get_xmin(df_sorted)
    print("True q: %s a: %s KS: %s xmin: %s" % (q, a, KS, xmin))
    p = calc_fit(df_sorted, q, a, KS, xmin)
    print("p: ", p)
    return p


def calc_fit(df_sorted, q, a, KS, xmin):
    """ determines whether dataset fits power law
    @param df_sorted: price change dataframe sorted from most frequent to least frequent
    @param q: quantile for karmatta point
    @param a: alpha of dataset
    @param KS: Kolgoromov Smirnoff statistic for fit
    @param xmin: minimum x-value for power law data determined by get_xmin()
    @returns p: p-value of fit explained in calc_empirical doc
    """
    KS_stats = []
    for i in range(2500):
        df = generate_dataset(df_sorted, q, a, xmin)
        q_local, a_local, KS_local, xminlocal = get_xmin(df)
        KS_stats.append(KS_local)
    KS_stats = np.array(KS_stats)
    p = np.mean(KS_stats > KS)
    return p


def generate_dataset(df_sorted, q, a, xmin):
    """ generates synthetic dataset with q,a,xmin
    @returns df: synthetic dataset frame
    """
    N = df_sorted.shape[0]
    df_below_xmin = df_sorted.loc[df_sorted['prob'] >= q]
    # number of samples to generate from power law
    n_power = np.sum(np.random.uniform(0, 1, N) < q)
    # number of samples to bootstrap from original dataset > q
    n_normal = N - n_power
    # generate power law variables
    r = np.random.uniform(0, 1, n_power)
    x1 = xmin * (1 - r) ** (-1 / (a - 1))
    # bootstrap n_normal variables from df_below_xmin
    x2 = np.random.choice(df_below_xmin['diff'], n_normal, replace=True)
    # make new dataframe
    df = df_sorted.drop(['diff'], axis=1)
    df['diff'] = list(np.sort(np.concatenate((x1, x2), axis=0)))
    return df


def get_xmin(df_sorted):
    """ finds xmin value, quantile, alpha, and KS-stat for emperical distribution.
    @param df_sorted: df sorted by quantile, either negative or positive changes
    @return: quantile, alpha param, KS-stat, xmin
    """
    quants = [.5, .35, .2, .14, .10, .07, .035]  # 100% of data - min quantile
    D, alphas1 = calc_KS(df_sorted, quants)
    sorted = list(np.argsort(D))
    num_searches = 6  # number of searches over smaller range
    # get quantiles of refined search range
    quants = list(np.linspace(quants[sorted[0]] - .025, quants[sorted[0]] + .025, num=num_searches))
    D2, alphas2 = calc_KS(df_sorted, quants)
    index = np.argmin(D2)
    # get quantile of best KS stat
    quant = np.round(quants[index], 3)
    # get alpha of that quantile
    alpha = alphas2[index]
    # get KS of that quantile
    KS = D2[index]
    # get minimum change in power law
    xmin = df_sorted['diff'].loc[df_sorted['prob'] < quant].iloc[0]
    return quant, alpha, KS, xmin


def calc_KS(df_sorted, quantiles):
    """ Calculate Kolmogorov-Smirnov Statistic for goodness of fit test using
    CDF for continuous power law
    @param df_sorted: sorted dataframe
    @param quantiles: quantiles to test in df_sorted
    @return D: Distances measured by KS-stat
    @return alphas: alpha parameters corresponding to D[i] quantiles
    """
    D = []
    alphas = []
    for i in range(len(quantiles)):
        # get subset for min quantile
        df = df_sorted.loc[df_sorted['prob'] < quantiles[i]]
        X = df['diff'].to_numpy()
        a = get_alpha_cont(X)  # get alpha for that subset
        alphas.append(a)
        # calculate emperical cdf for subset
        df['emp_cdf'] = 1 - np.arange(df.shape[0]) / df.shape[0]
        # get real cdf for continuous power law
        xmin = df['diff'].iloc[0]
        df['true_cdf'] = (df['diff'] / xmin) ** (-a + 1)
        # get KS Statistic on subset
        D.append(np.max(np.abs(df['emp_cdf'] - df['true_cdf']) / np.sqrt(df['true_cdf'] * (1 - df['true_cdf']))))
    return D, alphas


# CODE SHOWS POWER LAW
# function to print power laws for stock
def show_power_law(neg_sorted=None, pos_sorted=None, ax=None):
    """ prints graph of power law for price daily price changes in stock
    @param ax: matplotlib axes for the graph
    """
    if (neg_sorted is None) and (pos_sorted is None):
        raise SyntaxError("Must provide some dataframe")
    if ax is None:
        ax = plt.gca()

    if (neg_sorted is not None):
        # sort by move size and calculate cdf

        q, alpha_hat, KS, xmin = get_xmin(neg_sorted)
        print("q =",q.round(2), "a =",alpha_hat.round(2), "KS =",KS.round(2), "xmin =",xmin.round(6))
        select_df = neg_sorted.loc[neg_sorted['prob'] < q]
        select_df = select_df.reset_index(drop=True)
        size = select_df.shape[0]

        # neg sorted slope
        x = np.linspace(np.min(select_df['diff']), np.max(select_df['diff']), num=50)
        k = select_df.loc[0, 'prob'] * (x[0] ** (alpha_hat))
        y = k * (x ** (-alpha_hat))
        # plot
        ax.plot(neg_sorted['diff'], neg_sorted['prob'], '.', c='red', markeredgecolor='none')
        ax.plot(x, y)

        sigma_a = get_a_range(alpha_hat, size)
        sigma_a = list(map(lambda x: x.round(2), sigma_a))
        print("neg alpha range:", sigma_a)
        textstr = r'$\alpha_-$ = %s %sN=%s' % (np.round(alpha_hat, 2), '\n', size)
        ax.annotate(textstr, xy=(x[0], y[0]), xytext=(x[0] + .02, y[0] + .3),
                    arrowprops=dict(facecolor='black', shrink=0.1))

    if (pos_sorted is not None):
        q, alpha_hat2, KS, xmin = get_xmin(pos_sorted)
        print("q =",q.round(2), "a =",alpha_hat2.round(2), "KS =",KS.round(2), "xmin =",xmin.round(6))
        select_df = pos_sorted.loc[pos_sorted['prob'] < q]
        select_df = select_df.reset_index(drop=True)
        size2 = select_df.shape[0]

        x_pos = np.linspace(np.min(select_df['diff']), np.max(select_df['diff']), num=50)
        k = select_df['prob'][0] * (x_pos[0] ** (alpha_hat2))
        y_pos = k * (x_pos ** (-alpha_hat2))

        ax.plot(pos_sorted['diff'], pos_sorted['prob'], '.', c='blue', markeredgecolor='none')
        ax.plot(x_pos, y_pos)

        sigma_a = get_a_range(alpha_hat2, size2)
        sigma_a = list(map(lambda x: x.round(2), sigma_a))
        print("pos alpha range:", sigma_a)
        textstr2 = r'$\alpha_+$ = %s %sN=%s' % (np.round(alpha_hat2, 2), '\n', size2)
        ax.annotate(textstr2, xy=(x_pos[0], y_pos[0]), xytext=(x_pos[0], y_pos[0] + .3),
                    arrowprops=dict(facecolor='black', shrink=0.1))

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim(left=10 ** -3)
    # ax.set_title(ticker.upper())

    return ax
