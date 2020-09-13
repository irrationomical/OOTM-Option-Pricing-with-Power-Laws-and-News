# functions for analyzing a time series as a hawkes process
# input is the time series returns

# checks whether
#      1) residuals are iid and follow exp distribution,
#      2) Ljung and Box test follows chi squared distribution with k degrees freedom
#           - check that distributions are similar via Kolgomorov Smirnov test and Excess Dispersion


import numpy as np
import scipy.special as sc
from tick.hawkes import HawkesSumExpKern, HawkesExpKern


def chi_squared_cdf(k, x):
    """ returns cdf(x) of chi squared distribution with k degrees freedom """
    return sc.gammainc(k/2, x/2)


def determine_if_hawkes_process(data, kernel_type, kernel_dim):
    """" determines if data defined by hawkes process
    @param data: (numpy array) n x 3 array w/ columns: time_elapsed, pos, diff (log difference)
    @param kernel_type: (str) kernel type determining intensity decay (exp, double_exp, power_law)
    @param kernel_dim: (int) dimension of hawkes process
    @returns: residuals iid assumption p_value, residuals exp distributed p_value
    """
    residuals = get_hawkes_residuals(data, kernel_type, kernel_dim)
    iid_p_value = determine_if_iid(residuals)
    exp_p_value = determine_if_exp_distributed(residuals)
    return iid_p_value, exp_p_value


def get_hawkes_residuals(data, kernel_type, kernel_dim):
    """" gets residuals for hawkes process fit
    @param data: (numpy array) n x 3 array w/ columns: time_elapsed, pos, diff (log difference)
    @param kernel_type: (str) kernel type determining intensity decay (exp, double_exp, power_law)
    @param kernel_dim: (int) dimension of hawkes process
    @returns residuals: (list of lists) len(residuals) = kernel_dim
    """
    valid_kernels = {'exp':1, 'power law':1}
    if kernel_type not in valid_kernels:
        raise ValueError("provide valid kernel type")

    neg_times = data[np.where(data[:,1]==0),0][0]
    pos_times = data[np.where(data[:,1]==1),0][0]
    timestamps = [data[:,0]] if kernel_dim == 1 else [neg_times, pos_times]

    if kernel_type == 'exp':
        decays = np.ones((kernel_dim, kernel_dim))*3.
        learner = HawkesExpKern(decays)
    else:
        decays = np.ones((1, 15))*3. # sum of 15 exp() variables
        learner = HawkesSumExpKern(decays, penalty='elasticnet', elastic_net_ratio=0.8)

    learner.fit(timestamps)

    # get intensity over time
    intensity_track_step = data[-1,0] / (data.shape[0] * 100)
    tracked_intensity, intensity_times = learner.estimated_intensity(timestamps, intensity_track_step)
    print(tracked_intensity)
    # want to get integral of intensity between each event
    residuals = [] # len of residuals is dimension
    for i in range(kernel_dim):
        time_pairs = [(timestamps[i][n-1], timestamps[i][n]) for n in range(1,len(timestamps[i]))]
        local_residuals = []
        # this loop is slow, should replace it
        for t1, t2 in time_pairs:
            local_intensities_indices = np.where((intensity_times >= t1) & (intensity_times <= t2))
            local_intensities = np.take(tracked_intensity[i], local_intensities_indices)
            local_residuals.append((t2 - t1) * np.mean(local_intensities))
        residuals.append(local_residuals)

    return residuals


# note: generated data is not showing up as iid on this test. not sure why

def determine_if_iid(residuals):
    """ check if iid using Ljung-Box test
        Q follows chi-squared with h degrees of freedom if no autocorrelation
    @param residuals: (list) residuals from hawkes process fit
    @returns q_percentiles: values of Q plugged into chi squared CDF w. h degrees freedom
    """
    q_percentiles = []
    h = 50  # number of lags tested

    # each dimension must be iid
    for i in range(len(residuals)):
        N = len(residuals[i])

        cached_p_hats = dict([(1, 0)])

        def p_hat(k):
            """ gets sample autocorrelation """
            if k in cached_p_hats:
                return cached_p_hats[k]
            else:
                cached_p_hats[k] = np.mean([(residuals[i][j] * residuals[i][j - k]) for j in range(k, N)])
                print("cached", k, cached_p_hats[k])
                return cached_p_hats[k]

        local_q_percentiles = []

        for j in range(2, h + 1):
            Q = N * (N + 2) * np.sum(
                [(p_hat(k) ** 2) / (N - k) for k in range(2, j + 2)])  # i + 2 because range is exclusive
            # check that in range of chi squared cdf
            local_q_percentiles.append(chi_squared_cdf(j, Q))

        q_percentiles.append(local_q_percentiles)

    return q_percentiles


def determine_if_exp_distributed(residuals):
    """ uses Kolmogorov Smirnov test for similarity with exp(1) distribution
    @param residuals: (list of lists) residuals from hawkes process
    @returns p_value: (float) if greater than .05 and less than .95, it's a good fit
    """

    number_of_simulations = 1000
    p_values = []

    for i in range(len(residuals)): # len(residuals) = number of dimension in hawkes process
        actual_ks_value = calculate_exp_ks_statistic(residuals[i])
        sample_ks_values = []

        for j in range(number_of_simulations):
            sample_residuals = np.random.exponential(size=len(residuals[i]))
            sample_ks_values.append(calculate_exp_ks_statistic(sample_residuals))

        # get percentile of actual ks statistic
        sample_ks_values = np.sort(sample_ks_values)
        p_values.append((np.abs(sample_ks_values - actual_ks_value)).argmin() / number_of_simulations)

    return p_values


def calculate_exp_ks_statistic(residuals):
    """ calculates ks statistic for residuals following expo(1) distribution
    @param residuals: (np.array) exponentially distributed residuals
    @returns: ks statistic
    """
    emperical_cdf = np.arange(len(residuals)) / len(residuals)
    true_cdf = 1 - np.exp(-np.sort(residuals))

    return np.max(np.abs(emperical_cdf - true_cdf) / np.sqrt(true_cdf * (1 - true_cdf)))
