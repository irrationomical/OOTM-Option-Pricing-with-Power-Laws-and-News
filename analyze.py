# functions for analyzing a time series as a hawkes process
# input is the time series returns

# checks whether
#      1) residuals are iid and follow exp distribution,
#      2) Ljung and Box test follows chi squared distribution with k degrees freedom
#           - check that distributions are similar via Kolgomorov Smirnov test and Excess Dispersion


import pandas as pd
import numpy as np
import scipy.special as sc
from tick.hawkes import SimuHawkesSumExpKernels, SimuHawkesMulti, \
    HawkesSumExpKern, SimuHawkesExpKernels, HawkesExpKern


def chi_squared_cdf(k, x):
    return sc.gammainc(k/2, x/2)

def determine_if_hawkes_process(data, kernel_type, kernel_dim):
    """" determines if data defined by hawkes process
    @param data (numpy array): n x 3 array w/ columns: time_elapsed, pos, diff (log difference)
    @param kernel_type (str): kernel type determining intensity decay (exp, double_exp, power_law)
    @param kernel_dim (int): dimension of hawkes process
    @returns: graph of arrival process distribution, statistical significance of fit
    """
    valid_kernels = {'exp':1, 'power law':1}
    if kernel_type not in valid_kernels:
        raise ValueError("provide valid kernel type")

    # neg_times = data[np.where(data[:,1]==0),0][0]
    # pos_times = data[np.where(data[:,1]==1),0][0]
    # timestamps = list((data[0])) if kernel_dim == 1 else list((neg_times, pos_times))
    timestamps = [data[:,0]]

    kernel = HawkesExpKern if kernel_type == 'exp' else HawkesSumExpKern
    decays = np.ones((kernel_dim, kernel_dim))*.01

    learner = kernel(decays, penalty='elasticnet', elastic_net_ratio=0.8)
    learner.fit(timestamps)

    # get intensity over time
    intensity_timestep = data[-1,0] / (data.shape[0] * 10)
    tracked_intensity, intensity_times = learner.estimated_intensity(timestamps, intensity_timestep)

    # check that the intensity residuals are iid and follow exp distribution
    # for each event, get nearest intensity
    def get_index_of_nearest_time(time, times):
        """ returns index of element in times closest to time """
        return (np.abs(np.array(times) - time)).argmin()

    # intensity at event times
    nearest_event_time_indices = list(map(lambda t: get_index_of_nearest_time(t, intensity_times), timestamps[0]))
    event_intensity = np.take(tracked_intensity[0], nearest_event_time_indices)
    event_intensity_times = np.take(intensity_times, nearest_event_time_indices)

    #
    intensity_differences = ((np.roll(event_intensity, 1) + event_intensity)/2)[1:]
    time_differences = (np.roll(event_intensity_times,1) - event_intensity_times)[1:]

    # integral getting residuals
    residuals = intensity_differences * time_differences

    # check that iid using Ljung-Box test to test absense of autocorrelation
    # Q should follow chi-squared with h degrees of freedom
    Q_percentiles = []
    N = data.shape[0]
    h = 10 # number of lags tested

    cached_p_hats = dict([(1, 0)])
    def p_hat(k):
        """ gets sample autocorrelation """
        if k in cached_p_hats:
            return cached_p_hats[k]
        else:
            cached_p_hats[k] = np.mean((residuals * np.roll(residuals,k))[k:]) + p_hat(k-1)
            return cached_p_hats[k]

    for k in range(2, h+1):
        Q = N * (N+2) * p_hat(k)**2 / (N-k)
        # check that in range of chi squared cdf
        Q_percentiles.append(chi_squared_cdf(k, Q))

    # adjusted Kolmogorov Sminov test with residuals and exp(1)
    # calculate emperical cdf for subset
    sorted_residuals = np.sort(residuals) # increasing order
    emperical_cdf = np.arange(len(residuals)) / len(residuals)
    # get real cdf from exp() distribution for lambda = 1
    true_cdf = np.exp(-sorted_residuals)
    KS = np.max(np.abs(emperical_cdf - true_cdf) / np.sqrt(true_cdf * (1 - true_cdf)))

    # check that the KS stat follows the KS distribution

    return Q_percentiles, KS
