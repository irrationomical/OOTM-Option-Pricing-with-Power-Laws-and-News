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
    residuals = get_hawkes_residuals(data, kernel_type, kernel_dim)
    iid_p_value = determine_if_iid(residuals)
    exp_p_value = determine_if_exp_distributed(residuals)
    return iid_p_value, exp_p_value


# need to make this work for two dimensions and power law kernel
def get_hawkes_residuals(data, kernel_type, kernel_dim):
    """" determines if data defined by hawkes process
    @param data: (numpy array) n x 3 array w/ columns: time_elapsed, pos, diff (log difference)
    @param kernel_type: (str) kernel type determining intensity decay (exp, double_exp, power_law)
    @param kernel_dim: (int) dimension of hawkes process
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
    decays = np.ones((kernel_dim, kernel_dim))*3.

    # learner = kernel(decays, penalty='elasticnet', elastic_net_ratio=0.8)
    learner = kernel(decays)
    learner.fit(timestamps)

    # get intensity over time
    intensity_track_step = data[-1,0] / (data.shape[0] * 1000)
    tracked_intensity, intensity_times = learner.estimated_intensity(timestamps, intensity_track_step)

    # want to get integral of intensity between each event
    time_pairs = [(timestamps[0][n-1], timestamps[0][n]) for n in range(1,len(timestamps[0]))]

    residuals = []
    # this loop is slow - should replace it
    for t1,t2 in time_pairs:
        local_intensities_indices = np.where((intensity_times >= t1) & (intensity_times <= t2))
        local_intensities = np.take(tracked_intensity, local_intensities_indices)
        residuals.append((t2-t1) * np.mean(local_intensities))

    return residuals

    # print("tracked intensity", tracked_intensity)
    # print("intensity times", intensity_times)
    # print("real times", timestamps[0])
    #
    # # check that the intensity residuals are iid and follow exp distribution
    #
    # def get_index_of_nearest_time(time, times):
    #     """ returns index of element in times closest to time """
    #     return (np.abs(times - time)).argmin()
    #
    # # get intensity at each event time
    # nearest_event_time_indices = list(map(lambda t: get_index_of_nearest_time(t, intensity_times), timestamps[0]))
    # print("nearest indices", nearest_event_time_indices)
    # event_intensity = np.take(tracked_intensity[0], nearest_event_time_indices)
    # event_intensity_times = np.take(intensity_times, nearest_event_time_indices)
    #
    # print("event intensity", event_intensity)
    # print("event intensity times", event_intensity_times)

    # intensity_differences = ((np.roll(event_intensity, 1) + event_intensity)/2)[1:]
    # time_differences = (np.roll(event_intensity_times,1) - event_intensity_times)[1:]

    # integral getting residuals
    # residuals = intensity_differences * time_differences


# note: generated data is not showing up as iid on this test. not sure why

def determine_if_iid(residuals):
    """ check if iid using Ljung-Box test
        Q follows chi-squared with h degrees of freedom if no autocorrelation
    @param residuals: (list) residuals from hawkes process fit
    @returns Q_percentiles: values of Q plugged into chi squared CDF w. h degrees freedom
    """
    Q_percentiles = []
    N = len(residuals)
    h = 50  # number of lags tested

    cached_p_hats = dict([(1, 0)])

    def p_hat(k):
        """ gets sample autocorrelation """
        if k in cached_p_hats:
            return cached_p_hats[k]
        else:
            cached_p_hats[k] = np.mean([(residuals[i] * residuals[i-k]) for i in range(k, N)])
            print("cached",k,cached_p_hats[k])
            return cached_p_hats[k]

    for i in range(2, h + 1):
        Q = N * (N + 2) * np.sum( [(p_hat(k) ** 2)/(N - k) for k in range(2,i+2)] ) # i + 2 because range is exclusive
        # check that in range of chi squared cdf
        Q_percentiles.append(chi_squared_cdf(i, Q))

    return Q_percentiles


def determine_if_exp_distributed(residuals):
    """ uses Kolmogorov Smirnov test for similarity with exp(1) distribution
    @param residuals: (list) residuals from hawkes process
    @returns p_value: (float) if greater than .05 and less than .95, it's a good fit
    """
    number_of_simulations = 1000
    actual_ks_value = calculate_exp_ks_statistic(residuals)
    sample_ks_values = []
    for i in range(number_of_simulations):
        sample_residuals = np.random.exponential(size = len(residuals))
        sample_ks_values.append(calculate_exp_ks_statistic(sample_residuals))

    # get percentile of actual ks statistic
    sample_ks_values = np.sort(sample_ks_values)
    p_value = (np.abs(sample_ks_values - actual_ks_value)).argmin() / number_of_simulations

    return p_value


def calculate_exp_ks_statistic(residuals):
    """ calculates ks statistic for residuals following expo(1) distribution
    @param residuals: (np.array) exponentially distributed residuals
    @returns: ks statistic
    """
    emperical_cdf = np.arange(len(residuals)) / len(residuals)
    true_cdf = 1 - np.exp(-np.sort(residuals))
    return np.max(np.abs(emperical_cdf - true_cdf) / np.sqrt(true_cdf * (1 - true_cdf)))
