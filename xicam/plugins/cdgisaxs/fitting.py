import numpy as np

def corrections_DWI0Bk(Is, DW_factor, I0, Bk, qxs, qzs):
    I_corr = []
    for I, qx, qz in zip(Is, qxs, qzs):
        DW_array = np.exp(-(np.asarray(qx) ** 2 + np.asarray(qz) ** 2) * DW_factor ** 2)
        I_corr.append(np.asarray(I) * DW_array * I0 + Bk)
    return I_corr


def log_error(exp_I_array, sim_I_array):
    error = np.nansum(np.abs(np.log10(exp_I_array) - np.log10(sim_I_array))) / np.count_nonzero(~np.isnan(exp_I_array))
    return error


def abs_error(exp_I_array, sim_I_array):
    error = np.nansum(np.abs(exp_I_array - sim_I_array) / np.nanmax(exp_I_array)) / np.count_nonzero(~np.isnan(exp_I_array))
    return error


def squared_error(exp_I_array, sim_I_array):
    error = np.nansum((exp_I_array - sim_I_array) ** 2 / np.nanmax(exp_I_array) ** 2) / np.count_nonzero(~np.isnan(exp_I_array))
    return error
