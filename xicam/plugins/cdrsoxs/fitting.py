import numpy as np
from pyswarm import pso
import __init__
import simulation

from pyevolve import G1DList
from pyevolve import GSimpleGA
from pyevolve import Selectors
from pyevolve import Statistics
from pyevolve import DBAdapters
import pyevolve

from random import randrange
import cPickle as pickle
import psutil
import multiprocessing
from collections import deque
from itertools import repeat
import panda as pd
import emcee
import deap.base as deap_base
from deap import creator, tools
from deap import cma as cmaes
'''


def pso(initiale_value, lower_bnds, upper_bnds):


    lower_bnds, upper_bnds = [], []
    for i in initiale_value:
        lower_bnds.append(int(i - 10))
        upper_bnds.append(int(i + 10))
    xopt, fopt = pso(self.residual, lower_bnds, upper_bnds)
    print(xopt, fopt)
    self.residual(xopt)
    print(opt.message)

def py_evol(num_param, num_generation, qxs, qzs):
    best_score = 0
    genome = G1DList.G1DList(num_param)
    genome.setParams(rangemin=0, rangemax=1000)
    genome.evaluator.set(self.residual)

    ga = GSimpleGA.GSimpleGA(genome)
    ga.selector.set(Selectors.GRouletteWheel)
    ga.setGenerations(num_generation)

    #ga.stepCallback.set(evolve_callback)
    ga.evolve()

    #print ga.bestIndividual()
    best = ga.bestIndividual()
    #print(best.genomeList, best.score)

    return best


def evolve_callback(ga_engine):
    generation = ga_engine.getCurrentGeneration()
    if generation % 10 == 0:
        print "Current generation: %d" % (generation,)
        best = ga_engine.bestIndividual()
        if best.score > best_score:
            best_score = best.score
            self.residual(best.genomeList, 'True')

            self.modelParameter = 5 + 0.02 * best.genomeList[2], 20 + 0.04 * best.genomeList[3], 70 + 0.04 * best.genomeList[4], best.score
            H, LL, Beta = 5 + 0.02 * best.genomeList[2], 20 + 0.04 * best.genomeList[3], np.asarray([70 + 0.04 * best.genomeList[4], 70 + 0.04 * best.genomeList[5], 70 + 0.04 * best.genomeList[6], 70 + 0.04 * best.genomeList[7], 70 + 0.04 * best.genomeList[8]])
            Obj = simulation.multipyramid(H, LL, Beta, 500, 500)
            Obj_plot = np.rot90(Obj, 3)


def residual(p, test = 'False', plot_mode=False):
    DW = 0.0001 * p[0]
    I0 = 0.01 * p[1]
    Bk = 0.01 * p[2]
    H = 5 + 0.02 * p[3]
    LL = 20 + 0.04 * p[4]
    Beta = []

    for i in range(4, len(p), 1):
        Beta.append(50 + 0.08 * p[i])

    Beta = np.array(Beta)

    Qxfit = __init__.SL_model1(H, LL, Beta, DW_factor=DW, I0=I0, Bk=Bk)

    Qxfit = corrections_DWI0Bk(Qxfit, DW_factor=DW, I0=I0, Bk=Bk, qxs, qzs)

    #self.Qxfit = correc_Isim(DW, I_scale, 1)

    res = 0
    res_min = 1000

    for i in range(0, len(self.Qxexp), 1):
        res += np.sqrt(sum((self.Qxfit[i] - self.Qxexp[i])**2) / sum((self.Qxexp[i])**2))

    maxres = min(maxres, res)
    return res
'''


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

'''
def fittingp_to_simp(self, fittingp):
    # DW, I0, Bk, H, LL, *Beta[5] = simp
    # values assume initial fittingp centered at 0 and std. dev. of 100
    multiples = np.array([0.0001, 0.01, 0.01, 0.02, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04])
    simp = multiples * np.asarray(fittingp) + self.adds
    if np.any(simp[:5] < 0):
        return None
    if np.any(simp[5:] < 0) or np.any(simp[5:] > 180):
        return None
    return simp

@staticmethod
def fix_fitness_cmaes(fitness):
    """cmaes accepts the individuals with the lowest fitness, doesn't matter degree to which they are lower"""
    return fitness,

@staticmethod
def fix_fitness_mcmc(fitness):
    """
    Metropolis-Hastings criterion: acceptance probability equal to ratio between P(new)/P(old)
    where P is proportional to probability distribution we want to find
    for our case we assume that probability of our parameters being the best is proportional to a Gaussian centered at fitness=0
    where fitness can be log, abs, squared error, etc.
    emcee expects the fitness function to return ln(P(new)), P(old) is auto-calculated
    """
    c = 1e-1  # empirical factor to modify mcmc acceptance rate, makes printed fitness different than actual, higher c increases acceptance rate
    return -fitness / c
    # return -0.5 * fitness ** 2 / c ** 2

'''
