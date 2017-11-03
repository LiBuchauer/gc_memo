"""
gc_maps.py contains functions useful for running larger parameter scans and
also for analysing their results. Results of these scans are saved to
'/map_data', as their format is more compressed than the complete simulation
data stored in '/raw_data'.
"""

from __future__ import division
from itertools import product

import numpy as np
from os import listdir
import pandas as pd
import pylab

import seaborn
import sys
import time
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.pyplot import cm
import matplotlib as mpl

import cf
import gc_memo

reload(cf)
reload(gc_memo)

from gc_memo import main

# general plot settings
cc = 30  # number of clours for cycling through
cmap = seaborn.color_palette('Spectral', cc)
colors = iter(cm.rainbow(np.linspace(0, 1, cc)))
seaborn.set_style('ticks')
seaborn.set_context('talk')

plt.rc('text', usetex=True)
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica'],
              'style': 'normal'})
rc("text.latex", preamble=["\\usepackage{helvet}\\usepackage{sfmath}"])
pylab.ion()


def map_params(dose=[1], LFdecay=[10*12], nGCs=[1], nLFs=[25],
               naive_pool=[1000], nkey=[1, 2, 10], p_err=[0.003],
               tinf=[[0*12]],
               p_block=[0.5], repeats=1):

    """ Function for mapping out the effects of different parameter
    (combinations) on the standard TUCHMI protocol. All arguments are lists,
    either containing only the default value or a set of values, in which case
    all combinations of list arguments will be executed the given number
    of times (repeat).

    The paramter set used is stored together with the complete timecourse of
    of E_bind, SHM, entropy, memory number (means and std where applicable)
    and exported into a .h5 file.

    For running on a cluster, accepts an ID argument (e.g. job ID) for easier
    handling of errors etc.
    """

    # open file stamped with systemtime if no other ID was provided in the call
    try:
        sys.argv[1]
    except IndexError:
        filepath = 'map_data/data{}.h5'.format(int(time.time()*100))
    else:
        filepath = 'map_data/data{}.h5'.format(sys.argv[1])
    print(filepath)
    datafile = pd.HDFStore(filepath)
    # dict for collecting result series
    seriesdict = {}

    # set endtime and days for evaluation
    cf.endtime = 126*12  # run until challenge timepoint
    # evaluate pool every day
    evaldays = np.arange(126)

    # get parameter combinations
    paramsets = list(product(dose, LFdecay, nGCs, nLFs, naive_pool, nkey,
                             p_err, tinf, p_block))
    # for every parameter set, run the simulation repeat times and write
    # results to the file
    for p in paramsets:
        # set parameters
        cf.dose = [p[0] for i in range(len(p[7]))]
        cf.LFdecay = p[1]
        cf.nGCs = p[2]
        cf.nLFs = p[3]
        cf.naive_pool = p[4]*cf.nGCs
        cf.memory_pool = 100*cf.nGCs  # fixed! (or change manually)
        cf.nkey = p[5]
        cf.lAg = p[5]
        cf.lAb = 220 - p[5]
        cf.p_err_FWR = p[6]
        cf.p_err_CDR = p[6]
        cf.tinf = p[7]
        cf.p_block_FWR = p[8]

        for r in range(repeats):
            # get lists to store individual simulation results
            l_mems = []
            l_KDs = []
            s_KDs = []  # std
            l_SHMs = []
            s_SHMs = []  # std
            l_Entrs = []
            # run simulation and get filepath or dict
            evaldays, l_mems, l_KDs, s_KDs, l_SHMs, s_SHMs, l_Entrs = \
                main(store_export='minimal', evalperday=1)
            # write these lists to file together with parameters used.
            # Identifier system time.
            ID = 'ID_{}'.format(time.time())
            series = pd.Series([cf.dose, cf.LFdecay, cf.nGCs, cf.nLFs,
                                cf.naive_pool, cf.nkey, cf.p_block_FWR,
                                cf.p_err_CDR, cf.tinf, cf.memory_pool,
                                evaldays, np.array(l_mems),
                                np.array(l_KDs), np.array(s_KDs),
                                np.array(l_SHMs), np.array(s_SHMs),
                                np.array(l_Entrs)],
                               index=['dose', 'LFdecay', 'nGCs', 'nLFs',
                                      'naive_pool', 'nkey', 'p_block',
                                      'p_err', 'tinf', 'mem_pool',
                                      'evaldays', 'memcount', 'E_bind',
                                      'E_bind_std', 'SHM', 'SHM_std',
                                      'entropy'])
            seriesdict[ID] = series
    # make dataframe and store it
    df = pd.DataFrame(seriesdict)
    df = df.transpose()
    datafile['data'] = df
    # close datafile
    datafile.close()
    print('END')


def map_import(filepathlist):
    """ Imports data from all given filepaths and provides quick and dirty
    plots for testing purposes. Give filepaths as strings, please."""
    # import all frames and concatenate them
    frames = []
    for path in filepathlist:
        frames.append(pd.read_hdf(path, key='data'))
    df = pd.concat(frames)

    # plot affinities over time
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, sharex=True,
                                                 figsize=(15, 10))
    lines = []
    labels = []
    ax0.set_title('Normalised binding energy')
    ax0.set_ylim([0.6, 1])
    for index, row in df.iterrows():
        l, = ax0.plot(row['evaldays'], row['E_bind'])
        ax0.fill_between(row['evaldays'], row['E_bind']-row['E_bind_std'],
                         row['E_bind']+row['E_bind_std'], color='lightgrey')
        lines.append(l)
        labels.append('dose={}, LFdecay={}, nGCs={}, nLFs={}, naives={}, '
                      'nkey={}, pblock={}'.format(row['dose'], row['LFdecay'],
                                                  row['nGCs'], row['nLFs'],
                                                  row['naive_pool'],
                                                  row['nkey'],
                                                  row['p_block']))

    ax1.set_title('SHM counts')
    ax1.set_ylim([0, 70])
    for index, row in df.iterrows():
        l, = ax1.plot(row['evaldays'], row['SHM'])
        ax1.fill_between(row['evaldays'], row['SHM']-row['SHM_std'],
                         row['SHM']+row['SHM_std'], color='lightgrey')

    ax2.set_title('Memory cell number')
    for index, row in df.iterrows():
        l, = ax2.plot(row['evaldays'], row['memcount'])

    ax3.set_title('Entropy')
    for index, row in df.iterrows():
        l, = ax3.plot(row['evaldays'], row['entropy'])

    plt.legend(lines, labels, loc=(-0.5, -0.5))
    plt.tight_layout()
    seaborn.despine()
    pylab.savefig('figures/overview.pdf', bbox_inches='tight')


def data_import(folder='map_data'):
    """ Imports all simulation results found in the given folder.

    Prints all values that where found for each parameter as well as a list
    of parameter combinations and the number of simulation results available
    for each of them.

    Args:
        folder (string): Name of the folder to parse for data

    Returns:
        pandas datafile with the concatenated data.
    """

    # get all filenames in the data folder
    filenames = listdir(folder)
    # import and merge into one dataframe
    frames = []
    for name in filenames:
        if name.startswith('data'):
            frames.append(pd.read_hdf(folder+'/'+name, key='data'))
    df = pd.concat(frames)

    # replace values which contain lists by proxies as this is not useful
    df.dose = df.dose.apply(np.nanmean)
    df.dose = df.dose.apply(np.round, decimals=3)
    df.tinf = df.tinf.apply(len)
    # replace overall naive number by number per GC
    df.naive_pool = df.naive_pool.divide(df.nGCs)

    # find unique parameter values and print them
    print('LFdecay: {}'.format(df.LFdecay.unique()))
    print('nGCs: {}'.format(df.nGCs.unique()))
    print('nLFs: {}'.format(df.nLFs.unique()))
    print('naive_pool: {}'.format(df.naive_pool.unique()))
    print('nkey: {}'.format(df.nkey.unique()))
    print('p_block: {}'.format(df.p_block.unique()))
    print('p_err: {}'.format(df.p_err.unique()))
    print('dose: {}'.format(df.dose.unique()))
    print('tinf: {}'.format(df.tinf.unique()))

    gf = df.groupby(['dose', 'LFdecay', 'nGCs', 'nLFs', 'naive_pool', 'nkey',
                    'p_block', 'p_err', 'tinf']).count().mem_pool
    print(gf)

    return df


def aff_time_plot_mean(df, paramvar='nkey', dose=[1], LFdecay=[10*12],
                       nGCs=[1], nLFs=[25], naive_pool=[1000],
                       nkey=[1, 2], p_err=[0.003], tinf=[1],
                       p_block=[0.5]):
    """ Plots the simulation results matching the requirements. Import the
    data using data_import and pass the resultung dataframe.

    If there are several results matching identical criteria, their mean is
    calculated and plotted together with one std shaded area.

    Args:
        dataframe (pd df): Contains simulation results as produced by
                            data_import().
        Other parameters: Specify which simulation results are to be plotted.
    """
    # get subdataframe of only results where all requirements are fulfilled.
    sdf = df[(np.round(df.dose, 5).isin(dose)) & (df.LFdecay.isin(LFdecay)) &
             (df.nGCs.isin(nGCs)) & (df.nLFs.isin(nLFs))
             & (df.naive_pool.isin(naive_pool)) & (df.nkey.isin(nkey)) &
             (df.p_err.isin(p_err)) & (df.tinf.isin(tinf)) &
             (df.p_block.isin(p_block))]

    print(len(sdf))
    # group this remainder by everything
    grouped = sdf.groupby(['dose', 'LFdecay', 'nGCs', 'nLFs', 'naive_pool',
                           'nkey', 'p_block', 'p_err', 'tinf'])

    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, sharex=True,
                                        figsize=(7, 13))
    lines = []
    labels = []
    # fake line object for explanation label
    l, = ax0.plot([1, 2, 3], [0.6, 0.7, 0.8], alpha=0)
    lines.append(l)
    labels.append('$dose, LFdecay, n_{GCs}, n_{LFs}, naive pool, n_{key}, p_{block}, p_{err}, n_{boost}$')
    colors = iter(cm.rainbow(np.linspace(0, 1, cc)))
    for key, group in grouped:
        col = next(colors)
        ax0.set_title('Mean Affinity in Memory Pool')
        ax0.set_ylim([0.6, 1])
        ax0.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
        ax0.set_yticklabels(['$10^{-5}$', '$10^{-6}$', '$10^{-7}$',
                             '$10^{-8}$', '$10^{-9}$'])
        ax0.set_ylabel('$K_D$ of mean $E_\mathrm{Ab|Ag}$')
        labels.append(key)
        ebindlist = []
        for index, row in group.iterrows():
            ebindlist.append(row['E_bind'])
            timelist = row['evaldays']/12.
        Ebind = np.mean(np.array(ebindlist), axis=0)
        Estd = np.nanstd(np.array(ebindlist), axis=0)
        l, = ax0.plot(timelist, Ebind, color=col)
        ax0.fill_between(timelist, Ebind-Estd, Ebind+Estd, color=col,
                         alpha=0.3)
        lines.append(l)

        ax1.set_title('Mean SHMs in Memory Pool')
        ax1.set_ylabel('mean SHM count')
        shmlist = []
        for index, row in group.iterrows():
            shmlist.append(row['SHM'])
            timelist = row['evaldays']/12.
        SHMmean = np.mean(np.array(shmlist), axis=0)
        SHMstd = np.std(np.array(shmlist), axis=0)
        l, = ax1.plot(timelist, SHMmean, color=col)
        ax1.fill_between(timelist, SHMmean-SHMstd, SHMmean+SHMstd, color=col,
                         alpha=0.3)

        ax2.set_title('Normalised Shannon Entropy of Memory Pool')
        ax2.set_xticks([0, 7, 25, 35, 50, 63, 75, 100, 126])
        ax2.set_xticklabels(['0', 'I', '25', 'II', '50', 'III', '75', '100',
                             'C'])
        ax2.set_xlim([0, 130])
        ax2.set_xlabel('time (days)')
        ax2.set_ylabel('normalised H (bits)')
        ax2.set_ylim([0, 1])
        shannon = []
        for index, row in group.iterrows():
            shannon.append(np.array(row['entropy'].tolist()) /
                           np.log2(np.array(row['memcount'].tolist())))
            timelist = row['evaldays']/12
        shanmean = np.mean(np.array(shannon), axis=0)
        shanstd = np.std(np.array(shannon), axis=0)
        l, ax2.plot(timelist, shanmean, color=col)
        ax2.fill_between(timelist, shanmean-shanstd, shanmean+shanstd,
                         color=col, alpha=0.3)

        plt.legend(lines, labels, loc=(-0.5, -1))
        plt.tight_layout()
        seaborn.despine()
        pylab.savefig('figures/aff_time_map_mean_{}.pdf'.format(paramvar),
                       bbox_inches='tight')
    plt.show()


def aff_param_plot(df, paramvar='nkey', day=126, dose=[1], LFdecay=[10*12],
                   nGCs=[100], nLFs=[25], naive_pool=[1000],
                   nkey=[1, 2], p_err=[0.003], tinf=[3],
                   p_block=[0.5]):
    """ For the given parameter to vary (paramvar), finds all simulation
    results matching the other given parameter values and calculates mean
    affinity, SHMs, number of memory cells and entropy at the given day and
    plots these over paramvar with one std error bars. """
    # get subdataframe of only results where all requirements are fulfilled.
    sdf = df[(np.round(df.dose, 5).isin(dose)) & (df.LFdecay.isin(LFdecay)) &
             (df.nGCs.isin(nGCs)) & (df.nLFs.isin(nLFs))
             & (df.naive_pool.isin(naive_pool)) & (df.nkey.isin(nkey)) &
             (df.p_err.isin(p_err)) & (df.tinf.isin(tinf)) &
             (df.p_block.isin(p_block))]

    # group by all parameters that are NOT to be varied here
    grouplist = ['dose', 'LFdecay', 'nGCs', 'nLFs', 'naive_pool',
                 'nkey', 'p_block', 'p_err', 'tinf']
    grouplist.remove(paramvar)
    grouped = sdf.groupby(grouplist)
    # open plots
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, sharex=True,
                                                 figsize=(8, 5))
    ax0.set_title('Mean Affinity in Memory Pool at day {}'.format(day))
    ax0.set_ylim([0.6, 1])
    ax0.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
    ax0.set_yticklabels(['$10^{-5}$', '$10^{-6}$', '$10^{-7}$', '$10^{-8}$',
                         '$10^{-9}$'])
    ax0.set_ylabel('$K_D$ of mean $E_\mathrm{Ab|Ag}$')

    ax1.set_title('Mean SHMs in Memory Pool at day {}'.format(day))
    ax1.set_ylabel('number')

    ax2.set_title('Size of Memory Pool at day {}'.format(day))
    ax2.set_ylabel('cell number')
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax2.set_xlabel('${}$'.format(paramvar))

    ax3.set_title('Normalised Entropy of Memory Pool at day {}'.format(day))
    ax3.set_xlabel('${}$'.format(paramvar))
    ax3.set_ylabel('normalised H (bits)')
    ax3.set_ylim([0, 1])
    ax3.set_xlabel(paramvar)
    # find unique values of paramvar in each group
    # find all values at each paramval, plot means
    paramvar_per_group = []
    for key, group in grouped:
        paramvar_per_group = group[paramvar].unique().tolist()
        # affinity
        affs = []
        shms = []
        mems = []
        nentrs = []

        # for each paramvar val, get info
        for pv in paramvar_per_group:
            # move through runs and get required day info
            laff = []
            lshm = []
            lmem = []
            lnentr = []
            for run in range(len(group[(group[paramvar] == pv)])):
                laff.append(group[(group[paramvar] == pv)].E_bind[run][day-1])
                lshm.append(group[(group[paramvar] == pv)].SHM[run][day-1])
                lmem.append(group[(group[paramvar] == pv)].memcount[run][day-1])
                lnentr.append(group[(group[paramvar] == pv)].entropy[run][day-1]/
                              np.log2(group[(group[paramvar] == pv)].memcount[run][day-1]))
            affs.append((np.nanmean(laff), np.nanstd(laff)))
            shms.append((np.nanmean(lshm), np.nanstd(lshm)))
            mems.append((np.nanmean(lmem), np.nanstd(lmem)))
            nentrs.append((np.nanmean(lnentr), np.nanstd(lnentr)))

        # plot stuff gathered here
        ax0.errorbar(paramvar_per_group, [i[0] for i in affs], yerr=[i[1] for i in affs], fmt='o', label=key)
        ax1.errorbar(paramvar_per_group, [i[0] for i in shms], yerr=[i[1] for i in shms], fmt='o', label=key)
        ax2.errorbar(paramvar_per_group, [i[0] for i in mems], yerr=[i[1] for i in mems], fmt='o', label=key)
        ax3.errorbar(paramvar_per_group, [i[0] for i in nentrs], yerr=[i[1] for i in nentrs], fmt='o', label=key)

    # # fake label
    # ll1 = ' '.join(grouplist)
    # ax0.errorbar([0], [0], yerr=0, alpha=0, label='test')
    plt.legend(loc=(-0.5, -1))
    plt.tight_layout()
    seaborn.despine()
    pylab.savefig('figures/aff_{}_map.pdf'.format(paramvar),
                  bbox_inches='tight')


def heatmap_2p(df, paramvar1='dose', paramvar2='p_err', day=126,
               dose=[0.02, 0.05, 0.1, 0.2, 0.5, 1], LFdecay=[10*12],
               nGCs=[50], nLFs=[25], naive_pool=[1000],
               nkey=[10], p_err=[0, 0.0003, 0.001, 0.003, 0.01, 0.03], tinf=[3],
               p_block=[0.55]):

    # get subdataframe of only results where all requirements are fulfilled.
    sdf = df[(np.round(df.dose,3).isin(dose)) & (df.LFdecay.isin(LFdecay)) &
             (df.nGCs.isin(nGCs)) & (df.nLFs.isin(nLFs))
             & (df.naive_pool.isin(naive_pool)) & (df.nkey.isin(nkey)) &
             (df.p_err.isin(p_err)) & (df.tinf.isin(tinf)) &
             (df.p_block.isin(p_block))]

    # get a parameter value pair string for each non mapped parameter
    paravals = zip([dose, np.array(LFdecay)/12, nGCs, nLFs, naive_pool, nkey, p_block,
                    p_err, tinf], [r'$dose=$ ', r'$LFdecay=$ ', r'$n_{GCs}=$ ', r'$n_{LFs}=$ ',
                    r'$naive\ pool=$ ', r'$n_{key}=$ ', r'$p_{block}=$ ', r'$p_{err}=$ ', r'$n_{inf}=$ '])
    textblock = r'Parameters \\'
    for pv in paravals:
        textblock += pv[1] + (r'${}$\\'.format(pv[0]))
    print(textblock)

    # get unique values of the two variable parameters and prepare plot labels
    param1 = sdf[paramvar1].unique().tolist()
    param1.sort()
    labels1 = map(str, param1)
    print(labels1)

    param2 = sdf[paramvar2].unique().tolist()
    param2.sort()
    labels2 = map(str, param2)
    print(labels2)

    # get empty arrays of size p1*p2 to collect means later
    AffM = np.zeros((len(param1), len(param2)))
    AffS = np.zeros((len(param1), len(param2)))
    SHMM = np.zeros((len(param1), len(param2)))
    MemM = np.zeros((len(param1), len(param2)))
    EntrM = np.zeros((len(param1), len(param2)))

    # go through all combinations and get energy means, put to arrays
    for i in range(len(param1)):
        for j in range(len(param2)):
            laff = []
            lshm = []
            lmem = []
            lnentr = []
            for run in range(len(sdf[(sdf[paramvar1] == param1[i]) &
                                     (sdf[paramvar2] == param2[j])])):
                laff.append(sdf[(sdf[paramvar1] == param1[i]) &
                        (sdf[paramvar2] == param2[j])].E_bind[run][day-1])
            # for run in range(len(sdf[(aae(sdf[paramvar1], param1[i])) &
            #                          (aae(sdf[paramvar2], param2[j]))])):
            #     laff.append(sdf[(np.allclose(sdf[paramvar1], param1[i])) &
            #                              (np.allclose(sdf[paramvar2], param2[j]))].E_bind[run][day-1])
                # lshm.append(sdf[(sdf[paramvar1] == param1[i]) &
                #                          (sdf[paramvar2] == param2[j])].SHM[run][day-1])
                # lmem.append(sdf[(sdf[paramvar1] == param1[i]) &
                #                          (sdf[paramvar2] == param2[j])].memcount[run][day-1])
                # lnentr.append(sdf[(sdf[paramvar1] == param1[i]) &
                #                          (sdf[paramvar2] == param2[j])].entropy[run][day-1]/
                #               np.log2(sdf[(sdf[paramvar1] == param1[i]) &
                #                                        (sdf[paramvar2] == param2[j])].memcount[run][day-1]))

            AffM[i][j] = np.nanmean(laff)
            AffS[i][j] = np.nanstd(laff)
            SHMM[i][j] = np.nanmean(lshm)
            MemM[i][j] = np.nanmean(lmem)
            EntrM[i][j] = np.nanmean(lnentr)

    print(AffM)
    print(AffS)
    # # open plots
    fig, ax0 = plt.subplots(1, 1, figsize=(7,7))
    ax0.set_title('Mean affinity of memory pool on day {}'.format(day))

    sth = ax0.pcolor(AffM, vmin=0.6, vmax=0.9, cmap=plt.cm.coolwarm)
    ax0.set_xticks(np.arange(AffM.shape[1])+0.5, minor=False)
    ax0.set_yticks(np.arange(AffM.shape[0])+0.5, minor=False)
    ax0.set_xticklabels(labels2, minor=False)
    labels = ax0.get_xticklabels()
    plt.setp(labels, rotation=90)
    ax0.set_yticklabels(labels1, minor=False)

    ax0.set_xlabel(paramvar2)
    ax0.set_ylabel(paramvar1)
    # colourbar for affinities
    cbar_ax = fig.add_axes([1.02, 0.1, 0.025, 0.8])
    normKd = mpl.colors.Normalize(vmin=0.6,vmax=0.9)
    cmap=plt.cm.coolwarm
    cb1 = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap,
                                    norm=normKd,
                                    orientation='vertical',extend='min')
    cb1.set_label('$K_D$ of mean $E_\mathrm{Ab|Ag}$ \n in memory pool (mol/l)')
    cb1.set_ticks([0.6,0.7,0.8,0.9])
    cb1.set_ticklabels(['$10^{-6}$','$10^{-7}$','$10^{-8}$','$10^{-9}$'])

    seaborn.despine()
    plt.text(5,0.,textblock)
    pylab.savefig('paramaps/heatmap_{}_{}_day{}.pdf'.format(paramvar1,paramvar2,day), bbox_inches='tight')


def return_mean_std(df, dose=[0.5], LFdecay=[120],
                    nGCs=[50], nLFs=[25], naive_pool=[1000],
                    nkey=[10], p_err=[0.003], tinf=[3],
                    p_block=[0.55]):
    """ Averages the simulation results matching the requirements, returns
    mean and std of energy, SHMs and normalised H. Take care to feed only
    a single set of paramters for meaningful results.

    Args:
        dataframe (pd df): Contains simulation results as produced by
                            data_import().
        Other parameters: Specify which simulation results are to be averaged.
    """
    # get subdataframe of only results where all requirements are fulfilled.
    sdf = df[(np.round(df.dose, 5).isin(dose)) & (df.LFdecay.isin(LFdecay)) &
             (df.nGCs.isin(nGCs)) & (df.nLFs.isin(nLFs))
             & (df.naive_pool.isin(naive_pool)) & (df.nkey.isin(nkey)) &
             (df.p_err.isin(p_err)) & (df.tinf.isin(tinf)) &
             (df.p_block.isin(p_block))]

    print(len(sdf))
    # group this remainder by everything
    grouped = sdf.groupby(['dose', 'LFdecay', 'nGCs', 'nLFs', 'naive_pool',
                           'nkey', 'p_block', 'p_err', 'tinf'])

    for key, group in grouped:
        ebindlist = []
        for index, row in group.iterrows():
            ebindlist.append(row['E_bind'])
            timelist = row['evaldays']/12.
        Ebind = np.mean(np.array(ebindlist), axis=0)
        Estd = np.nanstd(np.array(ebindlist), axis=0)

        shmlist = []
        for index, row in group.iterrows():
            shmlist.append(row['SHM'])
            timelist = row['evaldays']/12.
        SHMmean = np.mean(np.array(shmlist), axis=0)
        SHMstd = np.std(np.array(shmlist), axis=0)

        shannon = []
        for index, row in group.iterrows():
            shannon.append(np.array(row['entropy'].tolist()) /
                           np.log2(np.array(row['memcount'].tolist())))
        shanmean = np.mean(np.array(shannon), axis=0)
        shanstd = np.std(np.array(shannon), axis=0)

        return Ebind[-1], Estd[-1], shanmean[-1], shanstd[-1]
        # return Ebind, Estd, shanmean, shanstd, timelist
