"""
gc_plots.py contains plotting functionality - most of the plots should
usually be called via one of the functions in gc_head.py, but others can/must
be called directly, see also readme/simulation_examples for
more information. All plots produced are saved to '/figures'.
"""
from __future__ import division

from collections import Counter
import itertools
import math
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.ticker import NullFormatter
import numpy as np
import pandas as pd
import pylab
import seaborn
import sys
import time

import gc_memo
from gc_memo import Ag_density, LF_presence, main
import cf

reload(gc_memo)
reload(cf)

# general plot settings
cc = 30  # number of colours for cycling through
cmap = seaborn.color_palette('Spectral', cc)

seaborn.set_style('ticks')
seaborn.set_context('talk')

plt.rc('text', usetex=True)
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica'],
              'style': 'normal'})
rc("text.latex", preamble=["\\usepackage{helvet}\\usepackage{sfmath}"])
pylab.ion()


def import_file(simdata):
    """ Imports a hdf file storing simulation results or values from a
    dictionary that's passed directly and prepares content for use.

    A storage file contains the following information:
        -- population level --
        'l_times' - time vector of all individual timesteps (NOT evaltimes)
        'l_fn' - counts of free naive cells at every timestep
        'l_fm' - counts of free memory cells at every timestep
        'l_GCs_{i}' - counts of cells in all i GCs
        'LFcurve' - curve of limiting factors at the follicular sites
        'Agcurve' - curve of antigen in the system

        -- cell level --
        'free_{time in times}' - pandas panel with information about all free
            memory cells at this time
        'GC{i}_{time in times}' - pandas panel with information about all cells
            in GC i at this time

        -- other --
        'times' - timevector of times at which detailed information was saved
        'nGCs' - number of GCs used in this run
        'mut_list' - list of all mutations having happened and their effects
        'ms_times' - list of lists of times memory was produced per GC
        'ms_vals' - list of lists of memory affinity per GC, corresponding to
            timepoints
        'ms_fams' - list of lists of memory families per GC, corresponding to
            timepoints
        'ms_muts' - list of lists of memory mutation counts per GC,
            corresponding to timepoints
        'E_list' - list of all energies of sequences in the common sequence
            repertoire
        """
    if type(simdata) is str:  # if a filepath was given, import from file
        nGCs = pd.read_hdf(simdata, 'nGCs')[0].tolist()[0]
        l_times = pd.read_hdf(simdata, 'l_times')[0].tolist()
        l_fn = pd.read_hdf(simdata, 'l_fn')[0].tolist()
        l_fm = pd.read_hdf(simdata, 'l_fm')[0].tolist()
        E_list = pd.read_hdf(simdata, 'E_list')[0].tolist()
        l_GCs = []
        for i in range(nGCs):
            l_GCs.append(pd.read_hdf(simdata,
                                     'l_GCs_{}'.format(i))[0].tolist())
        LFcurve = pd.read_hdf(simdata, 'LFcurve')[0].tolist()
        Agcurve = pd.read_hdf(simdata, 'Agcurve')[0].tolist()

        ms_times = pd.read_hdf(simdata, 'ms_times')
        ms_times = map(list, ms_times.values)
        ms_vals = pd.read_hdf(simdata, 'ms_vals')
        ms_vals = map(list, ms_vals.values)
        ms_fams = pd.read_hdf(simdata, 'ms_fams')
        ms_fams = map(list, ms_fams.values)
        ms_muts = pd.read_hdf(simdata, 'ms_muts')
        ms_muts = map(list, ms_muts.values)

        mut_listX = pd.read_hdf(simdata, 'mut_list')
        m0 = list(mut_listX.values[:, 0])
        m1 = list(mut_listX.values[:, 1])
        m2 = list(mut_listX.values[:, 2])
        m3 = list(mut_listX.values[:, 3])
        mut_list = zip(m0, m1, m2, m3)

        evaltimes = pd.read_hdf(simdata, 'times')[0].tolist()
        complete_Mem = {}  # dict for memory info data frames
        complete_GCs = [{} for gc in range(nGCs)]

        for tp in evaltimes:
            complete_Mem[tp] = pd.read_hdf(simdata, 'free_{}'.format(tp))
            for gc in range(nGCs):
                complete_GCs[gc][tp] = pd.read_hdf(simdata,
                                                   'GC{0}_{1}'.format(gc, tp))

        freePan = pd.Panel(complete_Mem)
        GCPans = []
        for i in range(nGCs):
            GCPan = pd.Panel(complete_GCs[i])
            GCPans.append(GCPan)

    elif type(simdata) is dict:  # if a dictionary was given, extract data
        nGCs = simdata['nGCs'][0].tolist()[0]
        l_times = simdata['l_times'][0].tolist()
        l_fn = simdata['l_fn'][0].tolist()
        l_fm = simdata['l_fm'][0].tolist()
        E_list = simdata['E_list'][0].tolist()
        l_GCs = []
        for i in range(nGCs):
            l_GCs.append(simdata['l_GCs_{}'.format(i)][0].tolist())
        LFcurve = simdata['LFcurve'][0].tolist()
        Agcurve = simdata['Agcurve'][0].tolist()

        ms_times = simdata['ms_times']
        ms_times = map(list, ms_times.values)
        ms_vals = simdata['ms_vals']
        ms_vals = map(list, ms_vals.values)
        ms_fams = simdata['ms_fams']
        ms_fams = map(list, ms_fams.values)
        ms_muts = simdata['ms_muts']
        ms_muts = map(list, ms_muts.values)

        m0 = simdata['mut_list'][0].tolist()
        m1 = simdata['mut_list'][1].tolist()
        m2 = simdata['mut_list'][2].tolist()
        m3 = simdata['mut_list'][3].tolist()
        mut_list = zip(m0, m1, m2, m3)

        evaltimes = simdata['times'][0].tolist()
        complete_Mem = {}  # dict for memory info data frames
        complete_GCs = [{} for gc in range(nGCs)]

        for tp in evaltimes:
            complete_Mem[tp] = simdata['free_{}'.format(tp)]
            for gc in range(nGCs):
                complete_GCs[gc][tp] = simdata['GC{0}_{1}'.format(gc, tp)]

        freePan = pd.Panel(complete_Mem)
        GCPans = []
        for i in range(nGCs):
            GCPan = pd.Panel(complete_GCs[i])
            GCPans.append(GCPan)
    else:  # throw error
        sys.exit("""The object given to the import function is neither a
                    filepath nor a dictionary and can therefore not be
                    imported.""")

    return (l_times, l_fn, l_fm, l_GCs, LFcurve, Agcurve, evaltimes,
            freePan, GCPans, ms_times, ms_vals, ms_fams, ms_muts,
            mut_list, E_list)


def population_plot(l_times, l_fn, l_fm, l_GCs, runID):
    """ Plots sizes of naive cell, memory cell, and each GC population as
    a function of time. """
    plt.figure()
    plt.plot(l_times, l_fn, label='free naive')
    plt.plot(l_times, l_fm, label='free memory')
    for i in range(len(l_GCs)):
        plt.plot(l_times, list(l_GCs[i]), label='cells in GC {}'.format(i))
    plt.legend(loc=0)
    plt.xlabel('time (days)')
    plt.ylabel('cell number')
    seaborn.despine()
    pylab.savefig('figures/population_plot{}.pdf'.format(runID))


def GC_dynamics_plot(GCPan, ms_times, ms_fams, ms_vals, ms_muts, runID, GC_ID):
    """ Plots the clonal dynamics within a GC together with the memory cell
    qualities produced by this GC. Take care of feeding a single GC Pan and
    the corresponding memory lists!
    Also, plots affinity/mutation scatter plots for the largest 1-3
    producing clones of the GC."""
    # get Counters for all timepoints and a list of families having appeared
    CList = []
    mC = []
    tList = GCPan.keys()
    for tp in range(len(tList)):
        C = Counter(GCPan[tList[tp]]['family'].dropna())
        CList.append(C)
        mC = mC + list(C)
    all_fams = list(set(mC))
    # prepare array with length timepoints and height of all families
    fam_times = np.zeros((len(all_fams), len(tList)))
    for f in range(len(all_fams)):
        for tp in range(len(tList)):
            fam_times[f][tp] = CList[tp][all_fams[f]]
    # collect all cells that remain unique in one row
    uniques = np.zeros(len(tList))
    fam_times2 = []
    cols = ['lightgrey']  # grey is for unique fraction
    for row in range(len(all_fams)):
        if max(fam_times[row]) <= 1:
            uniques = uniques + fam_times[row]
        else:
            fam_times2.append(fam_times[row])
            cols.append(cmap[int(all_fams[row] % cc)])
    fam_times2 = [uniques] + fam_times2
    fam_arr = np.array(fam_times2)
    # plot absolute stackplot with fam_times2

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(5, 10))
    axes[0].stackplot(np.array(tList)/12., fam_arr, colors=cols,
                      edgecolor=None)
    axes[0].set_ylabel('cell number')
    cols2 = [cmap[int(family % cc)] for family in ms_fams
             if not math.isnan(family)]
    axes[1].scatter(np.array(ms_times)/float(12), ms_vals, c=cols2, s=30, lw=0)
    axes[1].set_xlabel('time (days)')
    axes[1].set_xlim(xmin=0)
    axes[1].set_ylabel('$K_D$ of memory \n cells produced (mol/l)')
    axes[1].set_ylim([0.6, 1.03])
    axes[1].set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
    axes[1].set_yticklabels([r'$10^{-5}$', r'$10^{-6}$', r'$10^{-7}$',
                             r'$10^{-8}$', r'$10^{-9}$'])
    seaborn.despine()
    fig.subplots_adjust(wspace=0.6)
    pylab.savefig('figures/GC_dynamics_plot{0}_{1}.pdf'.format(runID, GC_ID),
                  bbox_inches='tight')

    # scatter plots of affinity over mutation count for three largest clones
    famCounter = Counter(ms_fams).most_common(3)
    largest_list = [x[0] for x in famCounter]
    # for the largest one, plot affinity over mutation scatter with aritificial
    # noise
    for i in [0]:
        these_muts = []
        these_affs = []
        for ff in range(len(ms_fams)):
            if ms_fams[ff] == largest_list[i]:
                # noise for plotting
                affnoise = (np.random.rand()-0.5)*0.02
                mutnoise = (np.random.rand()-0.5)*0.2
                these_affs.append(ms_vals[ff]+affnoise)
                these_muts.append(ms_muts[ff]+mutnoise)
        # specific colour
        this_color = cmap[int(largest_list[i] % cc)]
        plt.figure(figsize=(2, 2))
        plt.scatter(these_muts, these_affs, c=this_color, s=30, lw=0)
        plt.ylim([0.6, 1.03])
        plt.yticks([0.6, 0.7, 0.8, 0.9, 1.0])
        axx = plt.gca()
        axx.set_yticklabels([r'$10^{-5}$', r'$10^{-6}$', r'$10^{-7}$',
                             r'$10^{-8}$', r'$10^{-9}$'])
        plt.xlim([0, 15])
        plt.ylabel('$K_D$ of memory \n cells produced (mol/l)')
        plt.xlabel('SHM count')
        seaborn.despine()
        pylab.savefig('figures/GC_AM_clones{0}_{1}_{2}.pdf'.format(runID,
            GC_ID, i), bbox_inches='tight')


def GC_phases(GCPan, mut_list):
    """ Given a GC panel, calculates for each timepoint: cellnum, clonenum,
    mutations per day and clone, beneficial mutations per day and clone."""
    clonenums = []
    cellnums = []
    tList = GCPan.keys()

    for tp in range(len(tList)):
        cellnum = len(GCPan[tList[tp]]['family'].dropna())
        C = Counter(GCPan[tList[tp]]['family'].dropna())
        clonenums.append(len(C.keys()))
        cellnums.append(cellnum)

    """ Count mutations and beneficial mutations per clone and timestep.
    """
    # get set of clone names involved
    clonelist = [item[1] for item in mut_list]
    cloneset = list(set(clonelist))

    # for each clone, count muts an benmuts into array including all timepoints
    mutarr = np.zeros((len(cloneset), cf.endtime+1))
    benmutarr = np.zeros((len(cloneset), cf.endtime+1))
    tpclones = [[] for tp in range(cf.endtime+1)]

    # count stuff into the arrays
    for item in mut_list:
        # get clone index
        ci = cloneset.index(item[1])
        # time index is time of item
        tj = item[0]
        # add muts and benmuts to arrays
        mutarr[ci][tj] += item[2]
        benmutarr[ci][tj] += item[3]
        # append clonename to clonelist at tp for correct normalisation
        tpclones[item[0]].append(item[1])

    # count number of clones active at each tp for mormalisation
    clonetimenums = [len(set(item)) for item in tpclones]

    # sum up over all clones and get mean (do not normalise with number of LFs)
    mutmean = (np.sum(mutarr, axis=0)/np.array(clonetimenums))
    benmean = (np.sum(benmutarr, axis=0)/np.array(clonetimenums))
    return tList, cellnums, clonenums, cf.endtime, mutmean, benmean


def oneGC(repeats=100):
    """ fig 3C/D, showing clone number, cell number and mutation number per day
    in an average GC """

    cenL = []
    clnL = []
    mmL = []
    bmL = []
    for r in range(repeats):
        # get runID from current system time
        runID = int(time.time())
        # run simulation and get filepath or dict
        simdata = main(runID, store_export='datafile', evalperday=12)
        # import required information for small scale plots
        l_times, l_fn, l_fm, l_GCs, LFcurve, Agcurve, evaltimes, freePan, \
            GCPans, ms_times, ms_vals, ms_fams, ms_muts, mut_list, \
            E_list = import_file(simdata)
        tList, cen, cln, endtime, mm, bm = GC_phases(GCPans[0], mut_list)

        cenL.append(cen)
        clnL.append(cln)
        mmL.append(mm)
        bmL.append(bm)

    cen = np.nanmean(np.array(cenL), axis=0)
    cln = np.nanmean(np.array(clnL), axis=0)
    mm = np.nanmean(np.array(mmL), axis=0)
    bm = np.nanmean(np.array(bmL), axis=0)

    # bin mutation counts into days
    mmbin = []
    bmbin = []
    tend = int(tList[-1]/12)
    for i in [12*j for j in range(tend+1)]:
        if np.isinf(np.nansum(mm[i:i+12])):
            mmbin.append(0)
            bmbin.append(0)
        else:
            mmbin.append(np.nansum(mm[i:i+12]))
            bmbin.append(np.nansum(bm[i:i+12]))

    """ plot """
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 10))
    ax1.plot(np.array(tList)/12., cen, label='cells/GC', color='crimson')
    ax1.plot(np.array(tList)/12., cln, label='clones/GC',
             color='cornflowerblue')
    ax1.set_ylabel('count')
    ax1.legend(loc=0)
    seaborn.despine()

    ax2.plot(range(tend+1), mmbin, '-o', label='all', color='crimson')
    ax2.plot(range(tend+1), np.array(bmbin)*10, '-o',
             label='beneficial ($\cdot 10$)', color='cornflowerblue')
    ax2.legend(loc=0)
    ax2.set_ylabel('mutations/(clone$\cdot$day)')
    ax2.set_xlabel('time after infection (days)')

    seaborn.despine()
    pylab.savefig('figures/oneGC.pdf', bbox_inches='tight')

    # write the matrices to file
    datasave = open('processed_data/datafile_oneGC', 'w')
    datasave.write(str(tList)+'\n')
    datasave.write(str(cen)+'\n')
    datasave.write(str(cln)+'\n')
    datasave.write(str(mmbin)+'\n')
    datasave.write(str(bmbin)+'\n')
    datasave.close()


def energy_distributions_plot(E_list, ancestor_dist, final_dist, tag):
    """ Plots the 3 energy distributions passed to it as histograms. """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, figsize=(12, 4))

    # naive distribution
    ax1.hist(E_list, bins=np.linspace(0, 1, 41), color='crimson',
             edgecolor='none')
    ax1.set_xlim([0.6, 1])
    ax1.set_xticks([0.6, 0.7, 0.8, 0.9, 1])
    ax1.set_xticklabels([r'$10^{-5}$', r'$10^{-6}$', r'$10^{-7}$',
                         r'$10^{-8}$', r'$10^{-9}$'])
    labels = ax1.get_xticklabels()
    plt.setp(labels, rotation=30)
    ax1.set_yticks([])
    ax1.set_title('unselected naive repertoire, \n germline binders')

    ax2.hist(ancestor_dist, bins=np.linspace(0, 1, 41), color='darkseagreen',
             edgecolor='none')
    ax2.set_xlim([0.6, 1])
    ax2.set_xticks([0.6, 0.7, 0.8, 0.9, 1])
    ax2.set_xticklabels([r'$10^{-5}$', r'$10^{-6}$', r'$10^{-7}$',
                         r'$10^{-8}$', r'$10^{-9}$'])
    labels = ax2.get_xticklabels()
    plt.setp(labels, rotation=30)
    ax2.set_yticks([])
    ax2.set_title('germline ancestors of \n memory B cells on day {}'.format(
        tag))

    ax3.hist(final_dist, bins=np.linspace(0, 1, 41), color='cornflowerblue',
             edgecolor='none')
    ax3.set_xlim([0.6, 1])
    ax3.set_xticks([0.6, 0.7, 0.8, 0.9, 1])
    ax3.set_xticklabels([r'$10^{-5}$', r'$10^{-6}$', r'$10^{-7}$',
                         r'$10^{-8}$', r'$10^{-9}$'])
    labels = ax3.get_xticklabels()
    plt.setp(labels, rotation=30)
    ax3.set_yticks([])
    ax3.set_title('memory B cells \n on day {}'.format(tag))

    fig.text(0.5, -0.15, 'affinity $K_D$ (mol/l)', ha='center')
    ax1.set_ylabel('relative abundance')
    seaborn.despine()
    pylab.savefig('figures/distribution_plot_{}.pdf'.format(tag),
                  bbox_inches='tight')


def energy_scatter_plot(ancestor_dist, final_dist, tag):
    """ Plots scatter plots of current binding energy versus germline ancestor
    bindung energy for the given cells together with marginal histograms of
    these binding energy distributions. """
    length = len(ancestor_dist)
    same = len(np.where(np.array(ancestor_dist) == np.array(final_dist))[0])
    better = len(np.where(np.array(ancestor_dist) < np.array(final_dist))[0])
    worse = len(np.where(np.array(ancestor_dist) > np.array(final_dist))[0])
    print('time = {}, unaltered = {}, better = {}, worse = {}'.format(tag,
          same/length, better/length, worse/length))

    left, width = 0.1, 0.55
    bottom, height = 0.1, 0.55
    bottom_h = left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.3]
    rect_histy = [left_h, bottom, 0.3, height]

    # start with a rectangular Figure
    plt.figure(figsize=(7, 7))

    axScatter = plt.axes(rect_scatter)
    axScatter.set_xlim([0.6, 1])
    axScatter.set_xticks([0.6, 0.7, 0.8, 0.9, 1])
    axScatter.set_xticklabels([r'$10^{-5}$', r'$10^{-6}$', r'$10^{-7}$',
                               r'$10^{-8}$', r'$10^{-9}$'])
    axScatter.set_xlabel('affinity $K_D$ (mol/l) of germline ancestor')
    axScatter.set_ylim([0.6, 1])
    axScatter.set_yticks([0.6, 0.7, 0.8, 0.9, 1])
    axScatter.set_yticklabels([r'$10^{-5}$', r'$10^{-6}$', r'$10^{-7}$',
                               r'$10^{-8}$', r'$10^{-9}$'])
    axScatter.set_ylabel('affinity $K_D$ (mol/l) of memory B cell')

    axHistx = plt.axes(rect_histx)
    axHistx.set_yticks([])
    axHistx.set_ylabel('affinity distribution \n germline ancestors')
    axHisty = plt.axes(rect_histy)
    axHisty.set_xticks([])
    axHisty.set_xlabel('affinity distribution \n memory B cells')
    # remove axis labeling in places
    nullfmt = NullFormatter()
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    # Add noise
    ancestor_dist = ancestor_dist + 0.01*(np.random.rand(
        len(ancestor_dist))-0.5)
    final_dist = final_dist + 0.01*(np.random.rand(len(ancestor_dist))-0.5)
    axScatter.plot(np.linspace(0.6, 1, 100), np.linspace(0.6, 1, 100), 'k--',
                   linewidth=2, zorder=0)
    axScatter.scatter(ancestor_dist, final_dist, color='grey', alpha=0.1,
                      edgecolor=None, s=10, zorder=5)
    axScatter.text(0.65, 0.9, 'affinity improved \n by mutation', zorder=10)
    axScatter.text(0.8, 0.65, 'affinity impaired \n by mutation', zorder=10)
    axScatter.text(1.1, 1.1, 'day {}'.format(tag), fontsize=14,
                   bbox=dict(boxstyle='round', facecolor='crimson', alpha=0.1))

    # if booster is given on this day, indicate in the plot
    if tag*12 in cf.tinf:
        axScatter.text(1.02, 1.15, 'VACCINATION DAY', fontsize=18,
                       bbox=dict(boxstyle='round', facecolor='crimson',
                                 alpha=0.4))
    axHistx.hist(ancestor_dist, bins=np.linspace(0, 1, 41),
                 color='darkseagreen', edgecolor='none')
    axHisty.hist(final_dist, bins=np.linspace(0, 1, 41),
                 orientation='horizontal', color='cornflowerblue',
                 edgecolor='none')

    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())

    seaborn.despine()
    # modify tag for film purposes
    if tag < 10:
        savename = 'figures/energy_scatter_plot_00{}'.format(tag)
    elif tag < 100:
        savename = 'figures/energy_scatter_plot_0{}'.format(tag)
    else:
        savename = 'figures/energy_scatter_plot_{}'.format(tag)
    pylab.savefig(savename + '.pdf', bbox_inches='tight')
    # pylab.savefig(savename + '.png', bbox_inches='tight')
    plt.close()
    plt.close()


def stacked_energy_plot(bins, sum_plus, sum_minus, sum_zero, analysis_time):
    """ Plots unaltered, improved and impaired binding energies in a stacked
    histogram."""
    left_bins = bins[:-1]
    width = left_bins[1] - left_bins[0]
    plt.figure(figsize=(5, 4))
    p1 = plt.bar(left_bins, sum_plus, width, color='darkseagreen',
                 edgecolor='none')
    p2 = plt.bar(left_bins, sum_minus, width, bottom=sum_plus, color='crimson',
                 edgecolor='none')
    p3 = plt.bar(left_bins, sum_zero, width, bottom=sum_minus+sum_plus,
                 color='darkgrey', edgecolor='none')
    plt.ylabel('relative abundance in \n memory B cells on day {}'.format(
        analysis_time))
    plt.xticks([0.6, 0.7, 0.8, 0.9, 1], [r'$10^{-5}$', r'$10^{-6}$',
                                         r'$10^{-7}$', r'$10^{-8}$',
                                         r'$10^{-9}$'])
    plt.xlim([0.6, 1])
    plt.xlabel('affinity $K_D$ (M)')
    plt.yticks([])
    plt.legend((p1[0], p2[0], p3[0]), ('improved', 'impaired', 'unchanged'),
               loc=0)
    seaborn.despine()
    pylab.savefig('figures/stacked_histogram.pdf', bbox_inches='tight')


def sample_statistics_plot(subsample, tList, MSHM, SSHM,
                           MEntropies, SEntropies, Mclusterfracs,
                           Sclusterfracs):
    """ Plots mean hypermutation count in sample and clonal expansion
    (fraction of non-unique cells within the sample of size subsample).
    Shaded area covers three standard deviations. Axes labeling is optimised
    for TUCHMI trial protocol."""
    # 2x1 figure
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(9.5, 3.5))
    fig.subplots_adjust(wspace=0.6)
    ax1.set_ylabel('memory cells \n in sample')

    ax1.set_xlim(xmin=-1, xmax=130)
    ax1.set_xticks([0, 7, 25, 35, 50, 63, 75, 100, 126])
    ax1.set_xticklabels(['0', 'I', '25', 'II', '50', 'III', '75', '100', 'C'])
    ax1.set_xlabel('time (days)')
    ax1.set_ylabel('mean SHM \n in sample of size {}'.format(subsample))

    ax2.set_ylabel('clonal expansion \n in sample of size {}'.format(
        subsample))
    ax2.set_ylim([-0.05, 1])
    ax2.set_xlabel('time (days)')
    ax2.set_xlim(xmin=-1, xmax=130)
    ax2.set_xticks([0, 7, 25, 35, 50, 63, 75, 100, 126])
    ax2.set_xticklabels(['0', 'I', '25', 'II', '50', 'III', '75', '100', 'C'])
    ax2.set_xlabel('time (days)')

    # plot mean and std bands
    ax1.plot(np.array(tList)/12., MSHM, color='darkgrey', zorder=2)
    ax1.fill_between(np.array(tList)/12., MSHM-1*SSHM, MSHM+1*SSHM,
                     color='grey', alpha=0.3, zorder=1)

    ax2.plot(np.array(tList)/12., Mclusterfracs, color='darkgrey', zorder=2)
    ax2.fill_between(np.array(tList)/12., Mclusterfracs-1*Sclusterfracs,
                     Mclusterfracs+1*Sclusterfracs, color='grey', alpha=0.3,
                     zorder=1)

    seaborn.despine()
    pylab.savefig('figures/sample_statistics_plot.pdf', bbox_inches='tight')


def sample_scatter_plot(KD_list, SHM_list, orglist):
    """ Plots scatter plots of affinity of individual memory cells versus
    their SHM count as sampled 7 days after each infection."""

    # plot into given number of subplots
    fig, ax = plt.subplots(1, 3, sharey=True, figsize=(5, 2))

    for d in range(len(KD_list)):
        # ax[d].set_title('day {}'.format(timecourse[d]/12))
        ax[d].set_yscale('log')
        ax[d].set_ylim(ymin=math.pow(10, -9), ymax=math.pow(10, -5))
        ax[d].set_ylim(ax[d].get_ylim()[::-1])
        ax[d].set_xlim(xmin=-2, xmax=53)
        # translate origins into colorcode
        cols = ['darkgrey' if org == 'umem' else 'crimson' for org in
                orglist[d]]
        ax[d].scatter(SHM_list[d], KD_list[d], linewidth=0, c=cols)

    ax[0].set_title('I (day 7)')
    ax[1].set_title('II (day 35)')
    ax[2].set_title('III (day 63)')
    ax[0].set_ylabel('affinity $K_D$ (mol/l)')
    ax[1].set_xlabel('SHM')

    # create fake objects for legend
    ax[2].scatter([-10], [1], c='darkgrey', label='memory \n derived',
                  linewidth=0)
    ax[2].scatter([-10], [1], c='crimson', label='naive \n derived',
                  linewidth=0)
    ax[2].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    seaborn.despine()
    pylab.savefig('figures/sample_scatter_plot.pdf', bbox_inches='tight')


def clonal_scatter_plot(iSHMs, iKDs, iTPs):
    """ Plots SHM/KD scatter plots for individual clones. Each panel contains
    cells that belong to a clone. Clones which were sampled twice or more
    may appear, but since only 15 random clusters are shown, not all
    will. Colouring of dots according to the timepoint that each cell was
    sampled at."""

    # plot the first 15 clusters into separate plots
    fig, axes = plt.subplots(3, 5, sharey=True, sharex=True)
    fig.subplots_adjust(bottom=0.1, left=0.1)

    for cc in range(min(15, len(iSHMs))):
        fig.axes[cc].scatter(iSHMs[cc], iKDs[cc], linewidth=0, color=iTPs[cc])
        fig.axes[cc].set_yscale('log')
        fig.axes[cc].set_ylim(ymin=math.pow(10, -9), ymax=math.pow(10, -5))
        fig.axes[cc].set_xlim(xmin=-0.2, xmax=30)
        fig.axes[cc].set_xticks([0, 10, 20, 30])
        fig.axes[cc].set_ylim(fig.axes[cc].get_ylim()[::-1])
    fig.text(0.5, 0.0, 'SHM', ha='center')
    fig.text(0.0, 0.5, 'affinity $K_D$ (mol/l)', va='center',
             rotation='vertical')
    seaborn.despine()
    pylab.savefig('figures/clonal_scatter_plot.pdf', bbox_inches='tight')


def pool_affinity_plot(tList, Elist):
    """ Plots affinity as a function of time given a list of timepoints and a
    list of normalised mean energies. """
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(np.array(tList)/12., Elist, color='black')
    ax.set_xlabel('time (days)')
    ax.set_ylim([0.6, 1])
    ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticklabels(['$10^{-5}$', '$10^{-6}$', '$10^{-7}$', '$10^{-8}$',
                        '$10^{-9}$'])
    ax.set_ylabel('$K_D$ of mean $E_\mathrm{Ab|Ag}$ \n in memory pool (mol/l)')
    ax.set_xticks([0, 7, 25, 35, 50, 63, 75, 100, 126])
    ax.set_xticklabels(['0', 'I', '25', 'II', '50', 'III', '75', '100', 'C'])
    seaborn.despine()
    pylab.savefig('figures/pool_affinity_plot.pdf', bbox_inches='tight')


def AM_effect_plot(topElist):
    """ Plots affinity maturation within a single GC following infection in order
    to allow comparison of different numbers of key positions. """
    # colorlist for plotting
    colorr = itertools.cycle(('crimson', 'darkseagreen', 'cornflowerblue',
                              'darkgrey', 'tomato', 'darkblue', 'lightgreen'))
    fig, ax = plt.subplots()
    for item in topElist:
        col = colorr.next()
        ax.plot(np.array(item[1])/12., item[2], color=col,
                label='$n_\mathrm{key} $ ='+' {}'.format(item[0]))
        ax.fill_between(np.array(item[1])/12., item[2]-item[3],
                        item[2]+item[3], color=col, alpha=0.3)
    ax.set_xlabel('time after infection (days)')
    ax.set_ylabel('$K_D$ of mean $E_\mathrm{Ab|Ag}$ \n in single GCs (mol/l)')
    ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticklabels(['$10^{-5}$', '$10^{-6}$', '$10^{-7}$',
                        '$10^{-8}$', '$10^{-9}$'])
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    seaborn.despine()
    pylab.savefig('figures/AM_effect_plot.pdf', bbox_inches='tight')


def Ag_LF_plot():
    """ Produces a plot of Ag presence and resulting limiting factor presence
    at the follicular site according to the infection schedule."""
    agden = Ag_density()
    lfden = LF_presence()
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 10))

    ax[0].plot(np.array(range(cf.endtime+1))/float(12), np.array(agden),
               color='black')
    ax[0].set_ylabel('Ag presence in the \n system (\% of max)')
    ax[1].set_xticks([0, 7, 25, 35, 50, 63, 75, 100, 126])
    ax[1].set_xticklabels(['0', 'I', '25', 'II', '50', 'III', '75', '100',
                           'C'])
    ax[1].plot(np.array(range(cf.endtime+1))/float(12), np.array(lfden),
               color='black')
    ax[1].set_ylabel('Number \n of $T_{FH}$ cells')
    ax[1].set_xlabel('time (days)')
    ax[1].set_ylim([0, 26])
    seaborn.despine()
    pylab.savefig('figures/Ag_LF_plot.pdf', bbox_inches='tight')


def affinity_entropy_plot():
    """ For plotting affinity and entropy over time, put in values by hand for
    now.
    The simulation results are extracted from the folder of cluster simulations
    entitled 'runs2017_03_24/rest/data'. """
    # data: KD and entropy of standard conditions
    timelist =  np.array([   0.,    1.,    2.,    3.,    4.,    5.,    6.,    7.,    8.,
           9.,   10.,   11.,   12.,   13.,   14.,   15.,   16.,   17.,
          18.,   19.,   20.,   21.,   22.,   23.,   24.,   25.,   26.,
          27.,   28.,   29.,   30.,   31.,   32.,   33.,   34.,   35.,
          36.,   37.,   38.,   39.,   40.,   41.,   42.,   43.,   44.,
          45.,   46.,   47.,   48.,   49.,   50.,   51.,   52.,   53.,
          54.,   55.,   56.,   57.,   58.,   59.,   60.,   61.,   62.,
          63.,   64.,   65.,   66.,   67.,   68.,   69.,   70.,   71.,
          72.,   73.,   74.,   75.,   76.,   77.,   78.,   79.,   80.,
          81.,   82.,   83.,   84.,   85.,   86.,   87.,   88.,   89.,
          90.,   91.,   92.,   93.,   94.,   95.,   96.,   97.,   98.,
          99.,  100.,  101.,  102.,  103.,  104.,  105.,  106.,  107.,
         108.,  109.,  110.,  111.,  112.,  113.,  114.,  115.,  116.,
         117.,  118.,  119.,  120.,  121.,  122.,  123.,  124.,  125.])

    Ebind = np.array([0.65004671,  0.65006268,  0.65007861,  0.6500739 ,  0.65005364,
         0.65011532,  0.65505154,  0.66495895,  0.67463439,  0.6831568 ,
         0.69031496,  0.69627608,  0.70117548,  0.70529091,  0.70872266,
         0.71164673,  0.71417912,  0.71637392,  0.71828796,  0.71995978,
         0.72145351,  0.72275291,  0.72390019,  0.72497868,  0.72574045,
         0.72574045,  0.72574045,  0.72574045,  0.72574493,  0.72573877,
         0.72573523,  0.72573293,  0.72531561,  0.7225415 ,  0.72247822,
         0.72643501,  0.73076536,  0.73502033,  0.73881644,  0.74215562,
         0.74507906,  0.74762106,  0.7498564 ,  0.75181757,  0.75354478,
         0.75506763,  0.7564112 ,  0.75759399,  0.75863855,  0.75958075,
         0.7604323 ,  0.76119132,  0.76186332,  0.76211769,  0.76211769,
         0.76211769,  0.76211793,  0.76211488,  0.76212578,  0.76212066,
         0.76175995,  0.75983091,  0.76061302,  0.76384088,  0.76693234,
         0.76989468,  0.77254534,  0.77491325,  0.77700414,  0.7788485 ,
         0.78047799,  0.78193088,  0.78322301,  0.78437017,  0.78539661,
         0.78629868,  0.78710817,  0.78782606,  0.78850257,  0.7890877 ,
         0.78962058,  0.78982502,  0.78982502,  0.78982502,  0.78982502,
         0.78982502,  0.78982502,  0.78982502,  0.78982502,  0.78982502,
         0.78982502,  0.78982502,  0.78982502,  0.78982502,  0.78982502,
         0.78982502,  0.78982502,  0.78982502,  0.78982502,  0.78982502,
         0.78982502,  0.78982502,  0.78982502,  0.78982502,  0.78982502,
         0.78982502,  0.78982502,  0.78982502,  0.78982502,  0.78982502,
         0.78982502,  0.78982502,  0.78982502,  0.78982502,  0.78982502,
         0.78982502,  0.78982502,  0.78982502,  0.78982502,  0.78982502,
         0.78982502,  0.78982502,  0.78982502,  0.78982502,  0.78982502,
         0.78982502])

    Estd =  np.array([0.00097809,  0.00099589,  0.00101584,  0.00102293,  0.00100237,
         0.00096054,  0.00110161,  0.00161491,  0.00213464,  0.00259889,
         0.00296712,  0.00334343,  0.00363409,  0.00384468,  0.00405935,
         0.0042331 ,  0.00436967,  0.00450409,  0.00461999,  0.00473032,
         0.00482067,  0.00490371,  0.00497543,  0.00504856,  0.00510751,
         0.00510751,  0.00510751,  0.00510751,  0.0051106 ,  0.00511608,
         0.0051151 ,  0.00511199,  0.00509073,  0.00492034,  0.00496355,
         0.00532911,  0.00570899,  0.00608362,  0.00641294,  0.00671704,
         0.00694596,  0.0071554 ,  0.00733497,  0.00750458,  0.00764676,
         0.0077638 ,  0.00786782,  0.00796099,  0.00804248,  0.00811604,
         0.00818646,  0.00825101,  0.00830755,  0.00832413,  0.00832413,
         0.00832413,  0.00832207,  0.00832246,  0.00831904,  0.00831746,
         0.00829374,  0.00814108,  0.00831545,  0.00864606,  0.00892366,
         0.00918199,  0.00940079,  0.00961031,  0.0097925 ,  0.0099527 ,
         0.01008699,  0.0102035 ,  0.01030563,  0.01039988,  0.01048788,
         0.01056055,  0.01062791,  0.01069002,  0.01074569,  0.01079481,
         0.01083379,  0.01085296,  0.01085296,  0.01085296,  0.01085296,
         0.01085296,  0.01085296,  0.01085296,  0.01085296,  0.01085296,
         0.01085296,  0.01085296,  0.01085296,  0.01085296,  0.01085296,
         0.01085296,  0.01085296,  0.01085296,  0.01085296,  0.01085296,
         0.01085296,  0.01085296,  0.01085296,  0.01085296,  0.01085296,
         0.01085296,  0.01085296,  0.01085296,  0.01085296,  0.01085296,
         0.01085296,  0.01085296,  0.01085296,  0.01085296,  0.01085296,
         0.01085296,  0.01085296,  0.01085296,  0.01085296,  0.01085296,
         0.01085296,  0.01085296,  0.01085296,  0.01085296,  0.01085296,
         0.01085296])

    shanmean = np.array([ 1.        ,  1.        ,  1.        ,  1.        ,  0.99990366,
         0.99612694,  0.98072119,  0.95315919,  0.92501866,  0.89805416,
         0.87360926,  0.85193036,  0.83323756,  0.81697775,  0.80301162,
         0.79093933,  0.7803625 ,  0.77114576,  0.76308378,  0.75603941,
         0.749744  ,  0.74429054,  0.73952389,  0.7350867 ,  0.7319753 ,
         0.7319753 ,  0.7319753 ,  0.7319753 ,  0.73216383,  0.73410283,
         0.7352786 ,  0.73600837,  0.73791321,  0.7475818 ,  0.74884895,
         0.73556188,  0.72037582,  0.70526133,  0.69155342,  0.67928464,
         0.66832408,  0.65863513,  0.64997754,  0.64231526,  0.6355    ,
         0.62945702,  0.62412271,  0.61941225,  0.61525705,  0.61150643,
         0.60814782,  0.60516282,  0.60252051,  0.60151804,  0.60151804,
         0.60151804,  0.60170499,  0.60346031,  0.60446735,  0.60512247,
         0.6068191 ,  0.61403632,  0.61105493,  0.598713  ,  0.58684905,
         0.57551246,  0.56543387,  0.55648248,  0.54855798,  0.54158332,
         0.53541585,  0.52990877,  0.52502337,  0.52068001,  0.51679998,
         0.51341027,  0.51038576,  0.50770958,  0.50519189,  0.50302812,
         0.50106411,  0.50031834,  0.50031834,  0.50031834,  0.50031834,
         0.50031834,  0.50031834,  0.50031834,  0.50031834,  0.50031834,
         0.50031834,  0.50031834,  0.50031834,  0.50031834,  0.50031834,
         0.50031834,  0.50031834,  0.50031834,  0.50031834,  0.50031834,
         0.50031834,  0.50031834,  0.50031834,  0.50031834,  0.50031834,
         0.50031834,  0.50031834,  0.50031834,  0.50031834,  0.50031834,
         0.50031834,  0.50031834,  0.50031834,  0.50031834,  0.50031834,
         0.50031834,  0.50031834,  0.50031834,  0.50031834,  0.50031834,
         0.50031834,  0.50031834,  0.50031834,  0.50031834,  0.50031834,
         0.50031834])

    shanstd =  np.array([ 2.04715011e-16,   1.89655920e-16,   1.95188104e-16,
          1.69037857e-16,   6.63767844e-05,   3.54832910e-04,
          7.65180498e-04,   1.25162428e-03,   1.72188784e-03,
          2.19686932e-03,   2.72151530e-03,   3.19382769e-03,
          3.58785446e-03,   3.71451895e-03,   3.93744269e-03,
          4.07867454e-03,   4.14213732e-03,   4.22472128e-03,
          4.25058124e-03,   4.32882945e-03,   4.39813698e-03,
          4.43514615e-03,   4.44662425e-03,   4.43466266e-03,
          4.45080686e-03,   4.45080686e-03,   4.45080686e-03,
          4.45080686e-03,   4.46480089e-03,   4.52829391e-03,
          4.57360042e-03,   4.57794370e-03,   4.56729789e-03,
          4.36064671e-03,   4.46135651e-03,   5.07146372e-03,
          5.47168974e-03,   5.75077639e-03,   6.01665438e-03,
          6.24249317e-03,   6.44292104e-03,   6.69683242e-03,
          6.85216452e-03,   7.07372447e-03,   7.21184125e-03,
          7.38206831e-03,   7.49423227e-03,   7.59331460e-03,
          7.72295630e-03,   7.84730850e-03,   7.95251773e-03,
          8.04030699e-03,   8.11749328e-03,   8.13087577e-03,
          8.13087577e-03,   8.13087577e-03,   8.12895632e-03,
          8.17942281e-03,   8.19770178e-03,   8.21875481e-03,
          8.18595306e-03,   8.06864387e-03,   8.49120175e-03,
          9.18550733e-03,   9.71945361e-03,   1.02169031e-02,
          1.06663645e-02,   1.11608442e-02,   1.16151316e-02,
          1.20384880e-02,   1.24029587e-02,   1.27551670e-02,
          1.30476955e-02,   1.33292656e-02,   1.35912049e-02,
          1.38040717e-02,   1.39993423e-02,   1.41700208e-02,
          1.43436801e-02,   1.44854139e-02,   1.46169065e-02,
          1.46783967e-02,   1.46783967e-02,   1.46783967e-02,
          1.46783967e-02,   1.46783967e-02,   1.46783967e-02,
          1.46783967e-02,   1.46783967e-02,   1.46783967e-02,
          1.46783967e-02,   1.46783967e-02,   1.46783967e-02,
          1.46783967e-02,   1.46783967e-02,   1.46783967e-02,
          1.46783967e-02,   1.46783967e-02,   1.46783967e-02,
          1.46783967e-02,   1.46783967e-02,   1.46783967e-02,
          1.46783967e-02,   1.46783967e-02,   1.46783967e-02,
          1.46783967e-02,   1.46783967e-02,   1.46783967e-02,
          1.46783967e-02,   1.46783967e-02,   1.46783967e-02,
          1.46783967e-02,   1.46783967e-02,   1.46783967e-02,
          1.46783967e-02,   1.46783967e-02,   1.46783967e-02,
          1.46783967e-02,   1.46783967e-02,   1.46783967e-02,
          1.46783967e-02,   1.46783967e-02,   1.46783967e-02,
          1.46783967e-02,   1.46783967e-02,   1.46783967e-02])


    fig, axes = plt.subplots(1, 1, figsize=(5, 3.5))
    axes.set_ylim([0.6, 1])
    axes.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
    axes.set_yticklabels(['$10^{-5}$', '$10^{-6}$', '$10^{-7}$', '$10^{-8}$',
                          '$10^{-9}$'])
    axes.set_ylabel('$K_D$ of mean $E_\mathrm{Ab|Ag}$')
    axes.plot(timelist, Ebind, color='black')
    axes.fill_between(timelist, Ebind-Estd, Ebind+Estd, color='black',
                      alpha=0.2)
    axes.set_xticks([0, 7, 25, 35, 50, 63, 75, 100, 126])
    axes.set_xticklabels(['0', 'I', '25', 'II', '50', 'III', '75', '100', 'C'])
    axes.set_xlabel('time (days); experimental protocol')
    axes.set_xlim([0, 130])
    axes.spines['top'].set_visible(False)
    axes.xaxis.set_ticks_position('bottom')

    axB = axes.twinx()
    axB.plot(timelist, shanmean, color='steelblue')
    axB.set_ylim([0, 1.05])
    axB.set_ylabel('Normalised Shannon Entropy', color='steelblue')
    axB.tick_params('y', color='steelblue')
    axB.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], color='steelblue')
    axB.set_xticks([0, 7, 25, 35, 50, 63, 75, 100, 126])
    axB.set_xticklabels(['0', 'I', '25', 'II', '50', 'III', '75', '100', 'C'])
    axB.set_xlabel('time (days); experimental protocol')
    axB.set_xlim([0, 130])
    axB.fill_between(timelist, shanmean-shanstd, shanmean+shanstd,
                     color='steelblue', alpha=0.2)
    axB.spines['top'].set_visible(False)
    axB.xaxis.set_ticks_position('bottom')

    pylab.savefig('figures/affinity_entropy_plot.pdf', bbox_inches='tight')


def sensit_panels():
    """ For plotting sensitivity to 4 chosen parameters after full TUCHMI, data
    input by hand, values from 'runs2017_03_24/rest/data' and
    'runs2017_03_24/std_heatmap_with_memory/data'. """

    # data
    # data for lower panel: KD at last time point for varying
    standE = 0.790
    standStd = 0.011
    # GC decay time
    tdec_real = [2, 5, 10, 15, 20, 25, 30]
    tdec_std = 10
    tdec_x = np.array(tdec_real)/tdec_std
    tdec_E = [0.721, 0.755, standE, 0.831, 0.843, 0.855, 0.860]
    tdec_err = [0.004, 0.008, standStd, 0.011, 0.009, 0.009, 0.012]
    # GC size
    nLF_real = [5, 12, 25, 38, 50, 62, 75]
    nLF_std = 25
    nLF_x = np.array(nLF_real)/nLF_std
    nLF_E = [0.745, 0.774, standE, 0.798, 0.810, 0.807, 0.812]
    nLF_err = [0.008, 0.008, standStd, 0.008, 0.013, 0.008, 0.010]
    # n_key
    nkey_real = [1, 2, 4, 6, 8, 10, 12, 15]
    nkey_std = 10
    nkey_x = np.array(nkey_real)/nkey_std
    nkey_E = [0.870, 0.855, 0.838, 0.824, 0.813, standE, 0.783, 0.774]
    nkey_err = [0.011, 0.012, 0.012, 0.011, 0.012, standStd, 0.008, 0.007]
    # dose
    dose_real = [0.02, 0.05, 0.1, 0.2, 0.5, 1]
    dose_std = 1
    dose_x = np.array(dose_real)/dose_std
    dose_E = [0.671, 0.702, 0.727, 0.745, 0.772, standE]
    dose_err = [0.006, 0.004, 0.007, 0.011, 0.006, standStd]

    # figure
    fig, axes = plt.subplots(4, 1, figsize=(4, 11), sharey=True)
    axes[0].set_ylim([0.65, 0.92])
    axes[0].set_yticks([0.7, 0.8, 0.9])
    axes[0].set_yticklabels(['$10^{-6}$', '$10^{-7}$', '$10^{-8}$'])
    axes[0].set_ylabel('$K_D$ of mean $E_\mathrm{Ab|Ag}$ \n at challenge (day 126)')

    axes[0].errorbar(nkey_x, nkey_E, yerr=nkey_err, fmt='-o')
    axes[0].set_xlabel(r'n$_{key}$')
    axes[0].set_xlim([0, 1.58])
    axes[0].set_xticks(nkey_x)
    axes[0].set_xticklabels(map(str, nkey_real))

    axes[1].errorbar(nLF_x, nLF_E, yerr=nLF_err, fmt='-o')
    axes[1].set_xlabel('GC peak size (B cells)')
    axes[1].set_xlim([0, 3.15])
    axes[1].set_xticks(nLF_x)
    axes[1].set_xticklabels(['100', '250', '500', '750', '1000', '1250',
                             '1500'])

    axes[2].errorbar(tdec_x, tdec_E, yerr=tdec_err, fmt='-o')
    axes[2].set_xlabel('GC decay constant (days)')
    axes[2].set_xlim([0, 3.15])
    axes[2].set_xticks(tdec_x)
    axes[2].set_xticklabels(map(str, tdec_real))

    axes[3].errorbar(dose_x, dose_E, yerr=dose_err, fmt='-o')
    axes[3].set_xlabel('immunization dose (\% of simulation maximum)')
    axes[3].set_xlim([0, 1.05])
    axes[3].set_xticks([0.02, 0.2, 0.5, 1])
    axes[3].set_xticklabels(['2 \%', '20 \%', '50 \%', '100 \%'])

    seaborn.despine()
    plt.subplots_adjust(hspace=0.5)
    pylab.savefig('figures/sensit_panels.pdf', bbox_inches='tight')
    plt.close()
