"""
gc.head() offers functions for calling specific simulation scenarios and
producing plots and processed data files from the simulation results. Contains
functionality better suited for smaller as well as larger simulations as well
as evaluations of specific parameter changes or vaccination scenarios.
"""

from __future__ import division
import math
import numpy as np
import pandas as pd
import scipy.stats
import time
import seaborn
from collections import Counter

import cf
import gc_memo
import gc_plots

reload(cf)
reload(gc_memo)
reload(gc_plots)

from gc_memo import main
from gc_plots import *


def small_scale(store_export='dictionary'):
    """ Performs a single simulation of the given system, creates
    the following plots:
            - population dynamics overview (free naive cells, free memory cells
            and GC populations over time)
            - for each GC, a clonal composition plot together with its memory
            output in a separate panel
            - for each GC, the evolution of its largest clone's affinities over
            mutation count (this plot contains aritificial noise to increase
            visibility!).

    If store_export is set 'datafile', the simulation data is stored in a
    hdf5 file for future purposes, for 'dictionary' the data is passed
    internally and discarded after the run.

    Recommended only for small simulation sizes with up to ~5 GCs and ~5k
    cells, as otherwise things get crowded and plots get large.
    """
    # get runID from current system time
    runID = int(time.time())
    # run simulation and get filepath or dict
    simdata = main(runID, store_export=store_export, evalperday=12)
    # import required information for small scale plots
    l_times, l_fn, l_fm, l_GCs, LFcurve, Agcurve, evaltimes, freePan, GCPans, \
        ms_times, ms_vals, ms_fams, ms_muts, mut_list, \
        E_list = import_file(simdata)
    # plot population behaviour
    population_plot(l_times, l_fn, l_fm, l_GCs, runID)
    # plot GC contents and memory output for every GC
    for i in range(len(l_GCs)):
        GC_dynamics_plot(GCPans[i], ms_times[i], ms_fams[i], ms_vals[i],
                         ms_muts[i], runID, i)


def TUCHMI_sampling(store_export='datafile', d_export=True, subsample=12):
    """ Performs a single simulation of a given size using the TUCHMI vaccination
    protocol, samples memory from the simulated pool and creates several plots
    summarising the information. User settings regarding the protocol are
    overwritten.

    Plots produced include:
        - mean SHMs and clonal expansion (fraction of cells sampled from
        clones that appeared more than once within the sample) in samples of
        size subsample
        - scatter plot of affinity over mutational status in polyclonal samples
        at TUCHMI time points I, II and III
        - scatter plot of affinity over mutational status at a clonal level,
        cells sampled from three TUCHMI time points merged into single plots
        (but sampling time point encoded in colouring)

    If store_export is set 'datafile', the simulation data is stored in a
    hdf5 file for future purposes, for 'dictionary' the data is passed
    internally and lost after the run.

    If d_export is set True, textfiles containing the sampled data (used for
    plotting) are exported for each plot individually.

    Subsample gives the number of cells to be sampled at each timepoint in
    oder to calculate entropy and unique fraction.

    Can be used for all simulation sizes, but is especially useful for larger
    simulations (e.g. >=50 GCs, 50k cells).
    """
    # give protocol
    cf.endtime = 126*12
    cf.tinf = [0*12, 28*12, 56*12]
    cf.dose = [1, 1, 1]

    # get runID from current system time
    runID = int(time.time())

    # run simulation and get filepath or dict
    simdata = main(runID, store_export=store_export, evalperday=1)

    # import required information
    l_times, l_fn, l_fm, l_GCs, LFcurve, Agcurve, evaltimes, freePan, GCPans, \
        ms_times, ms_vals, ms_fams, ms_muts, mut_list, E_list = \
        import_file(simdata)

    # for affinity-mutation scatter plot, downsample for visibility
    samplefrac = 100./len(freePan[35*12].dropna())

    # get list of lists to catch values at every timepoint
    tList = freePan.keys()
    TT = len(tList)
    SHM_means = [[] for t in range(TT)]
    Entropies = [[] for t in range(TT)]
    clusterfracs = [[] for t in range(TT)]

    """ Cell pool affinity over time """
    # get mean affinity at all time points
    Elist = []
    for tp in range(len(tList)):
        C = freePan[tList[tp]]['affinity'].dropna().mean()
        Elist.append(C)
    # pass energies to plot function
    pool_affinity_plot(tList, Elist)

    """ Mean SHM and clonal expansion within sample of size subsample """
    # sample 100 times to calculate standard deviations
    for nn in range(100):
        for tp in range(TT):
            ttp = 12*tp
            # cellnumber to be sampled is either subsample or, if less cells
            # are available (more of a hypothetic case really), all cells
            cellnum = min(subsample, len(freePan[ttp].dropna()))
            if cellnum > 0:
                cells = freePan[ttp].dropna().sample(cellnum, replace=False)
                c_muts = cells.mutations.tolist()
                SHM_means[tp].append(np.nanmean(c_muts))
                # evaluate entropies and clusterfractions
                CC = Counter(cells.family.tolist())
                Entropies[tp].append(scipy.stats.entropy(CC.values(), base=2)
                                     / math.log(cellnum, 2))
                # count again to find how many clones have one member only,
                # calculate clusterfrac from this
                sizedist = list(CC.values())
                C2 = Counter(sizedist)
                uniquefrac = float(C2[1])/cellnum
                clusterfracs[tp].append(1-uniquefrac)
            else:
                SHM_means[tp].append(np.nan)
                Entropies[tp].append(np.nan)
                uniquefrac[tp].append(np.nan)
                clusterfracs[tp].append(np.nan)

    # pass information to plotting function
    MSHM = np.nanmean(SHM_means, axis=1)
    SSHM = np.nanstd(SHM_means, axis=1)
    MEntropies = np.nanmean(Entropies, axis=1)
    SEntropies = np.nanstd(Entropies, axis=1)
    Mclusterfracs = np.nanmean(clusterfracs, axis=1)
    Sclusterfracs = np.nanstd(clusterfracs, axis=1)

    sample_statistics_plot(subsample, tList, MSHM, SSHM,
                           MEntropies, SEntropies, Mclusterfracs,
                           Sclusterfracs)

    """ Plot of affinity/mutations on tps I, II and III """
    # sample cells 7 days post each infection, record SHM, KD and origin
    # (memory versus naive first activated ancestor)
    timecourse = [7*12, 35*12, 63*12]
    SHM_list = [[] for t in timecourse]
    KD_list = [[] for t in timecourse]
    orglist = [[] for t in timecourse]

    for d in range(len(timecourse)):
        tp = timecourse[d]
        cellnum = int(np.round(len(freePan[tp].dropna())*samplefrac))
        if cellnum > 0:
            cells = freePan[tp].dropna().sample(cellnum, replace=False)
            kdl = cells.affinity.tolist()
            # transform norm E to KD
            kdll = np.exp(cf.y0+np.array(kdl)*cf.m)
            KD_list[d] = list(kdll)
            # get mutation counts, correct them and origin
            SHM_list[d] = cells.mutations.tolist()
            orglist[d] = cells.origin.tolist()

    # pass information to plot function
    sample_scatter_plot(KD_list, SHM_list, orglist)

    """ Affinity/mutation plots for individual clusters """
    # samples from the memory pool at the given timepoints, split information
    # into clusters and plot SHM/KD scatter plots for some of these clusters.

    # lists to collect SHM, KD values, families and timepoints for all panels
    SHM_list = []
    KD_list = []
    fam_list = []
    tp_list = []

    for d in range(len(timecourse)):
        tp = timecourse[d]
        cellnum = int(len(freePan[tp].dropna())*samplefrac)
        if cellnum > 0:
            cells = freePan[tp].dropna().sample(cellnum, replace=False)
            kdl = cells.affinity.tolist()
            # transform norm E to KD
            kdll = np.exp(cf.y0+np.array(kdl)*cf.m)
            KD_list += list(kdll)
            SHM_list += list(cells.mutations.tolist())
            fam_list += cells.family.tolist()
            tp_list += [tp for k in range(cellnum)]

    # count into families and find clusters with more than xx members
    famcounter = Counter(fam_list)
    fams = famcounter.keys()
    clusters = []
    for fam in fams:
        if famcounter[fam] > 1:
            clusters.append(fam)

    # make separate lists for SHM, KD and TP (defining color) within clusters
    # and add information to list
    iSHMs = [[] for i in clusters]
    iKDs = [[] for i in clusters]
    iTPs = [[] for i in clusters]

    for ff in range(len(fam_list)):
        if fam_list[ff] in clusters:
            ii = clusters.index(fam_list[ff])
            iSHMs[ii].append(SHM_list[ff])
            iKDs[ii].append(KD_list[ff])
            # give different colors for different timepoints
            if tp_list[ff] == timecourse[0]:
                iTPs[ii].append('lightcoral')
            elif tp_list[ff] == timecourse[1]:
                iTPs[ii].append('indianred')
            else:
                iTPs[ii].append('firebrick')
    # pass information to plot function
    clonal_scatter_plot(iSHMs, iKDs, iTPs)

    # write information to file
    if d_export:
        datafile = open('processed_data/TUCHMI_sampling_data', 'w')

        datafile.write('1) SAMPLE STATISTICS \n \n')
        datafile.write('sampled fraction = {} \n \n'.format(samplefrac))
        datafile.write('timecourse (days) \n {} \n \n'.format(np.array(tList)/12.))
        datafile.write('SHMs of cells in sample, mean and std \n {} \n {} \n \n'.format(MSHM, SSHM))
        datafile.write('normalised Shannon entropy of cells in sample, mean and std \n {} \n {} \n \n'.format(MEntropies, SEntropies))
        datafile.write('fraction of non-unique cells in sample, mean and std \n {} \n {} \n \n'.format(Mclusterfracs, Sclusterfracs))

        datafile.close()


def selection_vs_mutation(store_export='dictionary', d_export=True):
    """ Performs a single simulation of a given size using a specified protocol
    of vaccination boosters. At specified timepoints, a specified number of
    memory cells is sampled and the affinities of their ancestors as well as
    their current affinities are written to a list. Also written to list
    are the binding energies of the naive cells. These three lists are then
    passed on to be plotted as distribution histograms.

    Plots produced include a collection of three histograms (unselected,
    selected germline energies, actual energies after mutations) for each
    queried timepoint and a more complex scatter plot with marginal histograms
    for each queried timepoint.

    For each timepoint, the fraction of cells with unaltered/improved/impaired
    affinity is printed to screen.

    If store_export is set 'datafile', the simulation data is stored in a
    hdf5 file for future purposes, for 'dictionary' the data is passed
    internally and lost after the run.

    If d_export is set True, textfiles containing the sampled data (used for
    plotting) are exported for each plot individually.
    """
    # parameters relevant to this analysis
    # evaluation timepoint in days
    analysis_times = [29]
    # prepare lists
    ancestor_dists = []
    final_dists = []
    # get runID from current system time
    runID = int(time.time())
    # run simulation and get filepath or dict
    simdata = main(runID, store_export=store_export, evalperday=1)
    # import required information for small scale plots
    l_times, l_fn, l_fm, l_GCs, LFcurve, Agcurve, evaltimes, freePan, GCPans, \
        ms_times, ms_vals, ms_fams, ms_muts, mut_list, E_list = \
        import_file(simdata)
    # extract the affinities and ancestor affinities at the analysis points
    tList = freePan.keys()
    for i in range(len(analysis_times)):
        # limit cell number to be drawn in order not to clatter the plot
        tp = analysis_times[i]
        cellnum = min(2000, len(freePan[tList[tp]].dropna()))
        cellnum = len(freePan[tList[tp]].dropna())
        cells = freePan[tList[tp]].dropna().sample(cellnum, replace=False)
        afflist = cells['affinity'].dropna().tolist()
        final_dists.append(afflist)
        aff0list = cells['affinity0'].dropna().tolist()
        ancestor_dists.append(aff0list)
    # send energy lists to histogram plot
    for i in range(len(analysis_times)):
        energy_distributions_plot(E_list, ancestor_dists[i], final_dists[i],
                                  analysis_times[i])
        energy_scatter_plot(ancestor_dists[i], final_dists[i],
                            analysis_times[i])

    if d_export:
        datafile = open('processed_data/energy_distribution_data', 'w')

        datafile.write('1) naive distribution \n \n')
        datafile.write('{} \n \n'.format(E_list))

        datafile.write('2) analysis days \n \n')
        datafile.write('{} \n \n'.format(analysis_times))

        datafile.write('3) ancestor distributions per time point \n \n')
        datafile.write('{} \n \n'.format(ancestor_dists))

        datafile.write('4) memory distributions per time point \n \n')
        datafile.write('{} \n \n'.format(final_dists))

        datafile.close()


def stacked_mutations(store_export='dictionary', d_export=True,
                      repeats=10):
    """ Performs a number of simulation runs, computes histograms for
    improved, impaired and unchanged binders at a single given timepoint
    and saves the individual as well as the summed values to file. For several
    repeats, data is accumulated in the histograms as well.

    If store_export is set 'datafile', the simulation data is stored in a
    hdf5 file for future purposes, for 'dictionary' the data is passed
    internally and lost after the run.

    If d_export is set True, textfiles containing the sampled data (used for
    plotting) are exported for each plot individually."""
    # parameters relevant to this analysis
    # evaluation timepoint in days
    analysis_time = 29
    bins = np.linspace(0.6, 1, 17)
    # collect results
    sum_zero = np.zeros(len(bins)-1)
    sum_plus = np.zeros(len(bins)-1)
    sum_minus = np.zeros(len(bins)-1)
    list_zero = []
    list_plus = []
    list_minus = []

    for i in range(repeats):
        # get runID from current system time
        runID = int(time.time())
        # run simulation and get filepath or dict
        simdata = main(runID, store_export=store_export, evalperday=1)
        # import required information for small scale plots
        l_times, l_fn, l_fm, l_GCs, LFcurve, Agcurve, evaltimes, freePan, \
            GCPans, ms_times, ms_vals, ms_fams, ms_muts, mut_list, E_list = \
            import_file(simdata)
        # extract the affinities and ancestor affinities at the analysis points
        tList = freePan.keys()
        # limit cell number to be drawn in order not to clatter the plot
        # possibility of subsampling here
        tp = analysis_time
        cellnum = len(freePan[tList[tp]].dropna())
        cells = freePan[tList[tp]].dropna().sample(cellnum, replace=False)
        final_dist = cells['affinity'].dropna().tolist()
        ancestor_dist = cells['affinity0'].dropna().tolist()

        # extract counts of unchanged, improved and impaired cells
        unchanged_list = np.array(final_dist)[np.where(np.array(
            ancestor_dist) == np.array(final_dist))[0]]
        improved_list = np.array(final_dist)[np.where(np.array(
            ancestor_dist) < np.array(final_dist))[0]]
        impaired_list = np.array(final_dist)[np.where(np.array(
            ancestor_dist) > np.array(final_dist))[0]]

        # make histograms, store information both in list and in sum.
        U_counts, _ = np.histogram(unchanged_list, bins=bins)
        plus_counts, _ = np.histogram(improved_list, bins=bins)
        minus_counts, _ = np.histogram(impaired_list, bins=bins)

        # collect results
        sum_zero += U_counts
        sum_plus += plus_counts
        sum_minus += minus_counts
        list_zero.append(U_counts)
        list_plus.append(plus_counts)
        list_minus.append(minus_counts)
    cellsum = np.sum(sum_zero)+np.sum(sum_plus)+np.sum(sum_minus)

    # plot
    stacked_energy_plot(bins, sum_plus, sum_minus, sum_zero, analysis_time)

    if d_export:
        datafile = open('processed_data/stacked_histogram_data', 'w')

        datafile.write('1) day \n \n')
        datafile.write('{} \n \n'.format(analysis_time))

        datafile.write('2) bins \n \n')
        datafile.write('{} \n \n'.format(bins))

        datafile.write('3) runs \n \n')
        datafile.write('{} \n \n'.format(repeats))

        datafile.write('4) sum of counts with unchanged energies\n \n')
        datafile.write('{} \n \n'.format(sum_zero))

        datafile.write('5) sum of counts with improved energies\n \n')
        datafile.write('{} \n \n'.format(sum_plus))

        datafile.write('6) sum of counts with impaired energies\n \n')
        datafile.write('{} \n \n'.format(sum_minus))

        datafile.write('7) list of counts with unchanged energies\n \n')
        datafile.write('{} \n \n'.format(list_zero))

        datafile.write('8) list of counts with improved energies\n \n')
        datafile.write('{} \n \n'.format(list_plus))

        datafile.write('9) list of counts with impaired energies\n \n')
        datafile.write('{} \n \n'.format(list_minus))

        datafile.write('10) percentage of umutated, improved, impaired \n \n')
        datafile.write('{}, {}, {}'.format(np.sum(sum_zero)/cellsum,
                       np.sum(sum_plus)/cellsum, np.sum(sum_minus)/cellsum))
        datafile.close()


def AM_effect_nkey(nkeys=[1, 5, 10, 15], repeats=100, d_export=True):
    """ Given a list of values for nkey and a number of individual GC reactions
    to be averaged over for each of them, computes and plots the improvement
    within single GCs for one infection. Thus, overwrites parameters giving
    the infection protocol and duration of the simulation as well as setting
    the nubmer of GCs to 1. Other parameters remain untouched. A textfile with
    the computed mean results is exported if d_export==True.
    """

    # set single infection and single GC for this analysis
    cf.endtime = 30*12
    cf.tinf = [0*12]
    cf.dose = [1]
    cf.nGCs = 1
    cf.naive_pool = 1000*1  # size of the naive precursor pool
    cf.memory_pool = 100*1  # size of the initial unspecific memory pool
    # function for calculating mean E_norm from GC panel

    def GC_affinity(GCPan):
        """ Given a GC panel, gets the mean E_norm for each timepoint."""
        energies = []
        tList = GCPan.keys()

        for tp in range(len(tList)):
            energy = GCPan[tList[tp]]['affinity'].dropna().mean()
            energies.append(energy)

        return tList, energies

    topElist = []
    for hs in nkeys:
        # set binding model parameters accordingly
        cf.nkey = hs
        cf.lAg = hs
        cf.lAb = 220 - hs
        eL = []  # list for collecting energies timecurses of all runs
        for r in range(repeats):
            simdata = main(store_export='dictionary', evalperday=12)
            l_times, l_fn, l_fm, l_GCs, LFcurve, Agcurve, evaltimes, freePan, \
                GCPans, ms_times, ms_vals, ms_fams, ms_muts, mut_list, E_list\
                = import_file(simdata)
            tList, energies = GC_affinity(GCPans[0])
            eL.append(energies)

        # calculate mean and std of all runs and plot
        eM = np.nanmean(np.array(eL), axis=0)
        eStd = np.nanstd(np.array(eL), axis=0)
        topElist.append((hs, tList, eM, eStd))

    # write information to file
    if d_export:
        datafile = open('processed_data/AM_effect_data', 'w')
        datafile.write('number of simulation runs per n_key = {} \n'.format(repeats))
        datafile.write('n_key, time (days), mean(normalised energies), std(normalised energies) \n')
        for i in range(len(nkeys)):
            datafile.write('{0}, {1}, {2}, {3}\n \n'.format(topElist[i][0], np.array(topElist[i][1])/12., topElist[i][2], topElist[i][3]))
        datafile.close()
    # plot
    AM_effect_plot(topElist)
