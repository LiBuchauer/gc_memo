"""
gc_memo.py contains the actual simulation; the function main()
contains the main simulation loop. If the simulated data is stored
as a file, this ends up in the folder '/raw_data'.
"""
from __future__ import division
import itertools
import math
import random
import time as tm
import numpy as np
import pandas as pd
import scipy.stats
from collections import Counter
import cf
reload(cf)


# ignore divide by 0 error as this happens routinely in Boltzchoice function
# when selcting the last cell, is not if'fed out for efficiency
np.seterr(divide='ignore', invalid='ignore')

"""
1) Functionality for initialising static compartment lists. Free cell pools
are shared between GCs; the GC waiting lists are specific per GC.
Introduce simple B cell class as information container.
"""

# function preparing cell compartment lists: only for the statis compartments
# of free cells and waiting cells, as all the other lists are generated as
# they are passed along


def new_lists():
    """ Returns lists for keeping track of free (outside of GCs) memory and
    naive B cells as well as a list of lists of B cells waiting for surivival
    signals in each GC. """
    free_naives, free_memory = [], []
    GC_waiting = [[] for gc in range(cf.nGCs)]
    return free_naives, free_memory, GC_waiting


class Bcell(object):
    """ B cell objects are the main unit of this simulation. They consist of
    the following information:
        - their key sequence (nkey codons given as number code)
        - their germline ancestors key sequence
        - their current affinity to the antigen sequence
        - their ancestors affinity to the antigen sequence
        - their origin: recruited into this process from naive or pre-existing
            memory pool?
        - their overall mutation count
        - their family ID (for grouping cells into clones)
        - their time of birth (for turnover of naive pool)
        - the most recent time they entered a GC (relevant for discarding
            cells from GC that have competed for survival signals without
            success for too long)
        - the time this cell or, if it was born inside the GC, its ancestor,
            entered the GC - relevant for knowing whether AID is aready acting
            and mutations can happen, as the enzyme first needs to be
            upregulated upon GC entry
        - whether or not affinity maturation (affinity improvement as compared
            to the germline ancestor) is blocked by detrimental mutations in
            the non-key region
    """
    uID = 0  # unique ID counter
    ufamID = 0  # unique family ID counter

    def __init__(self, sequence, sequence0, affinity, affinity0, origin,
                 birthtime, mutations=0, family=None, GCentrytime=None,
                 AIDstart=None, block=False):
        self.ID = self.uID
        Bcell.uID += 1

        if family is None:  # new family ID only if new family created
            self.family = self.ufamID
            Bcell.ufamID += 1
        else:
            self.family = family

        self.sequence = sequence  # current sequence
        self.sequence0 = sequence0  # sequence of the original ancestor

        self.affinity = affinity  # highest affinity
        self.affinity0 = affinity0  # affinity of original ancestor
        self.origin = origin  # was first ancestor naive or unspecific memory?
        self.mutations = mutations  # counter of bp mutations

        self.birthtime = birthtime  # time the family was produced from the BM
        self.GCentrytime = GCentrytime  # time the cell entered the waitlist
        self.AIDstart = AIDstart  # time since ancestor entered GC site
        self.block = block


class Rands(object):
    """ Returns random numbers from [0, 1) in a hopefully efficient way."""
    def __init__(self):
        self.rands = list(np.random.rand(int(cf.R)))

    def getR(self):
        try:
            R = self.rands.pop()
        except IndexError:
            self.rands = list(np.random.rand(int(cf.R)))
            R = self.rands.pop()
        return R


class RandInts(object):
    """ Returns random integers from {0, ..., nGCs-1} in a hopefully
    efficient way."""
    def __init__(self):
        self.rands = list(np.random.randint(cf.nGCs, size=cf.R))

    def getR(self):
        try:
            R = self.rands.pop()
        except IndexError:
            self.rands = list(np.random.randint(cf.nGCs, size=cf.R))
            R = self.rands.pop()
        return R


"""
2) Basic functions related to the binding model: generation of Ab and Ag
sequences and calculation of their interaction energies; enforcing specific
distribution for the unselected repretoire.
"""


def objective_distribution():
    """ Supplies the required shape of the binding energy distribution in an
    unselected repertoire. Assumes a Gaussian shape with hard-coded mean and
    variance for now and uses simulation parameters for determining how many
    naive cells will be needed and thus what the volume of the distribution
    should be.
    Returns the bin boundaries and the required cell number for each of the
    bins. Binwise cell requirements add up to the calculated volume. """
    # calculate the needed number of cells or take max value (above which
    # higher diversity should not have practic effects)
    volume = min(cf.naive_pool*len(cf.tinf), 10**5)
    # get bins in the required energy range, width depending on nkey
    if cf.nkey == 1:
        # for nkey = 1, a lot of small bins may not be occupied, thus choose
        # larger bins
        bin_size_goal = 0.1
        bin_number = max(np.round((cf.upperlim-cf.thr)/bin_size_goal), 1)
        bin_edges = np.linspace(cf.thr, cf.upperlim, bin_number+1)
    else:
        bin_size_goal = 0.025
        bin_number = max(np.round((cf.upperlim-cf.thr)/bin_size_goal), 1)
        bin_edges = np.linspace(cf.thr, cf.upperlim, bin_number+1)
    # for the midpoint of each bin, get Gaussian distribution value for
    # mean=0.5 and std=0.1
    bin_midpoints = bin_edges[:-1] + (bin_edges[1]-bin_edges[0])/2
    gauss_weights = np.exp(-np.power(bin_midpoints - 0.5, 2.) /
                           (2 * np.power(0.1, 2.)))
    # scale so that the sum over the bins contains the required cell number
    norm1 = np.sum(gauss_weights)
    obj_dist = np.floor((volume / norm1) * gauss_weights)
    # give back the objective distribution and bin_edges
    return bin_edges, obj_dist, volume


def make_shaped_repertoire(RNs):
    """ Given an epitopic sequence, queries the objective distribution of
    the unselected binding repertoire and generates sequences according to this
    distribution. Incorporates a cut-off mechanism in case there are bins
    that have not been filled at least once after a given number of tries.
    Returns a list of Ab sequences whose energies comply to the objective
    binding energy distribution."""
    # get objective distribution
    bin_edges, obj_dist, volume = objective_distribution()
    # get an antigenic epitope sequence, and in case of nkey=1,2 check whether
    # it can populate all required bins, thus avoiding infinite loop below
    AgEpitope = get_AgEpitope(RNs)
    if cf.nkey == 1 or cf.nkey == 2:
        while 1:
            # get list of all possible binding partners and their energies
            all_partners = get_all_partners()
            all_energies = [E_best(partner, AgEpitope)
                            for partner in all_partners]
            # check whether all bins are occupiable with these energies,
            # if not, get new epitope sequence
            indices = np.digitize(all_energies, bin_edges, right=True)
            ind_set = set(indices)
            ind_set.discard(0)
            # if all bins can be occupied, move on
            if ind_set == set(range(1, len(bin_edges))):
                break
            # else get a new epitope and check its validity
            else:
                AgEpitope = get_AgEpitope(RNs)
    # initialise empty list for counting how many seqs have been found per bin
    ist_dist = np.zeros(len(obj_dist))
    # seq_list for collecting identified sequences
    seq_list = []
    E_list = []
    # while ist_dist and obj_dist are not equal, get new sequences and position
    # them if they are useful
    # introduce a tolerance of how far bins are allowed to deviate from the
    # goal, as otherwise runtime explodes due to very long waiting times for
    # high binding energy codes in large nkey cases - allow an absolute
    # deviation of volume*tolerance % for each bin.
    abs_tol = volume * 0.005
    while np.sum(np.abs((ist_dist-obj_dist)) > abs_tol) > 0:
        ab = Ab_seq(RNs)
        Emax = E_best(ab, AgEpitope)
        # find index bin of this energy
        indx = np.digitize(Emax, bin_edges, right=True)
        # if the index is in the useful range and the bin is not yet full,
        # count the sequence and store it
        if indx in range(1, len(bin_edges)):
            if obj_dist[indx-1] - ist_dist[indx-1] > 0:
                ist_dist[indx-1] += 1
                seq_list.append(ab)
                E_list.append(Emax)

    return seq_list, E_list, AgEpitope


def get_all_partners():
    """ For a given nkey, generates a list of all possible key sequences and
    returns them, relevant for checking whether all bins for specified
    distributions can be filled by make_shaped_repertoire()."""
    single_position = [i for i in range(1, 21)]
    if cf.nkey == 1:
        return [[i] for i in range(1, 21)]
    else:
        all_positions = [single_position for i in range(cf.nkey)]
        # all permutations
        all_partners = list(itertools.product(*all_positions))
        return all_partners


def Boltzchoice(LFnum, energylist, RNs):
    """ Given a number of selecting limiting factors and a list of qualities
    of the B cells competing, selects LFnum winners based on a Boltzmann
    energy distribution. """
    # transform list to energy values in kT according to experimental
    # affinities and the energy window allowed by the threshold
    energylist = cf.y0 + np.array(energylist) * cf.m
    # calculate norm of initial list
    Norm = sum([math.exp(-ener) for ener in energylist])
    # calculate initial probability vector
    probs = np.array([math.exp(-ener) / Norm for ener in energylist])
    # list to catch indices of selected cells
    selected = []
    # cells to be picked: determined by the lesser of #waiters and #LFs
    cellpick = min(len(energylist), LFnum)
    while len(selected) < cellpick:
        bins = np.cumsum(probs)
        ind = np.digitize(RNs.getR(), bins)
        selected.append(ind)
        # now, set the probability of the selected cell to 0 and renormalise
        # the remaining probability vector
        newNorm = Norm - math.exp(-energylist[ind])
        probs[ind] = 0
        probs = probs * Norm / newNorm
        Norm = newNorm

    return selected


def Ab_seq(RNs):
    """ Creates an Ab CDR seq of length nkey consisting of 20 different
    symbols which are used probabilistically according to the codon number
    leading to each group. """
    seq = []
    for res in range(cf.nkey):
        randi = RNs.getR()
        for i in range(20):
            if randi < cf.cumprob20[i]:
                seq.append(i + 1)  # want amino acids between 1 and 20
                break
    return seq


def Ag_seq(RNs):
    """ Creates an Ag epitope seq of length lAg consisting of 20 different
    symbols which are used probabilistically according to the codon number
    leading to each group. [identical to Ab_seq in this version]
    """
    seq = []
    for res in range(cf.lAg):
        randi = RNs.getR()
        for i in range(20):
            if randi < cf.cumprob20[i]:
                seq.append(i + 1)  # want amino acids between 1 and 20
                break
    return seq


def best_B(Ag):
    """ Given an antigenic determinant Ag this function returns the binding
    value of the best possible binder. """
    top = 0
    for i in range(len(Ag)):
        etop = np.min(cf.TD20[int(Ag[i]) - 1])
        top += etop
    return top


def worst_B(Ag):
    """Given an antigenic determinant Ag this function returns the binding
    value of the best possible binder."""
    bottom = 0
    for i in range(len(Ag)):
        etop = np.max(cf.TD20[int(Ag[i]) - 1])
        bottom += etop
    return bottom


def E_norm(Ab, Ag, top, bottom):
    """ Calculates the normalized binding energy given Ab and Ag sequences,
    binding matrix and the best and worst binding values possible for the
    given Ag (required for normalization). 1 then means the best binder, 0 the
    worst. """

    # calculate binding energy before normalization
    Esum = sum([cf.TD20[int(Ab[i]) - 1][int(Ag[i]) - 1]
                for i in range(len(Ab))])

    # normalize using the supplied top and bottom values
    Enormal = (Esum - bottom) / (top - bottom)

    return Enormal


def E_best(Ab, AgEpitope):
    """ Given an Ab and a prepared list containing the epitope as well
    as its best and worst binding partner energies, this function calculates
    the normalized binding energy of the Ab towards the epitope. It then
    returns the binding energy value. """

    E = E_norm(Ab, AgEpitope[0], AgEpitope[1], AgEpitope[2])

    return E


def get_AgEpitope(RNs):
    """ Returns an epitope sequence together with its best and worst possible
    binding values. Format: [seq,top,bottom]. """
    ag = Ag_seq(RNs)
    top = best_B(ag)
    bottom = worst_B(ag)
    AgEpitope = (ag, top, bottom)

    return AgEpitope


def make_naive(RNs, seq_list, AgEpitope, tnow):
    """ Prepares a naive B cell with a sequence that binds to the epitope
    with above threshold affinity and returns a B cell object. """
    # pick a random sequence from the pregenerated pool
    ab = random.choice(seq_list)
    Emax = E_best(ab, AgEpitope)
    if tnow == 0:   # in initialisation, distribute ages evenly over
                    # lifespan
        birthtime = -np.round(RNs.getR() * cf.tlifeN)
    else:
        birthtime = tnow
    newcell = Bcell(sequence=ab, sequence0=ab, affinity=Emax, affinity0=Emax,
                    origin='naive', mutations=0,
                    family=None, birthtime=birthtime,
                    GCentrytime=None,
                    AIDstart=None, block=False)
    return newcell


def make_memory(RNs, seq_list, AgEpitope, tnow):
    """ Prepares an unspecific memory B cell with a sequence that binds to the
    epitope with above threshold affinity and returns a
    B cell object with a random number of mutations from its past (0-40). """
    ab = random.choice(seq_list)
    Emax = E_best(ab, AgEpitope)
    mutcount = np.round(RNs.getR() * 40)
    newcell = Bcell(sequence=ab, sequence0=ab, affinity=Emax, affinity0=Emax,
                    origin='umem', mutations=mutcount,
                    family=None, birthtime=tnow, GCentrytime=None,
                    AIDstart=None, block=False)
    return newcell


def get_low_binder(RNs, AgEpitope, ntest):
    """ If simulation is run for the purpose of comparing affinity maturation
    with different numbers of hotspots in equally seeded GCs, the following
    function checks out what low quality binders in the range just above 0.6
    can realistically be expected to appear (relevant for low HS case, where
    larger binding energy intervals may not be populated). """
    E_collect = []
    while len(E_collect) < ntest:
        ab = Ab_seq(RNs)
        Emax = E_best(ab, AgEpitope)
        if Emax >= cf.thr:
            E_collect.append(Emax)
    return min(E_collect)


def mutate_seq(seq, block0, RNs):
    """ Mutates a given Ab seq according to the rules for mutations in FWR and
    CDR parts. If no deadly mutation happens to the FWR part, there is a
    possibility of change in the CDR part. """
    sequence = seq
    block = block0
    # get the number of changes in the FWR part and key part
    # for framework part, include the rate of silent mutations (75%), this
    # is not necessary for the explicitly modeled residues as changes there
    # can lead to replacement with the same AA still
    FWR_changes = np.random.binomial(cf.lAb, cf.p_err_FWR*0.75)
    CDR_changes = np.random.binomial(cf.nkey, cf.p_err_CDR)
    if FWR_changes > 0:
        # determine number of deadly muts and blockmuts in the non-death
        # branch (p_death + (1-p_death)*p_block + (1-p_death)*(1-p_block)=1)
        # 0 signifies deathly mutation, 1 signifies blocking mutation
        mutIDs = list(np.random.choice([0, 1, 2],
                                       p=[cf.p_death_FWR,
                                          (1-cf.p_death_FWR) * cf.p_block_FWR,
                                          (1-cf.p_death_FWR) *
                                          (1-cf.p_block_FWR)],
                                          size=FWR_changes))

        if 0 in mutIDs:  # if deadly mutations happen, return no sequence
            return None, 0, 0
        elif 1 in mutIDs:  # if block mutation happens, set block to true
            block = True
    # if the cell has not died yet, analyse mutations in the CDR region
    if CDR_changes > 0:
        # get non-repetitive positions where mutation will be attempted
        changepos = random.sample(range(cf.nkey), CDR_changes)
        for pos in changepos:
            # get transition probabilities for the current amino acid
            cumprob = np.cumsum(cf.tp20[sequence[pos] - 1])
            randi = RNs.getR()
            # find replacement codon
            for i in range(21):  # 20 aa plus stop
                if randi < cumprob[i]:
                    sequence[pos] = i + 1
                    break
        # if stop codon was integrated into the sequence, return 0 as well
        if 21 in sequence:
            return None, 0, 0
    # only mutations of cells that survived are returnd for the counting
    return sequence, FWR_changes, block


def divide(mother, AgEpitope, tnow, mut_list, RNs):
    """ Given a mother cell and the epitope present in the system,
    the function produces between zero and two daughter cells, according to the
    division success and returns them as Bcell objects in a single list."""
    dlist = []
    # get new sequences, additional mutation counts and block status
    # for the daughters; mutations may happen during division ONLY if
    # the cell's family has been in the GC for long enough to have enough AID
    if ((tnow - mother.AIDstart) >= cf.tAID):  # mutations can happen
        seq1, mutcount1, block1 = mutate_seq(mother.sequence[:],
                                             mother.block, RNs)
        seq2, mutcount2, block2 = mutate_seq(mother.sequence[:],
                                             mother.block, RNs)
    else:  # mutational programme is not switched on yet (daughter=mother)
        seq1, mutcount1, block1 = mother.sequence[:], 0, mother.block
        seq2, mutcount2, block2 = mother.sequence[:], 0, mother.block

    num_muts = 0
    num_ben = 0
    # make new Bcell objects if sequences are okay
    if seq1 is not None:
        # if cell is blocked, affinity <= affinity0
        if not block1:
            Emax = E_best(seq1, AgEpitope)
        else:
            Emax = min(E_best(mother.sequence0, AgEpitope),
                       E_best(seq1, AgEpitope))
        daughter1 = Bcell(sequence=seq1, sequence0=mother.sequence0[:],
                          affinity=Emax, affinity0=mother.affinity0,
                          origin=mother.origin,
                          mutations=mother.mutations + mutcount1,
                          family=mother.family, birthtime=mother.birthtime,
                          GCentrytime=tnow,
                          AIDstart=mother.AIDstart, block=block1)
        dlist.append(daughter1)
        # mutation counting
        num_muts += mutcount1
        if Emax > mother.affinity:
            num_ben += 1

    if seq2 is not None:
        # if cell is blocked, affinity <= affinity0
        if not block2:
            Emax = E_best(seq2, AgEpitope)
        else:
            Emax = min(E_best(mother.sequence0, AgEpitope),
                       E_best(seq2, AgEpitope))
        daughter2 = Bcell(sequence=seq2, sequence0=mother.sequence0[:],
                          affinity=Emax, affinity0=mother.affinity0,
                          origin=mother.origin,
                          mutations=mother.mutations + mutcount2,
                          family=mother.family, birthtime=mother.birthtime,
                          GCentrytime=tnow,
                          AIDstart=mother.AIDstart, block=block2)
        dlist.append(daughter2)
        # mutation counting
        num_muts += mutcount2
        if Emax > mother.affinity:
            num_ben += 1

    mut_list.append((tnow, mother.family, num_muts, num_ben))
    del mother
    return dlist, mut_list


def Ag_density():
    """ Returns an array of Ag present in the system at every timestep using
    the initial dose and assuming exponential decay with the decay constant
    supplied. Maximum intial dose is 1 so that all values in this array range
    between 0 and 1. Values below 0.01 are set to 0. Several vaccination
    timepoints are handled by adding their effects and applying a ceiling
    afterwards. """
    # initialise no infection default for the number of infections required
    agcurves = [np.zeros(cf.endtime + 1) for inf in cf.tinf]
    # for every infection, calculate its individual effect per timepoint
    for i in range(len(cf.tinf)):
        pag = cf.dose[i]  # peak
        tai = 0  # tnow after infection
        while pag > 0.01:
            pag = cf.dose[i] * math.exp(-float(tai) / cf.tdecay)
            agcurves[i][cf.tinf[i] + tai] = pag
            tai += 1
            if cf.tinf[i] + tai >= cf.endtime:
                break
    # sum up all effects
    agcurve_uncapped = np.sum(agcurves, axis=0)
    # set all values above 100% to 100%
    agcurve = [np.min([val, 1]) for val in agcurve_uncapped]

    return agcurve


def LF_presence():
    """ Returns the number of available limiting factors per GC based on the
    simplified assumption that a GC forms at full size after the time the
    participating cells need for activation and migrating to the site, stays at
    this maximum until tmax, which is followed by an exponential decay.
    Several vaccination timepoints are handled by adding their effects and
    applying a ceiling afterwards."""
    # initialise no GC default
    LFcurves = [np.zeros(cf.endtime + 1) for inf in cf.tinf]
    # for every infection/vaccination replace the respective timepoints
    for i in range(len(cf.tinf)):
        # assume box shape between tmigration and tmax, exponential decay
        # afterwards
        for j in range(cf.tinf[i] + cf.tmigration,
                       min(cf.tinf[i] + cf.tmax, cf.endtime)):
            LFcurves[i][j] += 1
        # after tmax, the LF number decays exponentially
        for j in range(cf.tinf[i] + cf.tmax, cf.endtime):
            LFcurves[i][j] += math.exp(-float(j - (cf.tinf[i] + cf.tmax))
                                       / cf.LFdecay)
    # sum the individual contributions
    LFcurve_uncapped = np.sum(LFcurves, axis=0)
    # there cannot be more than a maximum number of LFs. Cut down to max.
    LFcurveB = [min(1, val) for val in LFcurve_uncapped]
    # don't allow the reaction to go on if the GC is smaller than 20% of max.
    for i in range(len(LFcurveB)):
        if LFcurveB[i] < 0.2:
            LFcurveB[i] = 0
    # adjust to maximum GC size
    LFcurve = list(np.round(np.array(LFcurveB) * cf.nLFs))

    return LFcurve


"""
3) Actual events of the main simulation loop: cell deaths, cell divisions, GC
entry, cell activation etc.
"""


def old_cells_die(celllist, tnow):
    """ Takes a list of cells, checks their birthtime and evaluates whether
    they live on based on their given lifetime. """
    survivors = [cell for cell in celllist
                 if tnow - cell.birthtime <= cf.tlifeN]
    return survivors


def long_waiters_die(celllist, tnow):
    """ Takes the list of sorted waiting lists and removes cells that have spent
    more than the allowed time period waiting for survival sinal."""
    survivors = []
    for sublist in celllist:
        newsub = []
        for cell in sublist:
            if tnow - cell.GCentrytime <= cf.tlifeGC:
                newsub.append(cell)
        survivors.append(newsub)
    return survivors


def try_activation(Agden, free_naives, free_memory, tnow, RNs):
    """ Given the current antigen density and the lists of free cells,
    the function tries to activate a corresponding number of cells and
    applies an activation probability corresponding to its affinity to
    every one of them. The lists of remaining free cells as well as an event
    of germination to be added to the event_list are returned. """
    activated = []
    fail_naive = []
    fail_memory = []
    # randomize free cell lists
    random.shuffle(free_naives)
    random.shuffle(free_memory)
    # get number of cells to activate from naive list
    act_n = np.random.binomial(len(free_naives), cf.p_base * Agden)
    # get number of memory cells to be activated to enter GC
    act_m = np.random.binomial(len(free_memory), cf.p_base * Agden)
    actsum = act_n + act_m
    # activation of act_n naive cells
    for i in range(int(act_n)):
        cell = free_naives.pop()
        activated.append(cell)

    # activation of act_m memory cells for GC
    for i in range(int(act_m)):
        cell = free_memory.pop()
        activated.append(cell)

    # merge lists to new free pool and create event to be returned
    new_free_naives = free_naives + fail_naive
    new_free_memory = free_memory + fail_memory
    if len(activated) > 0:
        migtime = max(1, cf.tmigration)
        event = (tnow + migtime, 'Enter', None, activated)
    else:
        event = None

    return new_free_naives, new_free_memory, event, actsum


def cells_enter_GCs(GC_waiting, celllist, tnow, RIs):
    """ Distributes the cells waiting to enter a GC at this timepoint randomly
    to the available GCs, supplies a GCentrytime to each of the cells and
    returns the modified GC waitlist. """
    for cell in celllist:
        # get a random GC for entry
        GCpos = RIs.getR()
        # set entrytnow into the waiting area and new position
        cell.GCentrytime = tnow
        cell.AIDstart = tnow
        # add cell to correct waitlist
        GC_waiting[GCpos].append(cell)

    return GC_waiting


def select_best_waiters(LFnum, cellSK, GCpos, tnow, AgEpitope, mut_list, RNs):
    """ Given the current number of available limiting factors and the sorted
    list of waiting cells, this function picks the LFnum cells to be moved on
    according to the Boltzmann choice and distributes them to the
    fates differentiation and division according to the given recycle
    frequency.

    In order to incorporate double cell division after selection, sequences of
    selected cells are directly send through division once to see if in the
    first division round one or two cells survive. The two daughters of this
    first division are then distributed to the fates of either dividing again
    or differentiating and are put to event lists.

    These events are returned with an eventtime according to the help time plus
    the time needed to divide once (first division as discussed above) plus
    another division time or differentiation time according to the chosen fate.
    The waitlist following selection is also returned. """
    # determine the indices of cells to be chosen
    selinds = Boltzchoice(LFnum, [cell.affinity for cell in cellSK], RNs)

    # put selected cells on one list, rest on another
    select = [cellSK[i] for i in range(len(cellSK)) if i in selinds]
    rest = [cellSK[i] for i in range(len(cellSK)) if i not in selinds]

    # divide the selected cells once to have survivors of first division round
    # only, then choose fate for surviving daughters: another division or
    # differeniation. since we are dividing all cells on the list here and they
    # are not added to the waitlist again, pass empty waitlist.
    selected_daughters, mut_list = cell_division([], select, AgEpitope, tnow,
                                                 mut_list, RNs)

    # for these viable daughters, decide how many to divide again and how many
    # to differentiate according to the recycle frequency
    div = np.random.binomial(len(selected_daughters), cf.recycle)
    diff = len(selected_daughters) - div
    # mix daughters (twice, don't trust this function so much)
    random.shuffle(selected_daughters)
    random.shuffle(selected_daughters)
    # make events if count > 0
    new_events = []
    if div > 0:
        event_div = (tnow + cf.thelp + 2*cf.tdiv, 'Divide', GCpos,
                     selected_daughters[:div])
        new_events.append(event_div)
    if diff > 0:
        # get number of cells that will become memory cells, ignore rest (PCs)
        memdiff = np.random.binomial(diff, (1 - cf.PCexport))
        event_diff = (tnow + cf.thelp + cf.tdiv + cf.tdiff, 'Differentiate',
                      GCpos, selected_daughters[div:div + memdiff])
        new_events.append(event_diff)

    return rest, new_events, mut_list


def cell_division(waitlist, celllist, AgEpitope, tnow, mut_list, RNs):
    """ Given the list of cells to be divided and potentially recycled at this
    timepoint, the function produces 0, 1 or 2 daughters from each mother,
    sets their new GC entrytime and adds them to the waitlist of the GC in
    question. The updated waitlist is returned. """
    for cell in celllist:
        # get list of 0 to 2 daughters
        dlist, mut_list = divide(cell, AgEpitope, tnow, mut_list, RNs)
        # add daughters to waitlist
        waitlist = waitlist + dlist
    return waitlist, mut_list


"""
4) Main simulation loop. Major player here is the event list which stores
events that have been chosen to happen in the future (e. g. cells got selected
to divide, but division will not be completed until some time steps later).
Events on this list are of the structure (execution time, type of the event,
GC concerned by this event,  cells oncerned by this event).

"""

# @profile
def main(runID=00, store_export='datafile', evalperday=1):
    """ runID is used to create filename in case of data export.

    store_export can be set to three different options. These are:
        - 'datafile'   : The simulation data is collected in detailed form and
                         ultimately written to a .h5 file
        - 'dictionary' : The simulation data is collected in detailed form and
                         ultimately a dictionary containing it is returned.
        - 'minimal'    : Only summary statistics are evaluated at every time
                         point in evaldays (number of memory cells, mean and
                         std affinity, mean and std mutation count, entropy).
                         Lists of these quantities are returned together with
                         an evaltime time vector. This version is far less
                         memory intensive than the above and recommended for
                         large scale parameter scans, use on clusters, etc.

    evalperday gives the number of time per day that the simulation data
    is written to either the dictionary or the external file. The maximum
    number is 12, as 12 timesteps are simulated per day. Other possible values
    are [1,2,3,4,6,12]. Higher numbers of evalperday mean higher runtimes,
    especially when data is written to file.
    """
    tnow = 0
    tstart = tm.time()

    # get lists for keeping count of cells.
    free_naives, free_memory, GC_waiting = new_lists()

    # get random number objects for uniform 0-1, ints for GCs
    RNs = Rands()
    RIs = RandInts()

    # get the premade pool of Ab sequences that bind a chosen Ag with the given
    # distribution of binding energies. An Ag of appropriate length is made
    # directly within the sequence repertoire function
    seq_list, E_list, AgEpitope = make_shaped_repertoire(RNs)

    # for the required number of naive cells in the system, make Abs and append
    # to free_naive list, same for unspecific memory cells
    for n in xrange(cf.naive_pool):
        newcell = make_naive(RNs, seq_list, AgEpitope, tnow)
        free_naives.append(newcell)

    for n in xrange(cf.memory_pool):
        newcell = make_memory(RNs, seq_list, AgEpitope, tnow)
        free_memory.append(newcell)

    # get Ag level over time
    Agcurve = Ag_density()

    # get available LFs over time
    LFcurve = LF_presence()

    # open event list, event structure: (execution time, type, GC, cell list)
    event_list = []

    # bookkeeping - general
    l_fm = []  # free memory
    mut_list = []  # for collecting all mutations and their effects

    if (store_export == 'datafile' or store_export == 'dictionary'):
        l_fn = []  # free naives
        l_GCs = [[] for i in range(cf.nGCs)]  # cells in each GC
        ms_times = [[] for gc in range(cf.nGCs)]  # times of memory prod./GC
        ms_vals = [[] for gc in range(cf.nGCs)]  # quality of memory prod./GC
        ms_fams = [[] for gc in range(cf.nGCs)]  # family of memory prod./GC
        ms_muts = [[] for gc in range(cf.nGCs)]  # mutations of memory prod./GC
        # external or internal data storage
        if store_export == 'datafile':
            filepath = 'raw_data/store{}.h5'.format(runID)
            store = pd.HDFStore(filepath)
        elif store_export == 'dictionary':
            store = {}
    # bookkeeping - minimal
    l_aff = []  # mean affinities
    s_aff = []  # std of affinities
    l_mut = []  # mean mutation counts
    s_mut = []  # std of mutation counts
    l_ents = []  # family entropies

    # timepoints at which to store the state of the simulation
    evalfac = int(12/evalperday)
    evaltimes = np.array(range(int(cf.endtime / evalfac))) * evalfac

    # start looping over all events at every timestep
    while tnow <= cf.endtime:
        if (store_export == 'datafile' or store_export == 'dictionary'):
            l_fm.append(len(free_memory))
            l_fn.append(len(free_naives))
            for i in range(len(l_GCs)):
                GCcount = len(GC_waiting[i])
                for event in event_list:
                    if (event[1] == 'Differentiate' or event[1] == 'Divide') \
                            and event[2] == i:
                        GCcount += len(event[3])
                l_GCs[i].append(GCcount)

        # remove cells which have died from the naive_pool
        free_naives = old_cells_die(free_naives, tnow)
        # remove cells which have died from the waiting_room
        GC_waiting = long_waiters_die(GC_waiting, tnow)

        # refill the naive_pool if it has fallen below standard size
        # taking care that it is not refilled instantaneously but at a speed
        # of the order of natural turnover (naive_pool/tlifeN)
        maxrefill = np.ceil(cf.naive_pool/cf.tlifeN)
        navcount = 0
        while len(free_naives) < cf.naive_pool and navcount < maxrefill:
            newcell = make_naive(RNs, seq_list, AgEpitope, tnow)
            free_naives.append(newcell)
            navcount += 1

        # execute list specific events if present at this timepoint
        if len(event_list) > 0:
            # check which events happen at this timepoint
            now_list = [event for event in event_list if event[0] == tnow]
            event_list = [event for event in event_list if event[0] != tnow]

            # execute events happening now
            for event in now_list:
                if event[1] == 'Enter':
                    GC_waiting = cells_enter_GCs(GC_waiting, event[3], tnow,
                                                 RIs)
                elif event[1] == 'Divide':
                    GC_waiting[event[2]], mut_list = cell_division(
                        GC_waiting[event[2]], event[3], AgEpitope, tnow,
                        mut_list, RNs)
                elif event[1] == 'Differentiate':
                    free_memory = free_memory + event[3]
                    if (store_export == 'datafile' or
                        store_export == 'dictionary'):
                        for cell in event[3]:
                            ms_times[event[2]].append(tnow)
                            ms_vals[event[2]].append(cell.affinity)
                            ms_fams[event[2]].append(cell.family)
                            ms_muts[event[2]].append(cell.mutations)

        # activate free naive and memory cells if Ag is present in the system
        if Agcurve[tnow] > 0:
            free_naives, free_memory, event, actsum = try_activation(
                Agcurve[tnow], free_naives, free_memory, tnow, RNs)
            if event is not None:
                event_list.append(event)

        # select waiting cells for help signals if LFs are present
        if LFcurve[tnow] > 0:
            # perform selection for every GC separately,
            for i in range(len(GC_waiting)):
                if len(GC_waiting[i]) >= 0:
                    GC_waiting[i], new_events, mut_list = select_best_waiters(
                        LFcurve[tnow], GC_waiting[i], i, tnow, AgEpitope,
                        mut_list, RNs)
                    event_list = event_list + new_events

        # evaluate everything and store results if tnow in evaltimes
        if tnow in evaltimes:
            if (store_export == 'datafile' or store_export == 'dictionary'):
                meminfo = []
                for cell in free_memory:
                    meminfo.append((cell.ID, cell.family, cell.sequence,
                                    cell.affinity, cell.affinity0,
                                    cell.birthtime,
                                    cell.mutations, cell.origin))
                memDF = pd.DataFrame(meminfo, columns=['ID', 'family',
                                                       'sequence', 'affinity',
                                                       'affinity0',
                                                       'birthtime',
                                                       'mutations', 'origin'])
                store['free_{}'.format(tnow)] = memDF

                for i in range(cf.nGCs):
                    GCinfo = []
                    for cell in GC_waiting[i]:
                        GCinfo.append((cell.ID, cell.family, cell.sequence,
                                       cell.affinity, cell.affinity0,
                                       cell.birthtime, cell.mutations))
                    for event in event_list:
                        if (event[1] == 'Differentiate' or
                            event[1] == 'Divide') and event[2] == i:
                            for cell in event[3]:
                                GCinfo.append((cell.ID, cell.family,
                                               cell.sequence,
                                               cell.affinity, cell.affinity0,
                                               cell.birthtime, cell.mutations))
                    GCDF = pd.DataFrame(GCinfo, columns=['ID', 'family',
                                                         'sequence',
                                                         'affinity',
                                                         'affinity0',
                                                         'birthtime',
                                                         'mutations'])
                    store['GC{0}_{1}'.format(i, tnow)] = GCDF
            elif store_export == 'minimal':
                l_fm.append(len(free_memory))
                afflist = [cell.affinity for cell in free_memory]
                mutatlist = [cell.mutations for cell in free_memory]
                familist = [cell.family for cell in free_memory]
                l_aff.append(np.nanmean(afflist))
                s_aff.append(np.nanstd(afflist))
                l_mut.append(np.nanmean(mutatlist))
                s_mut.append(np.nanstd(mutatlist))

                CC = Counter(familist)
                l_ents.append(scipy.stats.entropy(CC.values(), base=2))

        # increment time
        tnow += 1

    tend = tm.time()
    print('pure simulation time = {} s'.format(tend - tstart))

    if (store_export == 'datafile' or store_export == 'dictionary'):
        # put all remaining information into storage
        store['l_times'] = pd.DataFrame(np.arange(cf.endtime+1)/float(12))
        store['l_fn'] = pd.DataFrame(l_fn)
        store['l_fm'] = pd.DataFrame(l_fm)
        for i in range(len(l_GCs)):
            store['l_GCs_{}'.format(i)] = pd.DataFrame(l_GCs[i])
        store['LFcurve'] = pd.DataFrame(LFcurve)
        store['Agcurve'] = pd.DataFrame(Agcurve)
        store['mut_list'] = pd.DataFrame(mut_list)
        store['ms_fams'] = pd.DataFrame(ms_fams)
        store['ms_vals'] = pd.DataFrame(ms_vals)
        store['ms_times'] = pd.DataFrame(ms_times)
        store['ms_muts'] = pd.DataFrame(ms_muts)
        store['times'] = pd.DataFrame(evaltimes)
        store['nGCs'] = pd.DataFrame([cf.nGCs])
        store['E_list'] = pd.DataFrame(E_list)

        if store_export == 'datafile':
            store.close()
            return filepath
        elif store_export == 'dictionary':
            return store

    elif store_export == 'minimal':
        return evaltimes, l_fm, l_aff, s_aff, l_mut, s_mut, l_ents
