#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 15:37:33 2022

@author: molano
"""

import glob
import os
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
MAIN_FOLDER = '/home/molano/DMS_electro/DataEphys/pre_processed/'


def get_data_file(file, ev_algmt='S', pre_post=[-2, 2], w=0.1):
    """

    Parameters
    ----------
    file : str
        where to get the data.

    Returns
    -------
    None.
    bhv_ss:
     HitHistoryList: [348×1 double]
     CoherenceList: [348×1 double]
      CoherenceVec: [-1 -0.4816 -0.2282 0 0.2282 0.4816 1]
    RewardSideList: [348×1 double]
      StimulusList: [348×1 double]
       EnviroOrder: {348×1 cell}
      ParsedEvents: {348×1 cell}
     Cube_FrameCoh: [7×200×40 double]
     
     EventsTimesLabel
     {'CenterLedOn','PreStimDelay','Stimulus','MovementTime','OutcomeBegin' ,
      'OutcomeEnd','TimeOutBegin','EarlyWithdrawalBegin','EarlyWithdrawalEnd'};
     EventsTimesLabel =  {'LOn','P_S','S','MT','O_B' ,'O_E', 'TO','E_B','E_E'};

    """
    mat = loadmat(file)
    bhv_ss = mat['bhv_ss'][0][0]
    perf = bhv_ss[0]
    stim_vals = bhv_ss[2][0, :]
    coh = stim_vals[bhv_ss[1]-1]
    gt = bhv_ss[3]
    # stim = bhv_ss[4]
    blk = bhv_ss[5][:, 0]
    ctx = np.array(['']*len(blk))
    ctx[blk == 'Switching'] = '2'
    ctx[blk == 'Repetitive'] = '1'
    ev_times_lbl = mat['EventsTimesLabel'][0, :]
    all_evs_times = mat['ev_times_ss']
    spk_times = mat['spk_times_ss']
    ev_times = all_evs_times[:, ev_times_lbl == ev_algmt]
    units = np.unique(spk_times[:, 1])
    bins = np.linspace(pre_post[0], pre_post[1], int(np.diff(pre_post)/w)+1)
    _, ax = plt.subplots(nrows=5, ncols=6)
    ax = ax.flatten()
    for i_un, un in enumerate(units):
        spk_un = spk_times[spk_times[:, 1] == un, 0]
        algn_spks = np.array([spk_un - et for et in ev_times])
        algn_spks = algn_spks[np.logical_and(algn_spks > pre_post[0],
                                             algn_spks < pre_post[1])]
        # indx = np.searchsorted(bins, algn_spks)
        psth = np.histogram(algn_spks, bins)
        ax[i_un].plot(bins[:-1]+w/2, psth[0])
        print(1)
    # evs = bhv_ss[7]
    print(bhv_ss)


if __name__ == '__main__':
    get_data_file(file=MAIN_FOLDER+'/Rat32_ss_26_data_for_python.mat')
    # files = glob.glob('/home/molano/DMS_electro/DataEphys/pre_processed/' +
    #                   '*data_for_py*')

    # rats = np.unique([os.path.basename(x[:x.find('_ss_')]) for x in files])

    # for rt in rats:
    #     r_files = [x for x in files if x.find(rt) != -1]
